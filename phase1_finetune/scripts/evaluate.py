"""
Evaluate base vs. fine-tuned model on FinanceBench, using Gemini as
an LLM-judge (optional), logged to Weights & Biases (optional).

Usage (on Mac — no GPU needed):
    python evaluate.py --limit 20

With judging:
    GEMINI_API_KEY=... python evaluate.py --limit 20

With W&B logging:
    GEMINI_API_KEY=... WANDB_API_KEY=... python evaluate.py --limit 20
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

JUDGE_PROMPT = """You are grading a financial QA model's answer against the ground-truth \
answer for a question about a company's 10-K/earnings filing.

Question: {question}
Ground-truth answer: {gold}
Model answer: {prediction}

Score the model answer's correctness from 0.0 (wrong/irrelevant) to 1.0 (fully correct \
and consistent with the ground truth). Minor phrasing differences are fine if the facts \
match. Respond with ONLY a JSON object: {{"score": <float 0-1>, "reason": "<short reason>"}}
"""

# ── defaults matching the backend setup ─────────────────────────────────
DEFAULT_BASE_MODEL = "meta-llama/Llama-3.2-1B"
DEFAULT_ADAPTER = "sakshirathi/finsight-llama-3.2-1b"


def load_financebench(limit: int):
    from datasets import load_dataset

    ds = load_dataset("PatronusAI/financebench", split="train")
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def build_prompt(context: str, question: str) -> str:
    # Cap context to ~900 words to match the backend's prompt length
    words = context.split()
    if len(words) > 900:
        context = " ".join(words[:900])
    return f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:"


def load_model(base_model_name: str, adapter_repo: str = None):
    """Load the model on MPS/CPU with optional LoRA adapter. Returns (model, tokenizer, device)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = os.environ.get("HF_TOKEN") or None
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"  Loading base model {base_model_name} on {device}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name, torch_dtype=torch.float16, token=token
    )

    if adapter_repo:
        from peft import PeftModel

        print(f"  Applying adapter {adapter_repo}...")
        model = PeftModel.from_pretrained(model, adapter_repo, token=token)

    model = model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        adapter_repo or base_model_name, token=token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer, device


def generate_answer(model, tokenizer, device, prompt: str, max_tokens: int = 160, use_adapter: bool = True) -> str:
    import torch

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        if not use_adapter and hasattr(model, "disable_adapter"):
            with model.disable_adapter():
                out = model.generate(
                    **inputs, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
        else:
            out = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    # Trim after ### and drop repeated sentences
    text = text.split("###")[0].strip()
    seen, kept = set(), []
    for s in re.split(r"(?<=[.!?])\s+", text):
        key = s.strip().lower()
        if key in seen:
            break
        seen.add(key)
        kept.append(s)
    return " ".join(kept)


def judge_score(judge_model_tuple, question: str, gold: str, prediction: str, max_retries: int = 3):
    if judge_model_tuple is None:
        return None, "no_judge"
    client, model_name = judge_model_tuple
    prompt = JUDGE_PROMPT.format(question=question, gold=gold, prediction=prediction)
    delay = 2
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.interactions.create(model=model_name, input=prompt)
            raw = re.sub(r"^```(json)?|```$", "", resp.output_text.strip(), flags=re.MULTILINE).strip()
            data = json.loads(raw)
            return float(data["score"]), data.get("reason", "")
        except Exception as e:
            if attempt == max_retries:
                print(f"  [judge] giving up after {max_retries} attempts: {e}")
                return None, "judge_error"
            time.sleep(delay)
            delay *= 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", default=DEFAULT_ADAPTER)
    parser.add_argument("--limit", type=int, default=20, help="number of FinanceBench questions (150 total, ~45s each)")
    parser.add_argument("--gemini-judge-model", default="gemini-3.5-flash-lite")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "data" / "eval_results.jsonl"))
    args = parser.parse_args()

    # Judge (optional)
    judge_model = None
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        from google import genai
        client = genai.Client(api_key=api_key)
        judge_model = (client, args.gemini_judge_model)
        try:
            client.interactions.create(model=args.gemini_judge_model, input="Reply with OK")
        except Exception as e:
            parser.error(f"Gemini judge check failed for {args.gemini_judge_model}: {e}")
        print(f"[eval] Gemini judge enabled ({args.gemini_judge_model})")
    else:
        print("[eval] No GEMINI_API_KEY — skipping scoring, will just output answers")

    # W&B (optional)
    use_wandb = False
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        wandb.init(project="finsight-llama-qlora", name="financebench-eval", job_type="eval")
        use_wandb = True

    print("[eval] Loading FinanceBench...")
    bench = load_financebench(args.limit)

    print("[eval] Loading model (base + adapter)...")
    model, tokenizer, device = load_model(args.base_model, args.adapter)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_scores, ft_scores = [], []

    with output_path.open("w") as out_f:
        for i, row in enumerate(bench):
            question = row["question"]
            gold = row.get("answer", "")

            # FinanceBench provides evidence as a list of dicts with 'evidence_text'
            evidence = row.get("evidence") or []
            if isinstance(evidence, list):
                context = "\n\n".join(e.get("evidence_text", "") for e in evidence if isinstance(e, dict))
            else:
                context = str(evidence)

            prompt = build_prompt(context, question)

            print(f"[{i + 1}/{len(bench)}] {question[:80]}...")

            # Base answer (adapter off)
            base_answer = generate_answer(model, tokenizer, device, prompt, use_adapter=False)
            # Fine-tuned answer (adapter on)
            ft_answer = generate_answer(model, tokenizer, device, prompt, use_adapter=True)

            base_sc, base_reason = judge_score(judge_model, question, gold, base_answer)
            ft_sc, ft_reason = judge_score(judge_model, question, gold, ft_answer)

            record = {
                "question": question,
                "gold": gold,
                "base_answer": base_answer,
                "base_score": base_sc,
                "finetuned_answer": ft_answer,
                "finetuned_score": ft_sc,
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

            if base_sc is not None and ft_sc is not None:
                base_scores.append(base_sc)
                ft_scores.append(ft_sc)
                print(f"  base={base_sc:.2f}  ft={ft_sc:.2f}  gold={gold[:60]}")
            else:
                print(f"  base: {base_answer[:80]}")
                print(f"  ft:   {ft_answer[:80]}")
                print(f"  gold: {gold[:80]}")

            if use_wandb:
                import wandb
                log = {"step": i}
                if base_sc is not None:
                    log.update({"base_score": base_sc, "finetuned_score": ft_sc})
                wandb.log(log)

    # Summary
    if base_scores:
        base_avg = sum(base_scores) / len(base_scores)
        ft_avg = sum(ft_scores) / len(ft_scores)
        delta = ft_avg - base_avg
        print(f"\n{'='*50}")
        print(f"Base accuracy:       {base_avg:.3f}")
        print(f"Fine-tuned accuracy: {ft_avg:.3f}")
        print(f"Delta:               {delta:+.3f}")
        print(f"{'='*50}")

        if use_wandb:
            import wandb
            wandb.summary["base_accuracy"] = base_avg
            wandb.summary["finetuned_accuracy"] = ft_avg
            wandb.summary["accuracy_delta"] = delta
    else:
        print("\n[summary] No scores (no GEMINI_API_KEY). Review answers in the output file.")

    if use_wandb:
        import wandb
        wandb.finish()

    print(f"[done] Results written to {output_path}")


if __name__ == "__main__":
    main()
