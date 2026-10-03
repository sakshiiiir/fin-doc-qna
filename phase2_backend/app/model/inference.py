"""
Model inference client. Backend is selectable via INFERENCE_BACKEND so the
same API works with local Ollama today and swaps to the real fine-tuned
model (HF Inference API / vLLM) once phase 1 produces one.
"""

import json
import os
import re
import threading

import requests

BACKEND = os.environ.get("INFERENCE_BACKEND", "ollama")  # ollama | hf_api | local | mock
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_BASE_MODEL = os.environ.get("OLLAMA_BASE_MODEL", "llama3.1")
OLLAMA_FINETUNED_MODEL = os.environ.get("OLLAMA_FINETUNED_MODEL", OLLAMA_BASE_MODEL)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

JUDGE_PROMPT = """Question: {question}
Answer A (base model): {base_answer}
Answer B (fine-tuned model): {finetuned_answer}

Which answer is more accurate and helpful given the question? Respond with ONLY \
a JSON object: {{"preferred": "A" or "B" or "tie", "score": <float 0-1, where 1.0 \
means B is much better than A, 0.0 means A is much better, 0.5 means tie>}}
"""


def _ollama_model_name(variant: str) -> str:
    return OLLAMA_FINETUNED_MODEL if variant == "finetuned" else OLLAMA_BASE_MODEL


def _generate_ollama(prompt: str, model: str, max_tokens: int) -> str:
    resp = requests.post(
        f"{OLLAMA_HOST}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"num_predict": max_tokens},
        },
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json()["response"].strip()


def _generate_hf_api(prompt: str, max_tokens: int) -> str:
    if not HF_MODEL_REPO:
        raise RuntimeError("HF_MODEL_REPO must be set when INFERENCE_BACKEND=hf_api")
    resp = requests.post(
        f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_REPO}",
        headers={"Authorization": f"Bearer {HF_TOKEN}"},
        json={"inputs": prompt, "parameters": {"max_new_tokens": max_tokens, "return_full_text": False}},
        timeout=180,
    )
    resp.raise_for_status()
    data = resp.json()
    if isinstance(data, list):
        return data[0].get("generated_text", "").strip()
    return str(data).strip()


class ModelNotReady(RuntimeError):
    """Raised when the local model is still loading (or failed to load)."""


# Override base model if the adapter's base_model_name_or_path is a quantized/unavailable model
HF_BASE_MODEL = os.environ.get("HF_BASE_MODEL", "")

_local_model = None
_local_tokenizer = None
_local_device = None
_load_error = None
_load_lock = threading.Lock()
# One generation at a time: concurrent MPS use from several threadpool threads is unsafe.
_gen_lock = threading.Lock()


def load_local_model() -> None:
    """Load base model + LoRA adapter once. Safe to call repeatedly / from a thread."""
    global _local_model, _local_tokenizer, _local_device, _load_error
    with _load_lock:
        if _local_model is not None:
            return
        try:
            import torch
            from peft import PeftConfig, PeftModel
            from transformers import AutoModelForCausalLM, AutoTokenizer

            if not HF_MODEL_REPO:
                raise RuntimeError("HF_MODEL_REPO must be set when INFERENCE_BACKEND=local")

            print("Loading local model...", flush=True)
            token = HF_TOKEN or None
            peft_config = PeftConfig.from_pretrained(HF_MODEL_REPO, token=token)
            base_model_name = HF_BASE_MODEL or peft_config.base_model_name_or_path
            device = "mps" if torch.backends.mps.is_available() else "cpu"

            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name, torch_dtype=torch.float16, token=token
            )
            model = PeftModel.from_pretrained(base_model, HF_MODEL_REPO, token=token)
            model = model.to(device).eval()

            tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_REPO, token=token)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            _local_model, _local_tokenizer, _local_device = model, tokenizer, device
            _load_error = None
            print(f"Model ready on {device}.", flush=True)
        except Exception as e:  # surfaced via /health instead of killing the server
            _load_error = f"{type(e).__name__}: {e}"
            print(f"Model load failed: {_load_error}", flush=True)


def is_ready() -> bool:
    return BACKEND != "local" or _local_model is not None


def _trim_repeats(text: str) -> str:
    """Greedy decoding on a 1B model can loop; stop at the first repeated sentence.
    (A repetition penalty would also penalise numbers copied from the context.)"""
    seen, kept = set(), []
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        key = sentence.strip().lower()
        if key in seen:
            break
        seen.add(key)
        kept.append(sentence)
    return " ".join(kept)


def _clean_answer(text: str) -> str:
    text = re.sub(r"\s*[A-Z][A-Za-z .,]*\|\s*\d{4} Form 10-K\s*\|\s*\d+", "", text)  # page footer
    text = _trim_repeats(text)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) > 1 and not text.rstrip().endswith((".", "!", "?")):
        sentences = sentences[:-1]  # last sentence was cut off by the token limit
    return " ".join(sentences).strip()


def _generate_local(prompt: str, max_tokens: int, variant: str) -> str:
    if _local_model is None:
        if _load_error:
            raise ModelNotReady(f"Model failed to load: {_load_error}")
        raise ModelNotReady("Model is still loading, try again in a minute")

    import torch

    with _gen_lock:
        inputs = _local_tokenizer(prompt, return_tensors="pt").to(_local_device)
        gen_kwargs = dict(
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=_local_tokenizer.pad_token_id,
        )
        with torch.inference_mode():
            if variant == "base":
                with _local_model.disable_adapter():
                    out = _local_model.generate(**inputs, **gen_kwargs)
            else:
                out = _local_model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = _local_tokenizer.decode(new_tokens, skip_special_tokens=True)
    # The model tends to continue into the next "### Question:" block; keep only the answer.
    return _clean_answer(text.split("###")[0].strip())


def generate(prompt: str, variant: str = "finetuned", max_tokens: int = 160) -> str:
    if BACKEND == "ollama":
        return _generate_ollama(prompt, _ollama_model_name(variant), max_tokens)
    if BACKEND == "hf_api":
        return _generate_hf_api(prompt, max_tokens)
    if BACKEND == "local":
        return _generate_local(prompt, max_tokens, variant)
    if BACKEND == "mock":
        return f"[MOCK:{variant}] {prompt[-200:]}"
    raise ValueError(f"Unknown INFERENCE_BACKEND: {BACKEND}")


def describe(variant: str) -> str:
    if BACKEND == "ollama":
        return f"ollama:{_ollama_model_name(variant)}"
    if BACKEND == "hf_api":
        return f"hf_api:{HF_MODEL_REPO}"
    if BACKEND == "local":
        return f"local:{HF_MODEL_REPO}" + ("" if variant == "finetuned" else " (adapter off)")
    return f"mock:{variant}"


def judge(question: str, base_answer: str, finetuned_answer: str):
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        prompt = JUDGE_PROMPT.format(
            question=question, base_answer=base_answer, finetuned_answer=finetuned_answer
        )
        resp = client.interactions.create(model="gemini-3.5-flash-lite", input=prompt)
        raw = re.sub(r"^```(json)?|```$", "", resp.output_text.strip(), flags=re.MULTILINE).strip()
        return json.loads(raw)
    except Exception:
        return None


def health_check() -> dict:
    if BACKEND == "ollama":
        try:
            resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            return {"backend": "ollama", "reachable": True, "models": models}
        except Exception as e:
            return {"backend": "ollama", "reachable": False, "error": str(e)}
    if BACKEND == "local":
        return {"backend": "local", "reachable": True, "model_ready": is_ready(), "error": _load_error}
    return {"backend": BACKEND, "reachable": True, "model_ready": True}
