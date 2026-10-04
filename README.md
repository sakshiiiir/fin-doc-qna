# fin-doc-qna

Financial document Q&A: upload a 10-K or earnings PDF and chat with it. A LoRA fine-tuned Llama 3.2 1B, a FAISS RAG pipeline, a FastAPI backend and a Streamlit chat UI, with FinanceBench evaluation.

The fine-tuned adapter is on the Hugging Face Hub: [`sakshirathi/finsight-llama-3.2-1b`](https://huggingface.co/sakshirathi/finsight-llama-3.2-1b).

## How it works

```
PDF upload ──► pdfplumber ──► 250-word chunks ──► MiniLM embeddings ──► FAISS index
                                                                            │
question ──► embed ──► top-k chunks ──► prompt ──► Llama 3.2 1B + LoRA ──► answer + sources
```

| Phase | Folder | What it does |
|---|---|---|
| 1. Fine-tuning | `phase1_finetune/` | Generates synthetic Q&A from SEC 10-K filings (Gemini), fine-tunes Llama with LoRA, evaluates on FinanceBench |
| 2. Backend | `phase2_backend/` | FastAPI service: `/upload`, `/query`, `/compare`, `/health` |
| 3. UI | `phase3_ui/` | Streamlit chat: upload a PDF, ask questions, see source chunks |

The training set (`phase1_finetune/data/train.jsonl`) has 1,799 prompt/completion pairs in the format
`### Context: ... ### Question: ... ### Answer:`.

## Quick start

Requires Python 3.10+ and an Apple Silicon Mac (MPS) or CPU. The 1B model loads in about 15 seconds.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r phase2_backend/requirements.txt -r phase3_ui/requirements.txt

./run_dev.sh
```

This starts the backend on `localhost:8000` and the UI on `localhost:8501`. Wait for the sidebar to show
"Backend ready", upload a PDF, and ask questions in the chat box. Backend logs go to
`phase2_backend/logs/backend.log`.

The model repo is public, so no Hugging Face token is needed. If you have one, or other keys, put them in a
gitignored `.env` file (see below). Never commit it.

### Configuration

| Variable | Default | Purpose |
|---|---|---|
| `INFERENCE_BACKEND` | `local` (via `run_dev.sh`) | `local` (transformers + PEFT), `ollama`, `hf_api`, or `mock` (no model, for testing the pipeline) |
| `HF_MODEL_REPO` | `sakshirathi/finsight-llama-3.2-1b` | LoRA adapter repo |
| `HF_BASE_MODEL` | `meta-llama/Llama-3.2-1B` | Base model the adapter is applied to |
| `HF_TOKEN` | – | Optional, for rate limits or private/gated repos |
| `GEMINI_API_KEY` | – | Optional: enables the judge score in `/compare` and the evaluation script |
| `FINSIGHT_API_URL` | `http://localhost:8000` | Where the UI finds the backend |

### API

```
POST /upload    multipart PDF           -> {doc_id, num_chunks, filename}
POST /query     {doc_id, question, top_k, model}  -> {answer, sources, model_used}
POST /compare   {doc_id, question, top_k}         -> {base_answer, finetuned_answer, judge_score, sources}
GET  /health                            -> {status, backend, model_ready, error}
```

`model` is `finetuned` or `base`. In local mode "base" runs the same weights with the LoRA adapter disabled.

## Evaluation

`phase1_finetune/scripts/evaluate.py` runs base vs. fine-tuned on [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench)
(gold evidence passage as context, Gemini as judge, optional W&B logging):

```bash
GEMINI_API_KEY=... python phase1_finetune/scripts/evaluate.py --limit 20
```

**Current result (first 20 of 150 questions, Llama 3.2 1B):**

| Model | Judge accuracy |
|---|---|
| Base | 0.155 |
| Fine-tuned | 0.050 |

The fine-tuned adapter does **not** beat the base model on this sample, and 20 questions is a small, noisy
sample. Both models are weak on multi-step numeric questions, which dominate FinanceBench. Improving the training
data and running the full 150 questions are the obvious next steps.

## Project structure

```
phase1_finetune/
  scripts/        generate_qa.py, train.py, train_mlx.py, evaluate.py
  config/         training_config.yaml
  data/           train.jsonl (+ sample/)
  finsight_colab_train.ipynb
phase2_backend/app/
  main.py         FastAPI app, background model loading
  routes/         upload.py, query.py
  rag/            embeddings.py, retriever.py (FAISS)
  model/          inference.py (local / ollama / hf_api / mock)
phase3_ui/app.py  Streamlit chat UI
run_dev.sh        start backend + UI together
```

## Notes and limitations

- A 1B model is small: it is best at specific factual lookups ("What were total net sales in fiscal 2025?")
  and unreliable on calculations or long summaries. Check answers against the source chunks the UI shows.
- Answers are generated greedily, cut at the first `###`, and trimmed of repeated sentences.
- Model weights are not in this repo; they are hosted on the Hugging Face Hub.
- Not financial advice.
