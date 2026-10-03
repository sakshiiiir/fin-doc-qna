# FinSight — Financial Document QA with Domain Fine-Tuned LLM

A financial document Q&A product built on a QLoRA fine-tuned Llama 3.1 8B model, RAG pipeline, FastAPI backend, and Streamlit UI. Users can upload any 10-K/earnings PDF and ask natural language questions grounded in source documents.

---

## Project Structure

```
finsight/
├── phase1_finetune/
│   ├── data/                  # Raw SEC filings + generated Q&A pairs
│   ├── scripts/
│   │   ├── generate_qa.py     # Synthetic Q&A generation from filings
│   │   ├── train.py           # QLoRA fine-tuning script
│   │   └── evaluate.py        # FinanceBench evaluation + W&B logging
│   └── config/
│       └── training_config.yaml
├── phase2_backend/
│   ├── app/
│   │   ├── main.py            # FastAPI entrypoint
│   │   ├── routes/
│   │   │   ├── upload.py      # PDF upload + chunking
│   │   │   └── query.py       # Question answering endpoint
│   │   ├── rag/
│   │   │   ├── embeddings.py  # Embedding + FAISS index
│   │   │   └── retriever.py   # Chunk retrieval logic
│   │   └── model/
│   │       └── inference.py   # Fine-tuned model loader (vLLM / HF)
│   ├── Dockerfile
│   └── requirements.txt
├── phase3_ui/
│   ├── app.py                 # Streamlit UI
│   └── requirements.txt
└── CLAUDE.md
```

---

## Phase 1 — Fine-Tuning

**Goal:** Fine-tune Llama 3.1 8B on financial Q&A data using QLoRA, evaluate on FinanceBench, push to Hugging Face Hub.

### Stack
- Model: `meta-llama/Meta-Llama-3.1-8B`
- Fine-tuning: `trl`, `peft`, `bitsandbytes` (QLoRA 4-bit)
- Tracking: Weights & Biases (`wandb`)
- Eval dataset: [FinanceBench](https://huggingface.co/datasets/PatronusAI/financebench)
- Training data: Synthetic Q&A generated from SEC 10-K filings via Gemini API
- Hardware: Kaggle (2x T4, free) or Modal (A100, free credits) for training; M2 Mac for local dev/testing

### Step 1 — Dataset Prep

```python
# generate_qa.py
# 1. Download 10-K filings from SEC EDGAR (use sec-edgar-downloader)
# 2. Parse PDFs -> text chunks (use PyMuPDF or pdfplumber)
# 3. For each chunk, call Gemini API to generate 3-5 Q&A pairs
# 4. Save as JSONL: {"prompt": "...", "completion": "..."}
# Output: data/train.jsonl (~2000-5000 pairs)
```

**Target format (instruction-tuned):**
```json
{
  "prompt": "### Context:\n{filing_chunk}\n\n### Question:\n{question}\n\n### Answer:",
  "completion": "{answer}"
}
```

### Step 2 — Training Config

```yaml
# config/training_config.yaml
model_name: meta-llama/Meta-Llama-3.1-8B
dataset_path: data/train.jsonl

qlora:
  r: 16
  lora_alpha: 32
  lora_dropout: 0.05
  target_modules: [q_proj, v_proj]

training:
  num_epochs: 3
  batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 2e-4
  max_seq_length: 1024
  bf16: true

output:
  model_dir: ./finsight-llama-qlora
  hub_repo: your-hf-username/finsight-llama-3.1-8b
```

### Step 3 — Training Script

```python
# train.py — key dependencies
# pip install trl peft bitsandbytes transformers accelerate wandb

from trl import SFTTrainer
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# Load in 4-bit
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

# Apply LoRA
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"])

# Train with SFTTrainer (handles formatting, padding, etc.)
```

### Step 4 — Evaluate on FinanceBench

```python
# evaluate.py
# 1. Load FinanceBench questions
# 2. Run base Llama answers -> log to W&B
# 3. Run fine-tuned model answers -> log to W&B
# 4. Use LLM-as-a-judge (Gemini) to score correctness 0-1
# 5. Report delta: fine-tuned vs base accuracy
```

### Step 5 — Push to Hub

```python
model.push_to_hub("your-hf-username/finsight-llama-3.1-8b")
tokenizer.push_to_hub("your-hf-username/finsight-llama-3.1-8b")
```

### Local Dev on M2 Mac

```bash
# Install MLX for Apple Silicon
pip install mlx-lm

# Test pipeline locally with tiny dataset (100 samples)
mlx_lm.lora --model meta-llama/Meta-Llama-3.1-8B --train --data data/sample/

# Local inference testing
brew install ollama
ollama run llama3.1
```

### Done When
- [ ] ~2000+ synthetic Q&A pairs generated from SEC filings
- [ ] Training run completes without OOM errors
- [ ] Fine-tuned model scores higher than base on FinanceBench (record exact %)
- [ ] W&B run logged with loss curves + eval metrics
- [ ] Model pushed to Hugging Face Hub

---

## Phase 2 — Backend

**Goal:** FastAPI service that accepts PDF uploads, builds a FAISS index, and answers questions using the fine-tuned model + RAG.

### Stack
- API: `FastAPI`
- PDF parsing: `pdfplumber` or `PyMuPDF`
- Embeddings: `sentence-transformers` (all-MiniLM-L6-v2)
- Vector store: `FAISS`
- Model inference: Hugging Face Inference API (free tier) or `vLLM` on Modal
- Deploy: Modal or Hugging Face Spaces

### Endpoints

```
POST /upload          — Upload PDF, parse + chunk + embed, return doc_id
POST /query           — {doc_id, question} -> {answer, sources, model_used}
GET  /compare         — {doc_id, question} -> {base_answer, finetuned_answer, judge_score}
GET  /health          — Health check
```

### RAG Pipeline

```
PDF Upload
  → Parse text (pdfplumber)
  → Chunk (500 tokens, 50 overlap)
  → Embed chunks (sentence-transformers)
  → Store in FAISS index (keyed by doc_id)

Query
  → Embed question
  → Retrieve top-k chunks from FAISS
  → Build prompt: [system] + [context chunks] + [question]
  → Run fine-tuned Llama
  → Return answer + source chunk references
```

### Key Files

```python
# app/main.py
from fastapi import FastAPI
from app.routes import upload, query
app = FastAPI(title="FinSight API")
app.include_router(upload.router)
app.include_router(query.router)

# app/rag/retriever.py
# - load_or_create_index(doc_id)
# - add_chunks(doc_id, chunks)
# - retrieve(doc_id, question, top_k=5)

# app/model/inference.py
# - load_model()  # HF pipeline or vLLM client
# - generate(prompt, max_tokens=512)
```

### Environment Variables

```bash
HF_TOKEN=              # Hugging Face token (for gated model access)
HF_MODEL_REPO=         # your-hf-username/finsight-llama-3.1-8b
WANDB_API_KEY=         # for eval logging
```

### Deploy

```bash
# Modal (recommended)
pip install modal
modal deploy app/main.py

# Or Docker
docker build -t finsight-api .
docker run -p 8000:8000 finsight-api
```

### Done When
- [ ] `/upload` successfully parses a real 10-K PDF and builds FAISS index
- [ ] `/query` returns grounded answers with source chunk references
- [ ] `/compare` shows measurable quality difference between base and fine-tuned model
- [ ] API deployed with public URL

---

## Phase 3 — UI

**Goal:** Streamlit app where users upload a financial PDF, ask questions, and see answers with source grounding and model comparison.

### Stack
- UI: `Streamlit`
- Deploy: Streamlit Community Cloud (free)
- Calls: FinSight FastAPI backend (Phase 2)

### App Layout

```
[Header] FinSight — Ask your financial documents

[Sidebar]
  - Upload PDF (10-K, earnings report)
  - Model selector: Base Llama vs Fine-Tuned FinSight
  - Top-k chunks slider

[Main]
  - Question input box
  - [Ask] button
  - Answer panel
  - Source chunks (expandable, highlighted relevant text)
  - [Compare Models] toggle -> side-by-side base vs fine-tuned answers
```

### Key Streamlit Code Pattern

```python
# app.py
import streamlit as st
import requests

API_URL = "https://your-modal-app.modal.run"

st.title("FinSight — Financial Document QA")

uploaded_file = st.file_uploader("Upload a 10-K or earnings PDF", type="pdf")
if uploaded_file:
    res = requests.post(f"{API_URL}/upload", files={"file": uploaded_file})
    doc_id = res.json()["doc_id"]
    st.success("Document indexed!")

    question = st.text_input("Ask a question about this filing")
    if st.button("Ask") and question:
        res = requests.post(f"{API_URL}/query", json={"doc_id": doc_id, "question": question})
        data = res.json()
        st.write("**Answer:**", data["answer"])
        with st.expander("Source chunks"):
            for chunk in data["sources"]:
                st.markdown(f"> {chunk}")
```

### Done When
- [ ] User can upload any financial PDF and get answers
- [ ] Source chunks displayed with each answer
- [ ] Side-by-side model comparison works
- [ ] Deployed on Streamlit Community Cloud with public URL

---

## Resume Bullet (write this after completion)

> Built FinSight, a financial document QA product — generated 2000+ synthetic Q&A pairs from SEC 10-K filings, fine-tuned Llama 3.1 8B with QLoRA (HF TRL + PEFT), improving FinanceBench accuracy by X% over base model (tracked via W&B + LLM-as-a-judge eval); served via FastAPI + FAISS RAG pipeline deployed on Modal, with Streamlit frontend on HF Spaces.

---

## Commands Reference

```bash
# Phase 1 — Fine-tuning
pip install trl peft bitsandbytes transformers accelerate wandb sec-edgar-downloader
python phase1_finetune/scripts/generate_qa.py
python phase1_finetune/scripts/train.py
python phase1_finetune/scripts/evaluate.py

# Phase 2 — Backend
pip install fastapi uvicorn pdfplumber sentence-transformers faiss-cpu
uvicorn phase2_backend.app.main:app --reload --port 8000

# Phase 3 — UI
pip install streamlit requests
streamlit run phase3_ui/app.py
```
