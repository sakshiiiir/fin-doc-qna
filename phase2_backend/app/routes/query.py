from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..model import inference
from ..rag.retriever import doc_exists, retrieve

router = APIRouter()


class QueryRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 3
    model: str = "finetuned"  # "base" | "finetuned"


class CompareRequest(BaseModel):
    doc_id: str
    question: str
    top_k: int = 3


MAX_CONTEXT_WORDS = 900  # keeps the prompt near the 1024-token length the model was tuned on


def _build_prompt(context_chunks: list, question: str) -> str:
    context = "\n\n".join(context_chunks)
    context = " ".join(context.split(" ")[:MAX_CONTEXT_WORDS])
    return f"### Context:\n{context}\n\n### Question:\n{question}\n\n### Answer:"


def _get_chunks_or_404(doc_id: str, question: str, top_k: int) -> list:
    if not doc_exists(doc_id):
        raise HTTPException(404, f"Unknown doc_id: {doc_id}")
    chunks = retrieve(doc_id, question, top_k=top_k)
    if not chunks:
        raise HTTPException(404, "No indexed content found for this document")
    return chunks


def _generate(prompt: str, variant: str) -> str:
    try:
        return inference.generate(prompt, variant=variant)
    except inference.ModelNotReady as e:
        raise HTTPException(503, str(e))


@router.post("/query")
def query_document(req: QueryRequest):
    chunks = _get_chunks_or_404(req.doc_id, req.question, req.top_k)
    prompt = _build_prompt(chunks, req.question)
    answer = _generate(prompt, req.model)

    return {
        "answer": answer,
        "sources": chunks,
        "model_used": inference.describe(req.model),
    }


@router.post("/compare")
def compare_models(req: CompareRequest):
    chunks = _get_chunks_or_404(req.doc_id, req.question, req.top_k)
    prompt = _build_prompt(chunks, req.question)

    base_answer = _generate(prompt, "base")
    finetuned_answer = _generate(prompt, "finetuned")
    judge_score = inference.judge(req.question, base_answer, finetuned_answer)

    return {
        "base_answer": base_answer,
        "finetuned_answer": finetuned_answer,
        "judge_score": judge_score,
        "sources": chunks,
    }
