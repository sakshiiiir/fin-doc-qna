import io
import uuid

import pdfplumber
from fastapi import APIRouter, File, HTTPException, UploadFile

from ..rag.retriever import add_chunks

router = APIRouter()

CHUNK_SIZE = 250
CHUNK_OVERLAP = 25
MIN_CHUNK_CHARS = 50


def _extract_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages:
            text_parts.append(page.extract_text() or "")
    return "\n".join(text_parts)


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list:
    words = text.split()
    if not words:
        return []
    step = max(chunk_size - overlap, 1)
    chunks = []
    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if not chunk_words:
            break
        chunk = " ".join(chunk_words)
        if len(chunk.strip()) >= MIN_CHUNK_CHARS:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported")

    pdf_bytes = file.file.read()
    text = _extract_text(pdf_bytes)
    if not text.strip():
        raise HTTPException(422, "Could not extract any text from the PDF")

    chunks = _chunk_text(text)
    if not chunks:
        raise HTTPException(422, "No usable text chunks extracted from the PDF")

    doc_id = str(uuid.uuid4())
    add_chunks(doc_id, chunks)

    return {"doc_id": doc_id, "num_chunks": len(chunks), "filename": file.filename}
