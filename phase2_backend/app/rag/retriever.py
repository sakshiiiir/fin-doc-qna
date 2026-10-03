import json
from pathlib import Path

import faiss

from .embeddings import EMBEDDING_DIM, embed_texts

STORAGE_DIR = Path(__file__).resolve().parent.parent.parent / "storage"


def _doc_dir(doc_id: str) -> Path:
    d = STORAGE_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _index_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "index.faiss"


def _chunks_path(doc_id: str) -> Path:
    return _doc_dir(doc_id) / "chunks.json"


def doc_exists(doc_id: str) -> bool:
    return _index_path(doc_id).exists()


def load_or_create_index(doc_id: str) -> faiss.Index:
    path = _index_path(doc_id)
    if path.exists():
        return faiss.read_index(str(path))
    return faiss.IndexFlatIP(EMBEDDING_DIM)


def _load_chunks(doc_id: str) -> list:
    path = _chunks_path(doc_id)
    if path.exists():
        return json.loads(path.read_text())
    return []


def add_chunks(doc_id: str, chunks: list) -> None:
    index = load_or_create_index(doc_id)
    embeddings = embed_texts(chunks)
    index.add(embeddings)
    faiss.write_index(index, str(_index_path(doc_id)))

    existing = _load_chunks(doc_id)
    existing.extend(chunks)
    _chunks_path(doc_id).write_text(json.dumps(existing))


def retrieve(doc_id: str, question: str, top_k: int = 5) -> list:
    if not doc_exists(doc_id):
        return []
    index = load_or_create_index(doc_id)
    if index.ntotal == 0:
        return []
    chunks = _load_chunks(doc_id)
    q_emb = embed_texts([question])
    k = min(top_k, index.ntotal)
    _, idxs = index.search(q_emb, k)
    return [chunks[i] for i in idxs[0] if i != -1]
