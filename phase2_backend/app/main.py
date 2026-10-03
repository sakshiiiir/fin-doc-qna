import os

# Must be set before torch / faiss are imported: both ship an OpenMP runtime on
# macOS and loading the two in one process can segfault.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import faulthandler
import threading

from fastapi import FastAPI

from .model import inference
from .routes import query, upload

faulthandler.enable()

app = FastAPI(title="FinSight API")
app.include_router(upload.router)
app.include_router(query.router)


@app.on_event("startup")
def preload_model():
    # Load in the background so the port opens immediately; /health reports readiness
    # and /query returns 503 until the model is loaded.
    if inference.BACKEND == "local":
        threading.Thread(target=inference.load_local_model, daemon=True).start()


@app.get("/health")
def health():
    return {"status": "ok", **inference.health_check()}
