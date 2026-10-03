#!/usr/bin/env bash
# Start backend (8000) and Streamlit UI (8501) together. Ctrl+C stops both.
# HF_TOKEN is read from the environment or a gitignored .env file, never hardcoded.
set -euo pipefail
cd "$(dirname "$0")"

[ -f .env ] && set -a && . ./.env && set +a

export INFERENCE_BACKEND="${INFERENCE_BACKEND:-local}"
export HF_MODEL_REPO="${HF_MODEL_REPO:-sakshirathi/finsight-llama-3.2-1b}"
export HF_BASE_MODEL="${HF_BASE_MODEL:-meta-llama/Llama-3.2-1B}"
export PYTHONFAULTHANDLER=1

mkdir -p phase2_backend/logs
pkill -f "uvicorn phase2_backend" 2>/dev/null || true

uvicorn phase2_backend.app.main:app --port 8000 >> phase2_backend/logs/backend.log 2>&1 &
BACKEND_PID=$!
trap 'kill $BACKEND_PID 2>/dev/null || true' EXIT

echo "Backend log: phase2_backend/logs/backend.log (model takes ~1 min to load; UI shows status)"
streamlit run phase3_ui/app.py
