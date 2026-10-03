"""
LoRA fine-tune Llama 3.1 8B on Apple Silicon using MLX.

Equivalent to train.py but uses MLX instead of CUDA/bitsandbytes,
so it runs locally on M-series Macs.

Usage:
    export HF_TOKEN=...
    python train_mlx.py --config ../config/training_config.yaml
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def prepare_data(dataset_path: str, val_split: float, seed: int):
    """Split train.jsonl into MLX-expected train.jsonl + valid.jsonl in a data dir."""
    import random

    records = []
    with open(dataset_path) as f:
        for line in f:
            rec = json.loads(line)
            records.append({"prompt": rec["prompt"], "completion": rec["completion"]})

    random.seed(seed)
    random.shuffle(records)

    split_idx = int(len(records) * (1 - val_split))
    train_records = records[:split_idx]
    valid_records = records[split_idx:]

    data_dir = Path(dataset_path).parent / "mlx_data"
    data_dir.mkdir(exist_ok=True)

    for name, recs in [("train.jsonl", train_records), ("valid.jsonl", valid_records)]:
        with open(data_dir / name, "w") as f:
            for r in recs:
                # MLX "text" format: single field with full prompt+completion
                text = r["prompt"] + " " + r["completion"] + "</s>"
                f.write(json.dumps({"text": text}) + "\n")

    print(f"[data] {len(train_records)} train, {len(valid_records)} valid -> {data_dir}")
    return str(data_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config" / "training_config.yaml"))
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not os.environ.get("HF_TOKEN"):
        parser.error("HF_TOKEN environment variable is required (Llama 3.1 is gated)")

    data_dir = prepare_data(
        dataset_path=str(Path(__file__).resolve().parent.parent / cfg["dataset_path"]),
        val_split=cfg.get("val_split", 0.05),
        seed=cfg.get("seed", 42),
    )

    train_cfg = cfg["training"]
    adapter_path = str(Path(__file__).resolve().parent.parent / cfg["output"]["model_dir"] / "mlx-adapters")
    os.makedirs(adapter_path, exist_ok=True)

    num_iters = int(
        len(open(Path(data_dir) / "train.jsonl").readlines())
        / 1  # batch_size=1 for memory safety
        * train_cfg["num_epochs"]
    )

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", cfg["model_name"],
        "--data", data_dir,
        "--train",
        "--fine-tune-type", "lora",
        "--num-layers", "8",
        "--batch-size", "1",
        "--iters", str(num_iters),
        "--learning-rate", str(float(train_cfg["learning_rate"])),
        "--steps-per-report", str(train_cfg.get("logging_steps", 10)),
        "--steps-per-eval", str(train_cfg.get("eval_steps", 50)),
        "--val-batches", "0",
        "--save-every", str(train_cfg.get("save_steps", 50)),
        "--max-seq-length", "512",
        "--adapter-path", adapter_path,
        "--seed", str(cfg.get("seed", 42)),
        "--grad-checkpoint",
    ]

    print(f"[train] model: {cfg['model_name']}")
    print(f"[train] iters: {num_iters}, batch_size: 1")
    print(f"[train] num_layers: 8, lr: {train_cfg['learning_rate']}")
    print(f"[train] adapter output: {adapter_path}")
    print()

    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)

    print(f"\n[done] adapters saved to {adapter_path}")
    print(f"To test inference, run:")
    print(f"  python -m mlx_lm.generate --model {cfg['model_name']} --adapter-path {adapter_path} --prompt '### Context:\\n...\\n\\n### Question:\\n...\\n\\n### Answer:'")


if __name__ == "__main__":
    main()
