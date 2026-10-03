"""
QLoRA fine-tune Llama 3.1 8B on synthetic financial Q&A data.

Intended to run on a CUDA box (Kaggle 2x T4 / Modal A100) where bitsandbytes
4-bit quantization is available. For local M2 Mac dev, use MLX instead
(see CLAUDE.md "Local Dev on M2 Mac").

Usage:
    export HF_TOKEN=...
    export WANDB_API_KEY=...
    python train.py --config ../config/training_config.yaml
"""

import argparse
import os
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def format_example(example: dict) -> str:
    eos = "</s>"
    return f"{example['prompt']} {example['completion']}{eos}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config" / "training_config.yaml"))
    parser.add_argument("--push-to-hub", action="store_true", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    if os.environ.get("WANDB_API_KEY"):
        import wandb

        wandb.init(
            project=cfg["wandb"]["project"],
            name=cfg["wandb"].get("run_name"),
            config=cfg,
        )
        report_to = "wandb"
    else:
        report_to = "none"

    dataset_path = cfg["dataset_path"]
    if not Path(dataset_path).exists():
        raise FileNotFoundError(
            f"{dataset_path} not found. Run generate_qa.py first to produce training data."
        )

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    split = dataset.train_test_split(test_size=cfg.get("val_split", 0.05), seed=cfg.get("seed", 42))
    train_ds, eval_ds = split["train"], split["test"]

    quant_cfg = cfg["quantization"]
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=quant_cfg["load_in_4bit"],
        bnb_4bit_quant_type=quant_cfg["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, quant_cfg["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=quant_cfg["bnb_4bit_use_double_quant"],
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], token=os.environ.get("HF_TOKEN"))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"],
        quantization_config=bnb_config,
        device_map="auto",
        token=os.environ.get("HF_TOKEN"),
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["qlora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_cfg = cfg["training"]
    sft_config = SFTConfig(
        output_dir=cfg["output"]["model_dir"],
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=train_cfg.get("warmup_ratio", 0.03),
        max_seq_length=train_cfg["max_seq_length"],
        bf16=train_cfg.get("bf16", True),
        logging_steps=train_cfg.get("logging_steps", 10),
        eval_strategy="steps",
        eval_steps=train_cfg.get("eval_steps", 50),
        save_steps=train_cfg.get("save_steps", 50),
        save_total_limit=train_cfg.get("save_total_limit", 2),
        report_to=report_to,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        formatting_func=format_example,
    )

    trainer.train()

    model_dir = cfg["output"]["model_dir"]
    trainer.save_model(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"[done] model saved to {model_dir}")

    should_push = args.push_to_hub if args.push_to_hub is not None else cfg["output"].get("push_to_hub", False)
    if should_push:
        hub_repo = cfg["output"]["hub_repo"]
        model.push_to_hub(hub_repo, token=os.environ.get("HF_TOKEN"))
        tokenizer.push_to_hub(hub_repo, token=os.environ.get("HF_TOKEN"))
        print(f"[done] pushed to https://huggingface.co/{hub_repo}")


if __name__ == "__main__":
    main()
