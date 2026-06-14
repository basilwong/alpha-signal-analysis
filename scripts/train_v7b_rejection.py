"""
V7b: SFT on rejection-sampled data (best-of-4 scored against actual returns).
Starts from V4 SFT checkpoint.

Usage:
    modal run scripts/train_v7b_rejection.py
"""

import modal
import json
import os

app = modal.App("quantum-alpha-v7b-rejection")

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1")
    .pip_install(
        "datasets", "huggingface_hub", "trl", "transformers",
        "accelerate", "peft", "bitsandbytes", "sentencepiece", "protobuf",
    )
    .run_commands(
        "pip install --no-deps unsloth unsloth-zoo",
        "pip install --no-deps cut-cross-entropy",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

BASE_MODEL = "nvidia/OpenReasoning-Nemotron-7B"
SFT_ADAPTER = "/outputs/quantum-alpha-openreasoning-7b/checkpoint-100"
TRAIN_FILE = "/outputs/v7b_best_of_4_clean.jsonl"
OUTPUT_DIR = "/outputs/quantum-alpha-v7b-clean"


@app.function(
    image=train_image, gpu="A100-80GB", timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    from peft import PeftModel

    print("=" * 60)
    print("V7b: Rejection Sampling SFT (best-of-4 scored by returns)")
    print("=" * 60)

    # Load base + V4 adapter as starting point
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True,
    )

    # Load V4 adapter and make trainable for continued SFT
    model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)
    print(f"Model loaded with V4 adapter (continuing SFT)")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load best-of-4 data
    all_rows = []
    with open(TRAIN_FILE) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                text = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                all_rows.append({"text": text})

    print(f"Dataset: {len(all_rows)} best-of-4 examples")

    train_ds = Dataset.from_list(all_rows)

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=8,
        learning_rate=1e-5,  # Lower LR since we're continuing from V4
        weight_decay=0.01, warmup_steps=5,
        lr_scheduler_type="cosine", logging_steps=5,
        save_strategy="epoch", save_total_limit=1,
        bf16=True, seed=42, report_to="none",
        dataset_text_field="text", max_seq_length=2048, packing=True,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=train_ds, args=sft_config)

    print("Training...")
    trainer.train()
    print("Training complete!")

    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(f"  Step {entry.get('step', '?')}: loss={entry['loss']:.4f}")

    output_vol.commit()
    return json.dumps({"status": "complete", "steps": trainer.state.global_step})


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\nResult: {result}")
