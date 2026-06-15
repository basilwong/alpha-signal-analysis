"""
V6 Training ONLY (standalone, no chaining).
Run this first, then run merge_predict_v6_standalone.py separately.

Usage:
    modal run scripts/train_v6_standalone.py
"""

import modal
import json
import os

app = modal.App("quantum-alpha-v6-train")

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1", "torchaudio")
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
TRAIN_FILE = "/outputs/quantum_alpha_train_v6.jsonl"
OUTPUT_DIR = "/outputs/quantum-alpha-openreasoning-7b-v6"


@app.function(
    image=train_image,
    gpu="A100",
    timeout=7200,  # 2 hour timeout (generous for ~50 min job)
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    import random

    print("=" * 60)
    print("V6 TRAINING: OpenReasoning-Nemotron-7B (standalone)")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=4096,
        dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True,
    )
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"LoRA: {trainable:,} trainable params")

    all_rows = []
    with open(TRAIN_FILE) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                text = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                all_rows.append({"text": text})

    random.seed(42)
    random.shuffle(all_rows)
    split_idx = int(len(all_rows) * 0.9)
    train_ds = Dataset.from_list(all_rows[:split_idx])
    val_ds = Dataset.from_list(all_rows[split_idx:])
    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR, num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=2e-4, weight_decay=0.01, warmup_steps=10,
        lr_scheduler_type="cosine", logging_steps=10,
        save_strategy="epoch", save_total_limit=2, eval_strategy="epoch",
        bf16=True, seed=42, report_to="none",
        dataset_text_field="text", max_seq_length=4096, packing=True,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=train_ds, eval_dataset=val_ds, args=sft_config)

    print("Training...")
    trainer.train()
    print("Training complete!")

    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(f"  Step {entry.get('step', '?')}: loss={entry['loss']:.4f}")
        if "eval_loss" in entry:
            print(f"  Step {entry.get('step', '?')}: eval_loss={entry['eval_loss']:.4f}")

    final_train_loss = None
    final_eval_loss = None
    for entry in reversed(trainer.state.log_history):
        if "train_loss" in entry and final_train_loss is None:
            final_train_loss = entry["train_loss"]
        if "eval_loss" in entry and final_eval_loss is None:
            final_eval_loss = entry["eval_loss"]

    print(f"\nFinal: train_loss={final_train_loss}, eval_loss={final_eval_loss}")
    print(f"Total steps: {trainer.state.global_step}")

    output_vol.commit()
    print("Volume committed. Training done.")
    return json.dumps({"train_loss": final_train_loss, "eval_loss": final_eval_loss,
                       "steps": trainer.state.global_step})


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\nResult: {result}")
