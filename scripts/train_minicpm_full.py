"""
Full Fine-Tuning: MiniCPM-2B on V4 Training Data

Optimized hyperparameters based on adversarial review:
- r=16, alpha=32 (scaling factor 2.0)
- 2 epochs (reduced from 4 to prevent overfitting)
- Packing enabled for efficiency
- 10% validation holdout

Usage:
    modal run scripts/train_minicpm_full.py
"""

import modal
import json
import os

app = modal.App("quantum-alpha-minicpm-train")

finetune_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "datasets",
        "huggingface_hub",
        "trl",
        "transformers==4.57.3",
        "accelerate",
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        "pip install --no-deps unsloth unsloth-zoo",
        "pip install --no-deps cut-cross-entropy",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

BASE_MODEL = "openbmb/MiniCPM-2B-sft-bf16-llama-format"
TRAIN_FILE = "/outputs/quantum_alpha_train_v4.jsonl"
OUTPUT_DIR = "/outputs/quantum-alpha-minicpm-2b"
HUB_MODEL_ID = "basilwong/quantum-alpha-minicpm-2b"


@app.function(
    image=finetune_image,
    gpu="A100",
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    """Full fine-tuning run with merged export."""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    print("=" * 60)
    print("QUANTUM ALPHA: MiniCPM-2B Full Training Run")
    print("=" * 60)
    print(f"Model: {BASE_MODEL}")
    print(f"Data: {TRAIN_FILE}")
    print(f"LoRA: r=16, alpha=32 (scaling=2.0)")
    print(f"Epochs: 2")
    print("=" * 60)

    # Load model
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    print("Model loaded.")

    # Apply LoRA with corrected hyperparameters
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {trainable:,} trainable / {total:,} total ({100*trainable/total:.2f}%)")

    # Load and prepare dataset with 10% validation split
    all_rows = []
    with open(TRAIN_FILE) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                text = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                all_rows.append({"text": text})

    # 90/10 train/val split
    import random
    random.seed(42)
    random.shuffle(all_rows)
    split_idx = int(len(all_rows) * 0.9)
    train_rows = all_rows[:split_idx]
    val_rows = all_rows[split_idx:]

    train_ds = Dataset.from_list(train_rows)
    val_ds = Dataset.from_list(val_rows)
    print(f"Dataset: {len(train_rows)} train, {len(val_rows)} validation")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Training config
    sft_config = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="epoch",
        fp16=False,
        bf16=True,
        seed=42,
        report_to="none",
        dataset_text_field="text",
        max_seq_length=4096,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        args=sft_config,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Print loss history
    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(f"  Step {entry.get('step', '?')}: train_loss={entry['loss']:.4f}")
        if "eval_loss" in entry:
            print(f"  Step {entry.get('step', '?')}: eval_loss={entry['eval_loss']:.4f}")

    final_train_loss = None
    final_eval_loss = None
    for entry in reversed(trainer.state.log_history):
        if "train_loss" in entry and final_train_loss is None:
            final_train_loss = entry["train_loss"]
        if "eval_loss" in entry and final_eval_loss is None:
            final_eval_loss = entry["eval_loss"]

    print(f"\nFinal train loss: {final_train_loss}")
    print(f"Final eval loss: {final_eval_loss}")

    # Save merged 16-bit model
    merged_path = OUTPUT_DIR + "/merged_16bit"
    print(f"\nSaving merged 16-bit model to {merged_path}...")
    model.save_pretrained_merged(merged_path, tokenizer, save_method="merged_16bit")
    print("Merged model saved.")

    # Save GGUF Q4_K_M
    gguf_path = OUTPUT_DIR + "/gguf"
    print(f"Saving GGUF Q4_K_M to {gguf_path}...")
    try:
        model.save_pretrained_gguf(gguf_path, tokenizer, quantization_method="q4_k_m")
        print("GGUF saved successfully!")
    except Exception as e:
        print(f"GGUF save failed (non-critical): {e}")

    # Push to HuggingFace
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print(f"\nPushing to HF Hub: {HUB_MODEL_ID}...")
        try:
            model.push_to_hub_merged(
                HUB_MODEL_ID, tokenizer,
                save_method="merged_16bit",
                token=hf_token,
                private=True,
            )
            print("Pushed to HF Hub!")
        except Exception as e:
            print(f"HF push failed: {e}")

    # Commit volume
    output_vol.commit()
    print("Volume committed.")

    return json.dumps({
        "status": "complete",
        "train_loss": final_train_loss,
        "eval_loss": final_eval_loss,
        "total_steps": trainer.state.global_step,
        "train_examples": len(train_rows),
        "val_examples": len(val_rows),
    })


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\nTraining result: {result}")
