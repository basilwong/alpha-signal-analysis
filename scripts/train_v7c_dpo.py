"""
V7c: DPO on preference pairs (best vs worst scored against actual returns).
Starts from V4 SFT checkpoint.

Usage:
    modal run scripts/train_v7c_dpo.py
"""

import modal
import json
import os

app = modal.App("alpha-signal-v7c-dpo")

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

hf_cache_vol = modal.Volume.from_name("hf-cache-alpha-signal", create_if_missing=True)
output_vol = modal.Volume.from_name("alpha-signal-outputs", create_if_missing=True)

BASE_MODEL = "nvidia/OpenReasoning-Nemotron-7B"
SFT_ADAPTER = "/outputs/quantum-alpha-openreasoning-7b/checkpoint-100"
DPO_DATA = "/outputs/v7c_dpo_pairs_clean.jsonl"
OUTPUT_DIR = "/outputs/alpha-signal-v7c-dpo-clean"

SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

Output ONLY a valid JSON object with signal_vector containing all 10 tickers."""


@app.function(
    image=train_image, gpu="A100-80GB", timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train():
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig
    from peft import PeftModel

    print("=" * 60)
    print("V7c: DPO Training (preference pairs scored by returns)")
    print("=" * 60)

    # Load base + V4 adapter
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)
    print(f"Model loaded with V4 adapter")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load DPO pairs
    dpo_rows = []
    with open(DPO_DATA) as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                # DPO format: prompt, chosen, rejected
                messages_prompt = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["prompt"]},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages_prompt, tokenize=False, add_generation_prompt=True
                )
                dpo_rows.append({
                    "prompt": prompt_text,
                    "chosen": row["chosen"],
                    "rejected": row["rejected"],
                })

    print(f"DPO pairs: {len(dpo_rows)}")
    dataset = Dataset.from_list(dpo_rows)

    dpo_config = DPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        beta=0.1,  # DPO temperature
        logging_steps=5,
        save_strategy="epoch",
        save_total_limit=1,
        bf16=True,
        seed=42,
        report_to="none",
        max_length=2048,
    )

    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=dpo_config,
    )

    print("Training...")
    trainer.train()
    print("DPO training complete!")

    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(f"  Step {entry.get('step', '?')}: loss={entry['loss']:.4f}")

    output_vol.commit()
    return json.dumps({"status": "complete", "steps": trainer.state.global_step})


@app.local_entrypoint()
def main():
    result = train.remote()
    print(f"\nResult: {result}")
