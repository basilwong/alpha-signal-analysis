"""
14B GRPO Training (standalone, runs after SFT is complete).

Usage:
    modal run scripts/train_14b_grpo_standalone.py
"""

import modal
import json
import re
import os
import numpy as np

app = modal.App("alpha-signal-14b-grpo")

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

BASE_MODEL = "nvidia/OpenReasoning-Nemotron-14B"
SFT_ADAPTER = "/outputs/alpha-signal-14b-sft/checkpoint-100"
GRPO_DATA = "/outputs/grpo_train_articles_with_returns.jsonl"
GRPO_OUTPUT = "/outputs/alpha-signal-14b-grpo"
ACTIVE_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "HON"]


@app.function(
    image=train_image, gpu="A100-80GB", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_grpo():
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOTrainer, GRPOConfig
    from peft import PeftModel

    print("=" * 60)
    print("14B GRPO: Starting from 14B SFT checkpoint-100")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)
    print("Model loaded with 14B SFT adapter")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. Output ONLY a valid JSON object with signal_vector containing all 10 tickers."""

    grpo_data = []
    with open(GRPO_DATA) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": d["prompt"]}]
                prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                grpo_data.append({"prompt": prompt_text, "actual_returns": json.dumps(d["actual_returns_5d"])})

    dataset = Dataset.from_list(grpo_data)
    print(f"GRPO dataset: {len(dataset)} articles")

    def reward_fn(completions, prompts=None, **kwargs):
        rewards = []
        for i, completion in enumerate(completions):
            try:
                actual_str = kwargs.get("actual_returns", ["{}"] * len(completions))[i]
                actual = json.loads(actual_str)
                text = completion
                if "<think>" in text:
                    te = text.find("</think>")
                    if te != -1: text = text[te+8:].strip()
                sj = text.find("{"); ej = text.rfind("}")+1
                if sj < 0 or ej <= sj: rewards.append(-1.0); continue
                signal = json.loads(re.sub(r',\s*([}\]])', r'\1', text[sj:ej]))
                sv = signal.get("signal_vector", {})
                ps, ac = [], []
                for t in ACTIVE_TICKERS:
                    if t in sv and actual.get(t) is not None:
                        val = sv[t]
                        score = val.get("score", 0) if isinstance(val, dict) else val if isinstance(val, (int, float)) else 0
                        ps.append(score); ac.append(actual[t])
                if len(ps) < 2: rewards.append(-0.5); continue
                direction = sum(1 for s, r in zip(ps, ac) if (s>0 and r>0) or (s<0 and r<0) or abs(s)<0.01) / len(ps)
                corr = float(np.corrcoef(ps, ac)[0, 1]) if np.std(ps) > 0 and np.std(ac) > 0 else 0
                if np.isnan(corr): corr = 0
                selectivity = sum(1 for s, r in zip(ps, ac) if abs(s)<0.01 and abs(r)<0.02) / max(len(ps), 1) * 0.3
                reward = 0.4 * direction + 0.4 * max(corr, -0.5) + 0.2 * selectivity
                has_all = all(t in sv for t in ["IONQ","RGTI","QBTS","QUBT","IBM","HON","MSFT","GOOGL","NVDA"])
                if has_all: reward += 0.1
                rewards.append(reward)
            except: rewards.append(-1.0)
        return rewards

    grpo_config = GRPOConfig(
        output_dir=GRPO_OUTPUT,
        num_train_epochs=1,
        per_device_train_batch_size=1, gradient_accumulation_steps=4,
        learning_rate=5e-6, num_generations=4, max_completion_length=1500,
        logging_steps=5, save_strategy="steps", save_steps=50, save_total_limit=3,
        bf16=True, seed=42, report_to="none",
    )

    trainer = GRPOTrainer(model=model, tokenizer=tokenizer, train_dataset=dataset,
                          reward_funcs=reward_fn, args=grpo_config)
    print("Starting GRPO...")
    trainer.train()
    print("14B GRPO complete!")
    output_vol.commit()
    return json.dumps({"status": "complete", "steps": trainer.state.global_step})


@app.local_entrypoint()
def main():
    result = train_grpo.remote()
    print(f"Result: {result}")
