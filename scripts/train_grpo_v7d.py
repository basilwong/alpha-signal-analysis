"""
V7d: GRPO Training on OpenReasoning-Nemotron-7B (starting from V4 SFT checkpoint).

Uses actual market returns as the reward signal. The model generates predictions,
and gets reinforced for predictions that correlate with actual stock movements.

Usage:
    modal run scripts/train_grpo_v7d.py
"""

import modal
import json
import re
import os
import numpy as np

app = modal.App("alpha-signal-v7d-grpo")

train_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch==2.7.1", "torchvision==0.22.1")
    .pip_install(
        "datasets", "huggingface_hub", "trl>=1.5.0", "transformers",
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
GRPO_DATA = "/outputs/grpo_train_articles_with_returns.jsonl"
OUTPUT_DIR = "/outputs/alpha-signal-grpo-v7d-clean"

ACTIVE_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "HON"]

SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

The quantum computing universe consists of these 10 tickers:

**Active (scored):**
- IONQ: IonQ (trapped-ion, 100% quantum revenue, pure-play)
- RGTI: Rigetti Computing (superconducting, 100% quantum revenue, pure-play)
- QBTS: D-Wave Quantum (quantum annealing, 100% quantum revenue, pure-play)
- QUBT: Quantum Computing Inc. (neutral atom, 100% quantum revenue, pure-play)
- QNT: Quantinuum (trapped-ion, 100% quantum revenue, pure-play, IPO'd June 2026)
- IBM: International Business Machines (superconducting, ~2% quantum revenue)
- HON: Honeywell (trapped-ion, ~1% quantum revenue post-Quantinuum spinoff)

**Inactive (always 0.0):**
- MSFT, GOOGL, NVDA: always 0.0

Score ranges: Pure-play [-2.0, +2.0], HON [-0.3, +0.3], IBM [-0.15, +0.15], MSFT/GOOGL/NVDA = 0.0

Output ONLY a valid JSON object with signal_vector containing all 10 tickers."""


@app.function(
    image=train_image, gpu="A100-80GB", timeout=21600,  # 6 hours
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_grpo():
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOTrainer, GRPOConfig

    print("=" * 60)
    print("V7d: GRPO Training (reward = correlation with actual returns)")
    print("=" * 60)

    # Load the V4 SFT model (base + adapter)
    print("Loading base model + V4 SFT adapter...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True,
    )

    # Apply the V4 adapter as starting point
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, SFT_ADAPTER, is_trainable=True)
    print(f"Model loaded with V4 adapter")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load GRPO data
    grpo_data = []
    with open(GRPO_DATA) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                # Format as chat messages for the prompt
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": d["prompt"]},
                ]
                prompt_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                grpo_data.append({
                    "prompt": prompt_text,
                    "actual_returns": json.dumps(d["actual_returns_5d"]),
                })

    dataset = Dataset.from_list(grpo_data)
    print(f"GRPO dataset: {len(dataset)} articles with actual returns")

    # Define reward function
    def reward_fn(completions, prompts=None, **kwargs):
        """
        Reward function: scores each completion based on correlation with actual returns.
        
        For each completion:
        1. Parse the JSON signal vector
        2. Extract predicted scores for active tickers
        3. Compare against actual returns (stored in the dataset)
        4. Reward = direction accuracy + magnitude correlation
        """
        rewards = []

        for i, completion in enumerate(completions):
            try:
                # Get actual returns for this example
                actual_str = kwargs.get("actual_returns", ["{}"] * len(completions))[i]
                actual = json.loads(actual_str)

                # Parse the model's output
                text = completion
                if "<think>" in text:
                    think_end = text.find("</think>")
                    if think_end != -1:
                        text = text[think_end + 8:].strip()

                start_j = text.find("{")
                end_j = text.rfind("}") + 1

                if start_j < 0 or end_j <= start_j:
                    rewards.append(-1.0)  # Penalty for no JSON
                    continue

                json_str = re.sub(r',\s*([}\]])', r'\1', text[start_j:end_j])
                signal = json.loads(json_str)
                sv = signal.get("signal_vector", {})

                # Compute reward components
                pred_scores = []
                actual_cars = []
                for ticker in ACTIVE_TICKERS:
                    if ticker in sv and actual.get(ticker) is not None:
                        score = sv[ticker].get("score", 0)
                        if isinstance(score, (int, float)):
                            pred_scores.append(score)
                            actual_cars.append(actual[ticker])

                if len(pred_scores) < 2:
                    rewards.append(-0.5)  # Penalty for too few predictions
                    continue

                # Component 1: Direction accuracy
                direction_correct = sum(
                    1 for s, r in zip(pred_scores, actual_cars)
                    if (s > 0 and r > 0) or (s < 0 and r < 0) or (abs(s) < 0.01)
                ) / len(pred_scores)

                # Component 2: Selectivity bonus (reward for zeros on small moves)
                selectivity_bonus = sum(
                    1 for s, r in zip(pred_scores, actual_cars)
                    if abs(s) < 0.01 and abs(r) < 0.02  # Correctly stayed silent
                ) / max(len(pred_scores), 1) * 0.3

                # Component 3: Magnitude correlation
                if np.std(pred_scores) > 0 and np.std(actual_cars) > 0:
                    correlation = float(np.corrcoef(pred_scores, actual_cars)[0, 1])
                    if np.isnan(correlation):
                        correlation = 0.0
                else:
                    correlation = 0.0

                # Combined reward (range roughly -1 to +1)
                reward = 0.4 * direction_correct + 0.4 * max(correlation, -0.5) + 0.2 * selectivity_bonus

                # Format compliance bonus
                has_all_tickers = all(t in sv for t in ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"])
                inactive_correct = all(sv.get(t, {}).get("score", 1) == 0 for t in ["MSFT", "GOOGL", "NVDA"])
                if has_all_tickers and inactive_correct:
                    reward += 0.1

                rewards.append(reward)

            except Exception as e:
                rewards.append(-1.0)  # Penalty for any failure

        return rewards

    # GRPO config
    grpo_config = GRPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,  # Lower LR for RL (more conservative updates)
        num_generations=4,  # Generate 4 candidates per prompt
        max_completion_length=1500,
        logging_steps=5,
        save_strategy="steps",
        save_steps=50,  # Save every 50 steps to survive timeouts
        save_total_limit=3,
        bf16=True,
        seed=42,
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        reward_funcs=reward_fn,
        args=grpo_config,
    )

    print("Starting GRPO training...")
    trainer.train()
    print("GRPO training complete!")

    output_vol.commit()
    print("Volume committed.")

    return json.dumps({"status": "complete", "steps": trainer.state.global_step})


@app.local_entrypoint()
def main():
    result = train_grpo.remote()
    print(f"\nResult: {result}")
