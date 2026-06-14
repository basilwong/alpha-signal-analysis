"""
V7b/c Candidate Generation: Generate 4 predictions per article using V4 model,
score them against actual returns, save:
- Best-of-4 for V7b (rejection sampling SFT)
- Best vs worst pairs for V7c (DPO)

Usage:
    modal run scripts/generate_candidates_v7bc.py
"""

import modal
import json
import re
import time
import os
import numpy as np

app = modal.App("quantum-alpha-v7bc-candidates")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "vllm>=0.12.0", "huggingface_hub", "transformers",
        "accelerate", "sentencepiece", "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

MERGED_MODEL = "/outputs/openreasoning-7b-merged"  # V4 merged model
GRPO_DATA = "/outputs/grpo_train_articles_with_returns.jsonl"
V7B_OUTPUT = "/outputs/v7b_best_of_4_clean.jsonl"
V7C_OUTPUT = "/outputs/v7c_dpo_pairs_clean.jsonl"

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

Output ONLY a valid JSON object with signal_vector containing all 10 tickers with score and reasoning fields."""


def score_prediction(signal, actual_returns):
    """Score a prediction against actual returns. Returns reward in [-1, 1]."""
    sv = signal.get("signal_vector", {})
    pred_scores = []
    actual_cars = []

    for ticker in ACTIVE_TICKERS:
        if ticker in sv and actual_returns.get(ticker) is not None:
            val = sv[ticker]
            if isinstance(val, dict):
                score = val.get("score", 0)
            elif isinstance(val, (int, float)):
                score = val
            else:
                continue
            if isinstance(score, (int, float)):
                pred_scores.append(score)
                actual_cars.append(actual_returns[ticker])

    if len(pred_scores) < 2:
        return -1.0

    # Direction accuracy
    direction = sum(
        1 for s, r in zip(pred_scores, actual_cars)
        if (s > 0 and r > 0) or (s < 0 and r < 0) or (abs(s) < 0.01)
    ) / len(pred_scores)

    # Correlation
    if np.std(pred_scores) > 0 and np.std(actual_cars) > 0:
        corr = float(np.corrcoef(pred_scores, actual_cars)[0, 1])
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    return 0.5 * direction + 0.5 * max(corr, -0.5)


def parse_signal(raw_text):
    """Parse JSON signal from model output, handling thinking blocks."""
    text = raw_text
    if "<think>" in text:
        think_end = text.find("</think>")
        if think_end != -1:
            text = text[think_end + 8:].strip()

    start_j = text.find("{")
    end_j = text.rfind("}") + 1
    if start_j < 0 or end_j <= start_j:
        return None

    json_str = re.sub(r',\s*([}\]])', r'\1', text[start_j:end_j])
    try:
        return json.loads(json_str)
    except:
        return None


@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
)
def generate_candidates():
    """Generate 4 candidates per article, score, save best and pairs."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print("Loading V4 merged model with vLLM...")
    llm = LLM(model=MERGED_MODEL, trust_remote_code=True,
              max_model_len=2048, gpu_memory_utilization=0.90, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
    print("Ready!")

    # Load articles with returns
    articles = []
    with open(GRPO_DATA) as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    print(f"Loaded {len(articles)} articles with return data")

    # Sanity check: 2 articles first
    test_prompts = []
    for a in articles[:2]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": a["prompt"]}]
        test_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    test_out = llm.generate(test_prompts, SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1500, n=1))
    sanity_ok = sum(1 for o in test_out if parse_signal(o.outputs[0].text) is not None)
    print(f"Sanity check: {sanity_ok}/2 valid JSON")
    if sanity_ok == 0:
        return json.dumps({"error": "sanity_check_failed"})

    # Generate 4 candidates per article using temperature sampling
    # vLLM's n parameter generates multiple completions per prompt
    all_prompts = []
    for a in articles:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": a["prompt"]}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        all_prompts.append(prompt)

    print(f"Generating 4 candidates each for {len(all_prompts)} articles...")
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=1500, n=4)
    start_time = time.time()
    outputs = llm.generate(all_prompts, sampling_params)
    gen_time = time.time() - start_time
    print(f"Generation done in {gen_time/60:.1f} minutes")

    # Score and save
    v7b_data = []  # Best-of-4 for rejection sampling SFT
    v7c_data = []  # Best vs worst pairs for DPO
    total_valid = 0
    total_invalid = 0

    for i, (article, output) in enumerate(zip(articles, outputs)):
        actual_returns = article["actual_returns_5d"]
        candidates = []

        for completion in output.outputs:
            signal = parse_signal(completion.text)
            if signal is not None:
                reward = score_prediction(signal, actual_returns)
                candidates.append({
                    "text": completion.text,
                    "signal": signal,
                    "reward": reward,
                })
                total_valid += 1
            else:
                total_invalid += 1

        if len(candidates) < 2:
            continue

        # Sort by reward
        candidates.sort(key=lambda x: x["reward"], reverse=True)

        # V7b: Keep the best
        best = candidates[0]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": article["prompt"]},
            {"role": "assistant", "content": json.dumps(best["signal"])},
        ]
        v7b_data.append({"messages": messages, "reward": best["reward"]})

        # V7c: Best vs worst pair
        worst = candidates[-1]
        if best["reward"] - worst["reward"] > 0.1:  # Only use pairs with meaningful difference
            v7c_data.append({
                "prompt": article["prompt"],
                "chosen": json.dumps(best["signal"]),
                "rejected": json.dumps(worst["signal"]),
                "chosen_reward": best["reward"],
                "rejected_reward": worst["reward"],
            })

    # Save
    with open(V7B_OUTPUT, "w") as f:
        for d in v7b_data:
            f.write(json.dumps(d) + "\n")

    with open(V7C_OUTPUT, "w") as f:
        for d in v7c_data:
            f.write(json.dumps(d) + "\n")

    output_vol.commit()

    print(f"\nResults:")
    print(f"  Valid candidates: {total_valid}, Invalid: {total_invalid}")
    print(f"  V7b (best-of-4): {len(v7b_data)} examples")
    print(f"  V7c (DPO pairs): {len(v7c_data)} pairs")
    print(f"  Generation time: {gen_time/60:.1f} min")

    return json.dumps({
        "v7b_examples": len(v7b_data),
        "v7c_pairs": len(v7c_data),
        "valid_candidates": total_valid,
        "invalid_candidates": total_invalid,
    })


@app.local_entrypoint()
def main():
    result = generate_candidates.remote()
    print(f"\nResult: {result}")
