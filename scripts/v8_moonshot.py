"""
V8 Moonshot: Run concurrently:
1. V8 SFT training on 7B (from V8 GPT 5.5 data)
2. Base model predictions for 14B
3. Base model predictions for 32B

Usage:
    modal run scripts/v8_moonshot.py --function train_v8_sft
    modal run scripts/v8_moonshot.py --function predict_base_14b
    modal run scripts/v8_moonshot.py --function predict_base_32b
"""

import modal
import json
import re
import time
import os
import numpy as np

app = modal.App("alpha-signal-v8-moonshot")

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

predict_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision")
    .pip_install(
        "vllm>=0.12.0", "huggingface_hub", "transformers",
        "accelerate", "sentencepiece", "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-alpha-signal", create_if_missing=True)
output_vol = modal.Volume.from_name("alpha-signal-outputs", create_if_missing=True)

BASE_7B = "nvidia/OpenReasoning-Nemotron-7B"
BASE_14B = "nvidia/OpenReasoning-Nemotron-14B"
BASE_32B = "nvidia/OpenReasoning-Nemotron-32B"
EVAL_FILE = "/outputs/articles_eval.jsonl"

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

SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess novelty and likely decay speed.",
    "arxiv": "This is an academic paper abstract. Most are incremental. Only significant scores for genuine breakthroughs.",
}


def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_predictions(model_name, output_file, gpu="A10G", max_model_len=2048):
    """Generic prediction function for any base model."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"Loading {model_name} with vLLM...")
    llm = LLM(model=model_name, trust_remote_code=True,
              max_model_len=max_model_len, gpu_memory_utilization=0.90, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Ready!")

    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)

    # Sanity check
    test_prompts = []
    for a in articles[:3]:
        text = clean_text(a.get("text", ""))[:2500]
        source = a.get("source", "news")
        source_inst = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        user_msg = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
        test_prompts.append(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))

    test_out = llm.generate(test_prompts, SamplingParams(temperature=0.3, top_p=0.9, max_tokens=3000))
    sanity = 0
    for o in test_out:
        raw = o.outputs[0].text
        if "<think>" in raw:
            think_end = raw.find("</think>")
            if think_end != -1:
                raw = raw[think_end + 8:]
        if "{" in raw and "signal_vector" in raw:
            sanity += 1
    print(f"Sanity check: {sanity}/3")
    if sanity == 0:
        print("WARNING: Sanity check failed. Model may not produce JSON. Continuing anyway...")

    # Full batch
    prompts = []
    article_meta = []
    for article in articles:
        text = clean_text(article.get("text", ""))
        source = article.get("source", "news")
        if len(text) < 30:
            article_meta.append({"idx": article["idx"], "skip": True, "date": article.get("date", ""),
                                 "title": article.get("title", ""), "source": source})
            continue
        source_inst = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        user_msg = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text[:2500]}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        article_meta.append({"idx": article["idx"], "skip": False, "date": article.get("date", ""),
                             "title": article.get("title", ""), "source": source})

    sampling_params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=3000)
    print(f"Generating {len(prompts)} predictions...")
    start = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start
    print(f"Done in {total_time/60:.1f} min")

    success = 0
    errors = 0
    prompt_idx = 0

    with open(output_file, "w") as f:
        for meta in article_meta:
            if meta.get("skip"):
                f.write(json.dumps({"article_idx": meta["idx"], "date": meta["date"],
                                    "title": meta["title"], "source": meta["source"], "status": "skipped"}) + "\n")
                continue
            raw = outputs[prompt_idx].outputs[0].text
            prompt_idx += 1
            try:
                parse_text = raw
                if "<think>" in parse_text:
                    think_end = parse_text.find("</think>")
                    if think_end != -1:
                        parse_text = parse_text[think_end + 8:].strip()
                start_j = parse_text.find("{")
                end_j = parse_text.rfind("}") + 1
                if start_j >= 0 and end_j > start_j:
                    signal = json.loads(re.sub(r',\s*([}\]])', r'\1', parse_text[start_j:end_j]))
                else:
                    raise ValueError("No JSON")
                f.write(json.dumps({"article_idx": meta["idx"], "date": meta["date"],
                                    "title": meta["title"], "source": meta["source"],
                                    "status": "success", "signal": signal}) + "\n")
                success += 1
            except Exception as e:
                f.write(json.dumps({"article_idx": meta["idx"], "date": meta["date"],
                                    "title": meta["title"], "source": meta["source"],
                                    "status": "error", "error": str(e)[:200]}) + "\n")
                errors += 1

    output_vol.commit()
    print(f"Results: success={success} errors={errors}")
    return json.dumps({"success": success, "errors": errors, "minutes": round(total_time/60, 1)})


# ============================================================
# FUNCTION 1: V8 SFT Training
# ============================================================
@app.function(
    image=train_image, gpu="A100-80GB", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_v8_sft():
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig
    import random

    print("=" * 60)
    print("V8 SFT: OpenReasoning-Nemotron-7B on GPT 5.5 training data")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_7B, max_seq_length=4096,
        dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True,
    )
    model = FastLanguageModel.get_peft_model(
        model, r=16, lora_alpha=32, lora_dropout=0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none", use_gradient_checkpointing="unsloth", random_state=42,
    )
    print(f"LoRA applied: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    all_rows = []
    with open("/outputs/alpha_signal_train_v8_combined.jsonl") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                text = tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=False)
                all_rows.append({"text": text})

    random.seed(42)
    random.shuffle(all_rows)
    split_idx = int(len(all_rows) * 0.9)
    train_ds = Dataset.from_list(all_rows[:split_idx])
    val_ds = Dataset.from_list(all_rows[split_idx:])
    print(f"Dataset: {len(train_ds)} train, {len(val_ds)} val")

    sft_config = SFTConfig(
        output_dir="/outputs/alpha-signal-v8-sft",
        num_train_epochs=2,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=2e-4, weight_decay=0.01, warmup_steps=10,
        lr_scheduler_type="cosine", logging_steps=10,
        save_strategy="steps", save_steps=30, save_total_limit=3,
        eval_strategy="epoch",
        bf16=True, seed=42, report_to="none",
        dataset_text_field="text", max_seq_length=4096, packing=True,
    )

    trainer = SFTTrainer(model=model, tokenizer=tokenizer,
                         train_dataset=train_ds, eval_dataset=val_ds, args=sft_config)
    print("Training...")
    trainer.train()
    print("Done!")

    for entry in trainer.state.log_history:
        if "loss" in entry:
            print(f"  Step {entry.get('step', '?')}: loss={entry['loss']:.4f}")
        if "eval_loss" in entry:
            print(f"  Step {entry.get('step', '?')}: eval_loss={entry['eval_loss']:.4f}")

    output_vol.commit()
    return json.dumps({"status": "complete", "steps": trainer.state.global_step})


# ============================================================
# FUNCTION 2: Base 14B Predictions
# ============================================================
@app.function(
    image=predict_image, gpu="A100-80GB", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_base_14b():
    return run_predictions(BASE_14B, "/outputs/predictions_base_14b.jsonl", max_model_len=4096)


# ============================================================
# FUNCTION 3: Base 32B Predictions
# ============================================================
@app.function(
    image=predict_image, gpu="A100-80GB", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_base_32b():
    return run_predictions(BASE_32B, "/outputs/predictions_base_32b.jsonl", max_model_len=4096)


@app.local_entrypoint()
def main():
    # Launch all three concurrently
    import asyncio

    print("Launching all three jobs concurrently...")
    sft_handle = train_v8_sft.spawn()
    pred_14b_handle = predict_base_14b.spawn()
    pred_32b_handle = predict_base_32b.spawn()

    print(f"V8 SFT: {sft_handle.object_id}")
    print(f"14B predictions: {pred_14b_handle.object_id}")
    print(f"32B predictions: {pred_32b_handle.object_id}")
    print("\nAll jobs spawned. They will run independently.")
    print("Check Modal dashboard for progress.")
