"""
V8 Phase 2: Merge+Predict for V8 SFT, and GRPO training.
Run as separate functions.

Usage:
    modal run scripts/v8_predict_and_grpo.py --function predict_v8_sft
    modal run scripts/v8_predict_and_grpo.py --function train_v8_grpo
"""

import modal
import json
import re
import time
import os
import numpy as np

app = modal.App("quantum-alpha-v8-phase2")

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
        "accelerate", "peft", "sentencepiece", "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

BASE_MODEL = "nvidia/OpenReasoning-Nemotron-7B"
V8_ADAPTER = "/outputs/quantum-alpha-v8-sft/checkpoint-42"
V8_MERGED = "/outputs/v8-sft-merged"
EVAL_FILE = "/outputs/articles_eval.jsonl"
GRPO_DATA = "/outputs/grpo_train_articles_with_returns.jsonl"
ACTIVE_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "HON"]

SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

The quantum computing universe consists of these 10 tickers:

**Active (scored):**
- IONQ, RGTI, QBTS, QUBT, QNT, IBM, HON

**Inactive (always 0.0):**
- MSFT, GOOGL, NVDA

Score ranges: Pure-play [-2.0, +2.0], HON [-0.3, +0.3], IBM [-0.15, +0.15], MSFT/GOOGL/NVDA = 0.0

Output ONLY a valid JSON object with signal_vector containing all 10 tickers."""

SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess novelty and likely decay speed.",
    "arxiv": "This is an academic paper abstract. Most are incremental.",
}


def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================
# V8 SFT Predictions
# ============================================================
@app.function(
    image=predict_image, gpu="A10G", timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_v8_sft():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from vllm import LLM, SamplingParams

    # Merge
    if not os.path.exists(V8_MERGED + "/config.json"):
        print("Merging V8 SFT adapter...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
        model = PeftModel.from_pretrained(model, V8_ADAPTER)
        model = model.merge_and_unload()
        os.makedirs(V8_MERGED, exist_ok=True)
        model.save_pretrained(V8_MERGED)
        tokenizer.save_pretrained(V8_MERGED)
        output_vol.commit()
        del model; torch.cuda.empty_cache()
        print("Merged!")
    else:
        print("Merged model exists.")

    # Predict
    print("Loading vLLM...")
    llm = LLM(model=V8_MERGED, trust_remote_code=True,
              max_model_len=2048, gpu_memory_utilization=0.90, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(V8_MERGED, trust_remote_code=True)

    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                a = json.loads(line); a["idx"] = i; articles.append(a)

    prompts, meta = [], []
    for a in articles:
        text = clean_text(a.get("text", ""))
        source = a.get("source", "news")
        if len(text) < 30:
            meta.append({"idx": a["idx"], "skip": True, "date": a.get("date",""), "title": a.get("title",""), "source": source})
            continue
        si = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        msg = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{si}\n\n{text[:2500]}"}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        meta.append({"idx": a["idx"], "skip": False, "date": a.get("date",""), "title": a.get("title",""), "source": source})

    print(f"Generating {len(prompts)} predictions...")
    start = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.3, top_p=0.9, max_tokens=3000))
    total_time = time.time() - start

    success, errors, pi = 0, 0, 0
    with open("/outputs/predictions_v8_sft.jsonl", "w") as f:
        for m in meta:
            if m.get("skip"):
                f.write(json.dumps({**m, "status": "skipped"}) + "\n"); continue
            raw = outputs[pi].outputs[0].text; pi += 1
            try:
                pt = raw
                if "<think>" in pt:
                    te = pt.find("</think>")
                    if te != -1: pt = pt[te+8:].strip()
                sj = pt.find("{"); ej = pt.rfind("}")+1
                if sj >= 0 and ej > sj:
                    signal = json.loads(re.sub(r',\s*([}\]])', r'\1', pt[sj:ej]))
                else: raise ValueError("No JSON")
                f.write(json.dumps({**m, "status": "success", "signal": signal}) + "\n"); success += 1
            except Exception as e:
                f.write(json.dumps({**m, "status": "error", "error": str(e)[:200]}) + "\n"); errors += 1

    output_vol.commit()
    print(f"Done: success={success} errors={errors} time={total_time/60:.1f}min")
    return json.dumps({"success": success, "errors": errors})


# ============================================================
# V8 GRPO Training
# ============================================================
@app.function(
    image=train_image, gpu="A100-80GB", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def train_v8_grpo():
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOTrainer, GRPOConfig
    from peft import PeftModel

    print("=" * 60)
    print("V8 GRPO: Starting from V8 SFT checkpoint")
    print("=" * 60)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL, max_seq_length=2048,
        dtype=torch.bfloat16, load_in_4bit=True, trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, V8_ADAPTER, is_trainable=True)
    print("Model loaded with V8 SFT adapter")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        output_dir="/outputs/quantum-alpha-v8-grpo",
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
    print("GRPO complete!")
    output_vol.commit()
    return json.dumps({"status": "complete", "steps": trainer.state.global_step})


@app.local_entrypoint()
def main():
    print("Launching V8 SFT predictions and V8 GRPO concurrently...")
    pred_handle = predict_v8_sft.spawn()
    grpo_handle = train_v8_grpo.spawn()
    print(f"V8 SFT predict: {pred_handle.object_id}")
    print(f"V8 GRPO: {grpo_handle.object_id}")
    print("Both spawned. Check Modal dashboard for progress.")
