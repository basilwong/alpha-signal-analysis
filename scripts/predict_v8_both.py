"""
Run predictions for both V8 SFT and V8 GRPO models.
Uses max_tokens=10000 to accommodate thinking chains.

Usage:
    modal run scripts/predict_v8_both.py
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-v8-predictions")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
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
V8_SFT_ADAPTER = "/outputs/quantum-alpha-v8-sft/checkpoint-42"
V8_GRPO_ADAPTER = "/outputs/quantum-alpha-v8-grpo/checkpoint-184"
V8_SFT_MERGED = "/outputs/v8-sft-merged-fixed"
V8_GRPO_MERGED = "/outputs/v8-grpo-merged"
EVAL_FILE = "/outputs/articles_eval.jsonl"

SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

The quantum computing universe consists of these 10 tickers:
- IONQ, RGTI, QBTS, QUBT, QNT, IBM, HON (active, scored)
- MSFT, GOOGL, NVDA (always 0.0)

Score ranges: Pure-play [-2.0, +2.0], HON [-0.3, +0.3], IBM [-0.15, +0.15], MSFT/GOOGL/NVDA = 0.0

After your reasoning, output a valid JSON object with signal_vector containing all 10 tickers with score and reasoning fields."""


def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def merge_model(adapter_path, output_path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if os.path.exists(output_path + "/config.json"):
        print(f"  {output_path} exists, skipping merge.")
        return

    print(f"  Merging {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    output_vol.commit()
    del model
    import torch as t
    t.cuda.empty_cache()
    print(f"  Merged!")


def run_predictions(model_path, output_file):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"  Loading vLLM from {model_path}...")
    llm = LLM(model=model_path, trust_remote_code=True,
              max_model_len=12000, gpu_memory_utilization=0.92, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                a = json.loads(line); a["idx"] = i; articles.append(a)

    prompts, meta = [], []
    for a in articles:
        text = clean_text(a.get("text", ""))[:2000]
        source = a.get("source", "news")
        if len(text) < 30:
            meta.append({"idx": a["idx"], "skip": True, "date": a.get("date",""),
                         "title": a.get("title",""), "source": source})
            continue
        si = "This is a financial news article." if source == "news" else "This is an academic paper abstract."
        msg = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{si}\n\n{text}"}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        meta.append({"idx": a["idx"], "skip": False, "date": a.get("date",""),
                     "title": a.get("title",""), "source": source})

    print(f"  Generating {len(prompts)} predictions (max_tokens=10000)...")
    start = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.3, top_p=0.9, max_tokens=10000))
    total_time = time.time() - start
    print(f"  Done in {total_time/60:.1f} min")

    success, errors, pi = 0, 0, 0
    with open(output_file, "w") as f:
        for m in meta:
            if m.get("skip"):
                f.write(json.dumps({**m, "status": "skipped"}) + "\n")
                continue
            raw = outputs[pi].outputs[0].text; pi += 1
            try:
                pt = raw
                if "<think>" in pt:
                    te = pt.find("</think>")
                    if te != -1: pt = pt[te+8:].strip()
                sj = pt.find("{"); ej = pt.rfind("}") + 1
                if sj >= 0 and ej > sj:
                    signal = json.loads(re.sub(r',\s*([}\]])', r'\1', pt[sj:ej]))
                else:
                    raise ValueError("No JSON")
                f.write(json.dumps({**m, "status": "success", "signal": signal}) + "\n")
                success += 1
            except Exception as e:
                f.write(json.dumps({**m, "status": "error", "error": str(e)[:200]}) + "\n")
                errors += 1

    output_vol.commit()
    print(f"  Results: success={success} errors={errors}")
    return success, errors


@app.function(
    image=image, gpu="A10G", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_v8_sft():
    print("=" * 60)
    print("V8 SFT Predictions")
    print("=" * 60)
    merge_model(V8_SFT_ADAPTER, V8_SFT_MERGED)
    s, e = run_predictions(V8_SFT_MERGED, "/outputs/predictions_v8_sft_fixed.jsonl")
    return json.dumps({"model": "v8_sft", "success": s, "errors": e})


@app.function(
    image=image, gpu="A10G", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_v8_grpo():
    print("=" * 60)
    print("V8 GRPO Predictions")
    print("=" * 60)
    merge_model(V8_GRPO_ADAPTER, V8_GRPO_MERGED)
    s, e = run_predictions(V8_GRPO_MERGED, "/outputs/predictions_v8_grpo.jsonl")
    return json.dumps({"model": "v8_grpo", "success": s, "errors": e})


@app.local_entrypoint()
def main():
    print("Launching V8 SFT and V8 GRPO predictions...")
    h_sft = predict_v8_sft.spawn()
    h_grpo = predict_v8_grpo.spawn()
    print(f"V8 SFT: {h_sft.object_id}")
    print(f"V8 GRPO: {h_grpo.object_id}")
    print("Both spawned.")
