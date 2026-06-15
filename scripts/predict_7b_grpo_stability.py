"""
7B GRPO Stability Check: Run predictions 3 times with different temperatures/seeds
to verify results aren't luck.

Usage:
    modal run scripts/predict_7b_grpo_stability.py
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-7b-grpo-stability")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("torch", "torchvision")
    .pip_install(
        "vllm>=0.12.0", "huggingface_hub", "transformers",
        "accelerate", "sentencepiece", "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

MERGED_MODEL = "/outputs/v7d-grpo-merged"
EVAL_FILE = "/outputs/articles_eval.jsonl"

SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

The quantum computing universe consists of these 10 tickers:
- IONQ, RGTI, QBTS, QUBT, QNT, IBM, HON (active, scored)
- MSFT, GOOGL, NVDA (always 0.0)

Score ranges: Pure-play [-2.0, +2.0], HON [-0.3, +0.3], IBM [-0.15, +0.15], MSFT/GOOGL/NVDA = 0.0

Output ONLY a valid JSON object with signal_vector containing all 10 tickers."""


def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.function(
    image=image, gpu="A10G", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_multiple_runs():
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print("Loading 7B GRPO merged model...")
    llm = LLM(model=MERGED_MODEL, trust_remote_code=True,
              max_model_len=2048, gpu_memory_utilization=0.90, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)
    print("Ready!")

    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                a = json.loads(line); a["idx"] = i; articles.append(a)

    prompts, meta = [], []
    for a in articles:
        text = clean_text(a.get("text", ""))[:2500]
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

    # Run 3 times with different settings
    configs = [
        {"name": "run1_temp03", "temp": 0.3, "seed": 42},
        {"name": "run2_temp03", "temp": 0.3, "seed": 123},
        {"name": "run3_temp01", "temp": 0.1, "seed": 42},
    ]

    for config in configs:
        print(f"\n{'='*60}")
        print(f"Run: {config['name']} (temp={config['temp']}, seed={config['seed']})")
        print(f"{'='*60}")

        sp = SamplingParams(temperature=config["temp"], top_p=0.9, max_tokens=1500, seed=config["seed"])
        start = time.time()
        outputs = llm.generate(prompts, sp)
        total_time = time.time() - start

        success, errors, pi = 0, 0, 0
        output_file = f"/outputs/predictions_7b_grpo_{config['name']}.jsonl"
        with open(output_file, "w") as f:
            for m in meta:
                if m.get("skip"):
                    f.write(json.dumps({**m, "status": "skipped"}) + "\n"); continue
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

        print(f"  Results: success={success} errors={errors} time={total_time/60:.1f}min")

    output_vol.commit()
    print("\nAll 3 runs complete!")
    return json.dumps({"status": "complete"})


@app.local_entrypoint()
def main():
    result = predict_multiple_runs.remote()
    print(f"Result: {result}")
