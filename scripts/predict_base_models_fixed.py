"""
Re-run base model predictions for 14B and 32B with max_tokens=10000.
The base reasoning models need more output budget for <think> + JSON.

Usage:
    modal run scripts/predict_base_models_fixed.py
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-base-predictions-fixed")

image = (
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

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

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

After your reasoning, output a valid JSON object with signal_vector containing all 10 tickers with score and reasoning fields. The JSON must be valid and parseable."""

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


def run_base_predictions(model_name, output_file, max_model_len=4096):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"Loading {model_name}...")
    llm = LLM(model=model_name, trust_remote_code=True,
              max_model_len=max_model_len, gpu_memory_utilization=0.92, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print("Ready!")

    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                a = json.loads(line); a["idx"] = i; articles.append(a)

    # Sanity check with 2 articles
    test_prompts = []
    for a in articles[:2]:
        text = clean_text(a.get("text", ""))[:2000]
        source = a.get("source", "news")
        si = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        msg = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{si}\n\n{text}"}]
        test_prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))

    # Use high max_tokens to accommodate thinking + JSON
    test_out = llm.generate(test_prompts, SamplingParams(temperature=0.3, top_p=0.9, max_tokens=10000))
    for i, o in enumerate(test_out):
        raw = o.outputs[0].text
        tok_count = len(raw.split())
        has_think = "<think>" in raw
        has_json = "{" in raw and "signal_vector" in raw
        print(f"  Sanity {i}: {tok_count} words, think={has_think}, json={has_json}")
        if has_json:
            # Try to parse
            text = raw
            if "<think>" in text:
                te = text.find("</think>")
                if te != -1: text = text[te+8:].strip()
            sj = text.find("{"); ej = text.rfind("}")+1
            if sj >= 0 and ej > sj:
                try:
                    json.loads(re.sub(r',\s*([}\]])', r'\1', text[sj:ej]))
                    print(f"    -> Valid JSON!")
                except Exception as e:
                    print(f"    -> JSON parse error: {e}")

    # Full batch
    prompts, meta = [], []
    for a in articles:
        text = clean_text(a.get("text", ""))
        source = a.get("source", "news")
        if len(text) < 30:
            meta.append({"idx": a["idx"], "skip": True, "date": a.get("date",""), "title": a.get("title",""), "source": source})
            continue
        si = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        msg = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": f"{si}\n\n{text[:2000]}"}]
        prompts.append(tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        meta.append({"idx": a["idx"], "skip": False, "date": a.get("date",""), "title": a.get("title",""), "source": source})

    print(f"Generating {len(prompts)} predictions with max_tokens=10000...")
    start = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.3, top_p=0.9, max_tokens=10000))
    total_time = time.time() - start
    print(f"Done in {total_time/60:.1f} min")

    success, errors, pi = 0, 0, 0
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
                sj = pt.find("{"); ej = pt.rfind("}")+1
                if sj >= 0 and ej > sj:
                    json_str = re.sub(r',\s*([}\]])', r'\1', pt[sj:ej])
                    signal = json.loads(json_str)
                else:
                    raise ValueError("No JSON found after thinking")
                f.write(json.dumps({**m, "status": "success", "signal": signal}) + "\n"); success += 1
            except Exception as e:
                f.write(json.dumps({**m, "status": "error", "error": str(e)[:200]}) + "\n"); errors += 1

    output_vol.commit()
    print(f"Results: success={success} errors={errors}")
    return json.dumps({"success": success, "errors": errors, "minutes": round(total_time/60, 1)})


@app.function(
    image=image, gpu="A100-80GB", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_14b():
    return run_base_predictions(
        "nvidia/OpenReasoning-Nemotron-14B",
        "/outputs/predictions_base_14b_fixed.jsonl",
        max_model_len=12000,  # Need room for 2000 input + 10000 output
    )


@app.function(
    image=image, gpu="A100-80GB", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict_32b():
    return run_base_predictions(
        "nvidia/OpenReasoning-Nemotron-32B",
        "/outputs/predictions_base_32b_fixed.jsonl",
        max_model_len=12000,
    )


@app.local_entrypoint()
def main():
    print("Launching 14B and 32B base predictions with max_tokens=10000...")
    h14 = predict_14b.spawn()
    h32 = predict_32b.spawn()
    print(f"14B: {h14.object_id}")
    print(f"32B: {h32.object_id}")
    print("Both spawned.")
