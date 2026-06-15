"""
14B SFT Predictions: Merge adapter + vLLM inference.

Usage:
    modal run scripts/predict_14b_sft.py
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-14b-sft-predict")

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

BASE_MODEL = "nvidia/OpenReasoning-Nemotron-14B"
ADAPTER_PATH = "/outputs/quantum-alpha-14b-sft/checkpoint-100"
MERGED_OUTPUT = "/outputs/14b-sft-merged"
EVAL_FILE = "/outputs/articles_eval.jsonl"
PREDICTIONS_FILE = "/outputs/predictions_14b_sft.jsonl"

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


@app.function(
    image=image, gpu="A100-80GB", timeout=21600,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def predict():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from vllm import LLM, SamplingParams

    # Merge
    if not os.path.exists(MERGED_OUTPUT + "/config.json"):
        print("Merging 14B SFT adapter...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True)
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()
        os.makedirs(MERGED_OUTPUT, exist_ok=True)
        model.save_pretrained(MERGED_OUTPUT)
        tokenizer.save_pretrained(MERGED_OUTPUT)
        output_vol.commit()
        del model; torch.cuda.empty_cache()
        print("Merged!")
    else:
        print("Merged model exists.")

    # Predict
    print("Loading vLLM...")
    llm = LLM(model=MERGED_OUTPUT, trust_remote_code=True,
              max_model_len=12000, gpu_memory_utilization=0.92, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT, trust_remote_code=True)

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

    print(f"Generating {len(prompts)} predictions (max_tokens=10000)...")
    start = time.time()
    outputs = llm.generate(prompts, SamplingParams(temperature=0.3, top_p=0.9, max_tokens=10000))
    total_time = time.time() - start
    print(f"Done in {total_time/60:.1f} min")

    success, errors, pi = 0, 0, 0
    with open(PREDICTIONS_FILE, "w") as f:
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

    output_vol.commit()
    print(f"Results: success={success} errors={errors}")
    return json.dumps({"success": success, "errors": errors, "minutes": round(total_time/60, 1)})


@app.local_entrypoint()
def main():
    result = predict.remote()
    print(f"Result: {result}")
