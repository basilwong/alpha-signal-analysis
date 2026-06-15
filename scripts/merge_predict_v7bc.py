"""
Merge + Predict for V7b (rejection) and V7c (DPO).
Runs both sequentially in one function to save container startup time.

Usage:
    modal run scripts/merge_predict_v7bc.py
"""

import modal
import json
import re
import time
import os

app = modal.App("alpha-signal-v7bc-predict")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "vllm>=0.12.0", "huggingface_hub", "transformers",
        "accelerate", "peft", "sentencepiece", "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-alpha-signal", create_if_missing=True)
output_vol = modal.Volume.from_name("alpha-signal-outputs", create_if_missing=True)

BASE_MODEL = "nvidia/OpenReasoning-Nemotron-7B"
EVAL_FILE = "/outputs/articles_eval.jsonl"

MODELS = {
    "v7b": {
        "adapter": "/outputs/alpha-signal-v7b-clean/checkpoint-30",
        "merged": "/outputs/v7b-clean-merged",
        "predictions": "/outputs/predictions_v7b_clean.jsonl",
    },
    "v7c": {
        "adapter": "/outputs/alpha-signal-v7c-dpo-clean/checkpoint-64",
        "merged": "/outputs/v7c-clean-merged",
        "predictions": "/outputs/predictions_v7c_clean.jsonl",
    },
}

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


def merge_model(base_model, adapter_path, output_path):
    """Merge adapter into base model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    if os.path.exists(output_path + "/config.json"):
        print(f"  {output_path} exists, skipping merge.")
        return

    print(f"  Merging {adapter_path}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    output_vol.commit()
    del model
    import torch
    torch.cuda.empty_cache()
    print(f"  Merged to {output_path}")


def run_predictions(model_path, output_file):
    """Run vLLM predictions."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print(f"  Loading vLLM from {model_path}...")
    llm = LLM(model=model_path, trust_remote_code=True,
              max_model_len=2048, gpu_memory_utilization=0.90, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)

    prompts = []
    article_meta = []
    for article in articles:
        text = clean_text(article.get("text", ""))
        source = article.get("source", "news")
        if len(text) < 30:
            article_meta.append({"idx": article["idx"], "skip": True,
                                 "date": article.get("date", ""), "title": article.get("title", ""),
                                 "source": source})
            continue
        source_inst = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        user_msg = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text[:2500]}"
        messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_msg}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        article_meta.append({"idx": article["idx"], "skip": False,
                             "date": article.get("date", ""), "title": article.get("title", ""),
                             "source": source})

    sampling_params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=1500)
    print(f"  Generating {len(prompts)} predictions...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time

    success = 0
    errors = 0
    prompt_idx = 0

    with open(output_file, "w") as out_f:
        for meta in article_meta:
            if meta.get("skip"):
                out_f.write(json.dumps({"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "skipped"}) + "\n")
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

                out_f.write(json.dumps({"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "success", "signal": signal}) + "\n")
                success += 1
            except Exception as e:
                out_f.write(json.dumps({"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "error", "error": str(e)[:200]}) + "\n")
                errors += 1

    output_vol.commit()
    print(f"  Done: success={success} errors={errors} time={total_time/60:.1f}min")
    return success, errors


@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_all():
    """Merge and predict for both V7b and V7c."""
    results = {}

    for name, config in MODELS.items():
        print(f"\n{'='*60}")
        print(f"Processing {name}")
        print(f"{'='*60}")

        # Merge
        merge_model(BASE_MODEL, config["adapter"], config["merged"])

        # Predict
        success, errors = run_predictions(config["merged"], config["predictions"])
        results[name] = {"success": success, "errors": errors}

    return json.dumps(results)


@app.local_entrypoint()
def main():
    result = run_all.remote()
    print(f"\nResult: {result}")
