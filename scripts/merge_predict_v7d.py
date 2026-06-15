"""
V7d GRPO: Merge + Predict (standalone).

Usage:
    modal run scripts/merge_predict_v7d.py
"""

import modal
import json
import re
import time
import os

app = modal.App("alpha-signal-v7d-predict")

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
ADAPTER_PATH = "/outputs/alpha-signal-grpo-v7d-clean/checkpoint-184"
MERGED_OUTPUT = "/outputs/v7d-grpo-merged"
EVAL_FILE = "/outputs/articles_eval.jsonl"
PREDICTIONS_FILE = "/outputs/predictions_v7d_grpo_clean.jsonl"

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


@app.function(
    image=image, gpu="A10G", timeout=7200,
    volumes={"/root/.cache/huggingface": hf_cache_vol, "/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_and_predict():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Merge
    if not os.path.exists(MERGED_OUTPUT + "/config.json"):
        print("Merging GRPO adapter...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
        )
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        model = model.merge_and_unload()
        os.makedirs(MERGED_OUTPUT, exist_ok=True)
        model.save_pretrained(MERGED_OUTPUT)
        tokenizer.save_pretrained(MERGED_OUTPUT)
        output_vol.commit()
        del model
        torch.cuda.empty_cache()
        print("Merged!")
    else:
        print("Merged model exists.")

    # Predict
    print("Loading vLLM...")
    from vllm import LLM, SamplingParams

    llm = LLM(model=MERGED_OUTPUT, trust_remote_code=True,
              max_model_len=2048, gpu_memory_utilization=0.90, dtype="bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT, trust_remote_code=True)

    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)
    print(f"Loaded {len(articles)} articles")

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
    print(f"Generating {len(prompts)} predictions...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    print(f"Done in {total_time/60:.1f} minutes")

    success = 0
    errors = 0
    prompt_idx = 0

    with open(PREDICTIONS_FILE, "w") as out_f:
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
    print(f"Results: success={success} errors={errors} time={total_time/60:.1f}min")
    return json.dumps({"success": success, "errors": errors, "total_minutes": round(total_time/60, 1)})


@app.local_entrypoint()
def main():
    result = merge_and_predict.remote()
    print(f"\nResult: {result}")
