"""
Fast Batch Predictions: MiniCPM-2B via vLLM Offline LLM Class

Uses the merged 16-bit model with vLLM's offline batch interface for maximum throughput.
Expected: 421 articles in ~30-60 minutes on A10G.

Usage:
    modal run scripts/predict_vllm_minicpm.py
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-minicpm-vllm-predict")

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install(
        "vllm==0.8.5",
        "huggingface_hub",
        "transformers==4.57.3",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

MERGED_MODEL = "/outputs/minicpm-2b-merged"
EVAL_FILE = "/outputs/articles_eval.jsonl"
OUTPUT_FILE = "/outputs/predictions_minicpm_v4.jsonl"

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
- MSFT: Microsoft — quantum revenue <0.1%, signal is noise
- GOOGL: Alphabet/Google — quantum revenue <0.1%, signal is noise
- NVDA: NVIDIA — moves on AI/GPU demand, not quantum news

Score ranges (MUST respect):
- Pure-play (IONQ, RGTI, QBTS, QUBT, QNT): [-2.0, +2.0]
- HON: [-0.3, +0.3]
- IBM: [-0.15, +0.15]
- MSFT, GOOGL, NVDA: always 0.0

Output a valid JSON object with this exact structure:
{
    "signal_vector": {
        "IONQ": {"score": float, "reasoning": "1-2 sentences"},
        "RGTI": {"score": float, "reasoning": "1-2 sentences"},
        "QBTS": {"score": float, "reasoning": "1-2 sentences"},
        "QUBT": {"score": float, "reasoning": "1-2 sentences"},
        "QNT": {"score": float, "reasoning": "1-2 sentences"},
        "IBM": {"score": float, "reasoning": "1-2 sentences"},
        "HON": {"score": float, "reasoning": "1-2 sentences"},
        "MSFT": {"score": 0.0, "reasoning": "Inactive"},
        "GOOGL": {"score": 0.0, "reasoning": "Inactive"},
        "NVDA": {"score": 0.0, "reasoning": "Inactive"}
    },
    "event_type": "descriptive event category",
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "information_novelty": "high" | "medium" | "low",
    "technical_translation": "2-3 sentences.",
    "signal_rationale": "Why these scores?",
    "chain_of_thought": "Step-by-step reasoning."
}

Output ONLY the JSON object. No additional text, no markdown, no code blocks."""

SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess novelty and likely decay speed.",
    "arxiv": "This is an academic paper abstract. Most are incremental. Only significant scores for genuine breakthroughs.",
    "sec_filing": "This is a regulatory filing. High reliability.",
    "press_release": "Company press release. Be skeptical.",
    "social_media": "Social media post. High noise, low reliability.",
    "earnings_call": "Earnings call. Forward guidance matters most.",
}


def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.function(
    image=vllm_image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
)
def run_predictions():
    """Run batch predictions using vLLM offline LLM class."""
    from vllm import LLM, SamplingParams

    print("Loading merged model with vLLM...")
    llm = LLM(
        model=MERGED_MODEL,
        trust_remote_code=True,
        max_model_len=4096,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )
    print("vLLM engine ready!")

    # Load evaluation articles
    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)
    print(f"Loaded {len(articles)} evaluation articles")

    # Load tokenizer for chat template formatting
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MERGED_MODEL, trust_remote_code=True)

    # Prepare all prompts using the SAME chat template as training
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
        user_msg = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text[:3000]}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # Use the tokenizer's chat template (matches training format exactly)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        prompts.append(prompt)
        article_meta.append({"idx": article["idx"], "skip": False,
                             "date": article.get("date", ""), "title": article.get("title", ""),
                             "source": source})

    print(f"Prepared {len(prompts)} prompts (skipped {sum(1 for m in article_meta if m.get('skip'))} short articles)")

    # Run batch inference
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=1500,
    )

    print("Starting batch generation...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    print(f"Generation complete in {total_time/60:.1f} minutes")

    # Process outputs
    success = 0
    errors = 0
    prompt_idx = 0

    with open(OUTPUT_FILE, "w") as out_f:
        for meta in article_meta:
            if meta.get("skip"):
                result = {"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "skipped", "reason": "too short"}
                out_f.write(json.dumps(result) + "\n")
                continue

            output = outputs[prompt_idx]
            prompt_idx += 1
            raw = output.outputs[0].text

            try:
                # Extract JSON
                start_j = raw.find("{")
                end_j = raw.rfind("}") + 1
                if start_j >= 0 and end_j > start_j:
                    json_str = raw[start_j:end_j]
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    signal = json.loads(json_str)
                else:
                    raise ValueError("No JSON found")

                result = {"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "success", "signal": signal}
                success += 1
            except Exception as e:
                result = {"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "error", "error": str(e)[:200]}
                errors += 1

            out_f.write(json.dumps(result) + "\n")

    # Commit results
    output_vol.commit()

    print(f"\nResults: success={success} errors={errors}")
    print(f"Total time: {total_time/60:.1f} minutes")
    print(f"Avg per article: {total_time/len(prompts):.1f} seconds")
    print(f"Output: {OUTPUT_FILE}")

    return json.dumps({
        "success": success,
        "errors": errors,
        "total_minutes": round(total_time/60, 1),
        "avg_seconds_per_article": round(total_time/len(prompts), 1),
    })


@app.local_entrypoint()
def main():
    result = run_predictions.remote()
    print(f"\nPrediction result: {result}")
