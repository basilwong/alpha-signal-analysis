"""
Batch Predictions: MiniCPM-2B Fine-tuned Model on Modal

Uses Unsloth's FastLanguageModel.for_inference() for optimized generation.
Processes all 421 evaluation articles in batches with volume commits.

Usage:
    modal run scripts/predict_minicpm_modal.py
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-minicpm-predict")

predict_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "datasets",
        "huggingface_hub",
        "trl",
        "transformers==4.57.3",
        "accelerate",
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        "pip install --no-deps unsloth unsloth-zoo",
        "pip install --no-deps cut-cross-entropy",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

BASE_MODEL = "openbmb/MiniCPM-2B-sft-bf16-llama-format"
ADAPTER_PATH = "/outputs/quantum-alpha-minicpm-2b/checkpoint-82"
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
    image=predict_image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_predictions():
    """Load merged model and run predictions on all evaluation articles."""
    import torch
    from unsloth import FastLanguageModel

    print("Loading base model + LoRA adapter from checkpoint...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=4096,
        dtype=torch.bfloat16,
        load_in_4bit=True,
        trust_remote_code=True,
    )
    # Apply the trained LoRA adapter
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    FastLanguageModel.for_inference(model)
    print(f"Model + adapter loaded on {torch.cuda.get_device_name(0)}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load evaluation articles
    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)
    print(f"Loaded {len(articles)} evaluation articles")

    # Check for existing results to resume
    completed = set()
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    completed.add(r.get("article_idx", -1))
        print(f"Resuming: {len(completed)} already completed")

    # Process articles
    success = 0
    errors = 0
    total_time = 0

    with open(OUTPUT_FILE, "a") as out_f:
        for i, article in enumerate(articles):
            if article["idx"] in completed:
                continue

            start = time.time()
            text = clean_text(article.get("text", ""))
            source = article.get("source", "news")
            title = article.get("title", "")
            date = article.get("date", "")

            if len(text) < 30:
                result = {"article_idx": article["idx"], "date": date, "title": title,
                          "source": source, "status": "skipped", "reason": "too short"}
                out_f.write(json.dumps(result) + "\n")
                continue

            source_inst = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
            user_msg = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text[:3000]}"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            try:
                inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True
                ).to(model.device)

                input_len = inputs.shape[-1]

                with torch.no_grad():
                    outputs = model.generate(
                        inputs, max_new_tokens=1500,
                        temperature=0.3, do_sample=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                gen_ids = outputs[0][input_len:]
                raw = tokenizer.decode(gen_ids, skip_special_tokens=True)

                # Extract JSON
                start_j = raw.find("{")
                end_j = raw.rfind("}") + 1
                if start_j >= 0 and end_j > start_j:
                    json_str = raw[start_j:end_j]
                    # Remove trailing commas
                    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
                    signal = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in output")

                elapsed = time.time() - start
                result = {"article_idx": article["idx"], "date": date, "title": title,
                          "source": source, "status": "success", "signal": signal,
                          "time_seconds": round(elapsed, 2)}
                success += 1

            except Exception as e:
                elapsed = time.time() - start
                result = {"article_idx": article["idx"], "date": date, "title": title,
                          "source": source, "status": "error", "error": str(e)[:200],
                          "time_seconds": round(elapsed, 2)}
                errors += 1

            out_f.write(json.dumps(result) + "\n")
            out_f.flush()
            total_time += elapsed

            processed = success + errors
            if processed % 20 == 0 or processed == 1:
                avg = total_time / max(processed, 1)
                remaining = (len(articles) - len(completed) - processed) * avg
                print(f"[{processed}/{len(articles) - len(completed)}] "
                      f"success={success} errors={errors} "
                      f"avg={avg:.1f}s ETA={remaining/60:.0f}min")

    # Commit results to volume
    output_vol.commit()
    print(f"\nDone! success={success} errors={errors} total_time={total_time/60:.1f}min")
    print(f"Results saved to {OUTPUT_FILE}")

    return json.dumps({"success": success, "errors": errors, "total_minutes": round(total_time/60, 1)})


@app.local_entrypoint()
def main():
    result = run_predictions.remote()
    print(f"\nPrediction result: {result}")
