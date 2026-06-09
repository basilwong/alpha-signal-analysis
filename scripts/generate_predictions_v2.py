"""
Batch Prediction V3: Cleaned inputs, per-article timing, resume support.

Fixes:
- Strips HTML tags and URLs from article text before inference
- Skips already-completed articles
- Times each individual article
- Saves timing data for analysis

Usage:
    # Upload articles to volume:
    modal volume put quantum-alpha-outputs data/raw/articles.jsonl articles.jsonl --force

    # Run (detached):
    modal run --detach scripts/generate_predictions_v2.py

    # Download results:
    modal volume get quantum-alpha-outputs predictions_v2_final.jsonl data/eval/predictions_v2_final.jsonl
"""

import modal
import json
import re
import time
import traceback

app = modal.App("quantum-alpha-predictions-v2")

predict_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

MODEL_ID = "basilwong/quantum-alpha-qwen3-8b"

SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

The quantum computing universe consists of these 9 tickers:
- IONQ: IonQ (trapped-ion, 100% quantum revenue)
- RGTI: Rigetti Computing (superconducting, 100% quantum revenue)
- QBTS: D-Wave Quantum (quantum annealing, 100% quantum revenue)
- QUBT: Quantum Computing Inc. (neutral atom, 100% quantum revenue)
- IBM: International Business Machines (superconducting, ~2% quantum revenue)
- GOOGL: Alphabet/Google (superconducting, <0.1% quantum revenue)
- MSFT: Microsoft (topological, <0.1% quantum revenue)
- HON: Honeywell/Quantinuum (trapped-ion, ~5% quantum revenue)
- NVDA: NVIDIA (adjacent/enabler, ~1% quantum revenue)

Key domain knowledge:
- Trapped-ion breakthroughs: bullish IONQ/HON, bearish RGTI/IBM/GOOGL
- Superconducting breakthroughs: bullish RGTI/IBM/GOOGL, bearish IONQ/HON
- Error correction advances: benefit ALL gate-based approaches
- Government funding: broadly bullish for entire sector
- Scale by revenue exposure: GOOGL/MSFT max +/-0.05, HON max +/-0.3, IBM max +/-0.15
- If the content is NOT related to quantum computing, assign all scores to 0.0

Output a valid JSON object:
{
    "signal_vector": {
        "IONQ": {"score": float, "reasoning": "1 sentence"},
        "RGTI": {"score": float, "reasoning": "1 sentence"},
        "QBTS": {"score": float, "reasoning": "1 sentence"},
        "QUBT": {"score": float, "reasoning": "1 sentence"},
        "IBM": {"score": float, "reasoning": "1 sentence"},
        "GOOGL": {"score": float, "reasoning": "1 sentence"},
        "MSFT": {"score": float, "reasoning": "1 sentence"},
        "HON": {"score": float, "reasoning": "1 sentence"},
        "NVDA": {"score": float, "reasoning": "1 sentence"}
    },
    "event_type": str,
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "signal_decay": "fast" | "medium" | "slow",
    "information_novelty": "high" | "medium" | "low",
    "technical_translation": "2-3 sentences.",
    "signal_rationale": "Why these scores?"
}

Output ONLY the JSON object. No markdown, no code blocks, no extra text."""

SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess novelty and likely decay speed.",
    "arxiv": "This is an academic paper abstract. Most are incremental. Only significant scores for genuine breakthroughs. Slow decay.",
    "sec_filing": "This is a regulatory filing. High reliability. Fast decay.",
    "press_release": "Company press release. Be skeptical.",
    "social_media": "Social media post. High noise, low reliability.",
    "earnings_call": "Earnings call. Forward guidance matters most.",
}


def clean_article_text(text: str) -> str:
    """
    Clean article text by removing HTML tags, URLs, and other artifacts
    that could confuse the model's JSON output.
    """
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove HTML entities
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', '', text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


@app.function(
    image=predict_image,
    gpu="A100",
    timeout=10800,  # 3 hours
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_all_predictions():
    """Fully remote: loads, cleans, predicts, saves with timing."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    articles_path = "/outputs/articles.jsonl"
    prev_results_path = "/outputs/predictions_v2.jsonl"
    output_path = "/outputs/predictions_v2_final.jsonl"

    # Load evaluation articles (proper walk-forward split)
    articles = []
    eval_path = "/outputs/articles_eval.jsonl"
    if os.path.exists(eval_path):
        with open(eval_path, "r") as f:
            for i, line in enumerate(f):
                if line.strip():
                    article = json.loads(line)
                    article["idx"] = i
                    articles.append(article)
    else:
        # Fallback to old method
        with open(articles_path, "r") as f:
            for i, line in enumerate(f):
                if line.strip() and i >= 200:
                    article = json.loads(line)
                    article["idx"] = i
                    articles.append(article)

    print(f"Total evaluation articles: {len(articles)}")

    # Load previous successful results to skip
    completed_indices = set()
    prev_successes = []
    try:
        with open(prev_results_path, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    if result.get("status") == "success":
                        completed_indices.add(result.get("article_idx"))
                        prev_successes.append(result)
        print(f"Previous successes to carry forward: {len(completed_indices)}")
    except FileNotFoundError:
        print("No previous results found, starting fresh")

    # Filter to remaining articles
    remaining = [a for a in articles if a["idx"] not in completed_indices]
    print(f"Remaining to process: {len(remaining)}")

    # Load model
    print(f"Loading model: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print(f"Model loaded on {torch.cuda.get_device_name(0)}")

    # Process articles
    success = 0
    errors = 0
    skipped = 0
    timing_data = []
    start_time = time.time()

    # Write previous successes first
    with open(output_path, "w") as f_out:
        for prev in prev_successes:
            f_out.write(json.dumps(prev) + "\n")

        for i, article in enumerate(remaining):
            article_start = time.time()

            text = article.get("text", "")
            source = article.get("source", "news")
            idx = article.get("idx", -1)
            title = article.get("title", "")
            date = article.get("date", "")

            # Clean the article text
            cleaned_text = clean_article_text(text)

            # Skip if cleaned text is too short
            if len(cleaned_text.strip()) < 30:
                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "skipped",
                    "reason": "text too short after cleaning",
                    "original_length": len(text),
                    "cleaned_length": len(cleaned_text),
                }
                f_out.write(json.dumps(result) + "\n")
                skipped += 1
                timing_data.append({"idx": idx, "time_s": 0, "status": "skipped"})
                continue

            source_instruction = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
            user_message = f"{source_instruction}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{cleaned_text}"

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ]

            try:
                inputs = tokenizer.apply_chat_template(
                    messages, return_tensors="pt", add_generation_prompt=True,
                    return_dict=True, enable_thinking=False
                ).to(model.device)

                input_length = inputs["input_ids"].shape[-1]

                with torch.no_grad():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=1024,
                        temperature=0.3,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                    )

                generated_ids = outputs[0][input_length:]
                raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

                # Strip thinking tags
                if "<think>" in raw_output:
                    parts = raw_output.split("</think>")
                    if len(parts) > 1:
                        raw_output = parts[-1].strip()

                # Parse JSON
                start = raw_output.find("{")
                end = raw_output.rfind("}") + 1
                if start != -1 and end > start:
                    signal = json.loads(raw_output[start:end])
                else:
                    signal = json.loads(raw_output)

                article_time = time.time() - article_start
                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "success", "signal": signal,
                    "input_tokens": input_length,
                    "output_tokens": len(generated_ids),
                    "time_seconds": round(article_time, 2),
                }
                f_out.write(json.dumps(result) + "\n")
                success += 1
                timing_data.append({"idx": idx, "time_s": article_time, "status": "success", "input_tokens": input_length, "output_tokens": len(generated_ids)})

            except Exception as e:
                article_time = time.time() - article_start
                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "error",
                    "error": str(e)[:300],
                    "input_tokens": input_length if 'input_length' in locals() else None,
                    "time_seconds": round(article_time, 2),
                    "cleaned_text_preview": cleaned_text[:200],
                }
                f_out.write(json.dumps(result) + "\n")
                errors += 1
                timing_data.append({"idx": idx, "time_s": article_time, "status": "error", "input_tokens": input_length if 'input_length' in locals() else 0})
                print(f"  ERROR [{idx}] {title[:50]}: {str(e)[:80]}")

            # Progress update every 10 articles
            if (i + 1) % 10 == 0:
                f_out.flush()
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                avg_time = sum(t["time_s"] for t in timing_data[-10:]) / min(10, len(timing_data[-10:]))
                print(f"  [{i+1}/{len(remaining)}] success={success} errors={errors} "
                      f"skipped={skipped} | avg: {avg_time:.1f}s/art | ETA: {eta/60:.1f} min")

    # Save timing data separately
    timing_path = "/outputs/prediction_timing.jsonl"
    with open(timing_path, "w") as f:
        for t in timing_data:
            f.write(json.dumps(t) + "\n")

    elapsed = time.time() - start_time
    total = success + errors + skipped + len(prev_successes)

    summary = {
        "status": "complete",
        "total_in_output": total,
        "new_success": success,
        "new_errors": errors,
        "new_skipped": skipped,
        "carried_forward": len(prev_successes),
        "elapsed_seconds": int(elapsed),
        "avg_time_per_article": round(elapsed / max(1, success + errors + skipped), 2),
    }

    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"  Carried forward (previous successes): {len(prev_successes)}")
    print(f"  New successes: {success}")
    print(f"  New errors: {errors}")
    print(f"  New skipped: {skipped}")
    print(f"  Total in output file: {total}")
    print(f"  Avg time per article: {summary['avg_time_per_article']}s")
    print(f"Results: {output_path}")
    print(f"Timing: {timing_path}")

    # Commit volume
    output_vol.commit()

    return json.dumps(summary)


@app.local_entrypoint()
def main():
    print("Starting fully-remote prediction run (cleaned inputs, with timing)...")
    print("Safe to disconnect. Results saved to Modal volume.")
    print("Monitor: modal app logs quantum-alpha-predictions-v2")
    print("Download: modal volume get quantum-alpha-outputs predictions_v2_final.jsonl data/eval/predictions_v2_final.jsonl")
    result = run_all_predictions.remote()
    print(f"\nResult: {result}")
