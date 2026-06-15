"""
Batch Prediction: Processes articles in batches of 50 with volume commit after each batch.

Key fix: Commits volume every 50 articles so progress is never lost on timeout.
Also reads from the output file for resume support.

Usage:
    # Upload eval articles to volume:
    modal volume put alpha-signal-outputs data/raw/articles_eval.jsonl articles_eval.jsonl --force

    # Run (detached):
    modal run --detach scripts/generate_predictions_v2.py

    # Download results:
    modal volume get alpha-signal-outputs predictions_v2_final.jsonl data/eval/predictions_v2_final.jsonl
"""

import modal
import json
import re
import time
import os
import traceback

app = modal.App("alpha-signal-predictions-v2")

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

hf_cache_vol = modal.Volume.from_name("hf-cache-alpha-signal", create_if_missing=True)
output_vol = modal.Volume.from_name("alpha-signal-outputs", create_if_missing=True)

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
    """Strip HTML tags and URLs from article text."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


BATCH_SIZE = 50  # Commit volume every 50 articles


def repair_json(raw: str) -> dict:
    """
    Attempt to repair common JSON formatting issues from LLM output.
    Handles: trailing commas, unescaped newlines in strings, missing commas.
    """
    import re as _re
    # Remove trailing commas before } or ]
    fixed = _re.sub(r',\s*([}\]])', r'\1', raw)
    # Replace unescaped newlines inside strings
    # (between quotes, replace literal newlines with \n)
    fixed = _re.sub(r'(?<=")([^"]*?)\n([^"]*?)(?=")', lambda m: m.group(1) + '\\n' + m.group(2), fixed)
    # Try to fix missing commas between key-value pairs (}\n"key" -> },\n"key")
    fixed = _re.sub(r'(\})\s*\n\s*(")', r'\1,\n\2', fixed)
    # Fix missing comma after string value before next key
    fixed = _re.sub(r'("[^"]*")\s*\n\s*(")', r'\1,\n\2', fixed)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        # Last resort: try to extract just the signal_vector portion
        sv_start = fixed.find('"signal_vector"')
        if sv_start != -1:
            # Wrap in minimal valid JSON
            try:
                return json.loads('{' + fixed[sv_start:] + '}')
            except:
                pass
        raise


@app.function(
    image=predict_image,
    gpu="A100",
    timeout=7200,  # 2 hours (enough for 50 articles at 60s each = 50 min)
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_batch(batch_json: str):
    """
    Process a batch of articles. Saves results and commits volume after completion.
    Each batch is independent and self-contained.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    batch = json.loads(batch_json)
    batch_id = batch["batch_id"]
    articles = batch["articles"]
    output_path = f"/outputs/predictions_batch_{batch_id:03d}.jsonl"

    print(f"Batch {batch_id}: Processing {len(articles)} articles")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded on {torch.cuda.get_device_name(0)}")

    results = []
    success = 0
    errors = 0

    for i, article in enumerate(articles):
        article_start = time.time()
        text = article.get("text", "")
        source = article.get("source", "news")
        idx = article.get("idx", -1)
        title = article.get("title", "")
        date = article.get("date", "")

        cleaned_text = clean_article_text(text)

        if len(cleaned_text.strip()) < 30:
            results.append({
                "article_idx": idx, "date": date, "title": title,
                "source": source, "status": "skipped", "reason": "text too short",
            })
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
                    **inputs, max_new_tokens=1024, temperature=0.3, do_sample=True,
                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][input_length:]
            raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

            if "<think>" in raw_output:
                parts = raw_output.split("</think>")
                if len(parts) > 1:
                    raw_output = parts[-1].strip()

            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            json_str = raw_output[start:end] if (start != -1 and end > start) else raw_output
            try:
                signal = json.loads(json_str)
            except json.JSONDecodeError:
                # Attempt repair
                signal = repair_json(json_str)

            article_time = time.time() - article_start
            results.append({
                "article_idx": idx, "date": date, "title": title,
                "source": source, "status": "success", "signal": signal,
                "input_tokens": input_length, "output_tokens": len(generated_ids),
                "time_seconds": round(article_time, 2),
            })
            success += 1

        except Exception as e:
            article_time = time.time() - article_start
            results.append({
                "article_idx": idx, "date": date, "title": title,
                "source": source, "status": "error", "error": str(e)[:300],
                "time_seconds": round(article_time, 2),
            })
            errors += 1
            print(f"  ERROR [{idx}] {title[:50]}: {str(e)[:80]}")

        if (i + 1) % 10 == 0:
            print(f"  Batch {batch_id}: [{i+1}/{len(articles)}] success={success} errors={errors}")

    # Save results to volume
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    # COMMIT THE VOLUME - this is the critical fix
    output_vol.commit()
    print(f"Batch {batch_id} COMPLETE: {success} success, {errors} errors. Saved and committed.")

    return json.dumps({"batch_id": batch_id, "success": success, "errors": errors, "total": len(articles)})


@app.local_entrypoint()
def main():
    """Split articles into batches of 50 and process each batch sequentially."""

    # Load evaluation articles
    eval_path = "data/raw/articles_eval.jsonl"
    articles = []
    with open(eval_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)

    print(f"Total evaluation articles: {len(articles)}")

    # Check which batches are already done
    # (We can check by looking for batch files locally after downloading)

    # Split into batches of 50
    batches = []
    for i in range(0, len(articles), BATCH_SIZE):
        batch_articles = articles[i:i + BATCH_SIZE]
        batches.append({
            "batch_id": i // BATCH_SIZE,
            "articles": batch_articles,
        })

    print(f"Split into {len(batches)} batches of up to {BATCH_SIZE} articles each")
    print(f"Each batch takes ~50 minutes on A100")
    print(f"Processing sequentially (one batch at a time)...")

    # Skip already-completed batches (check if file exists locally)
    start_batch = 0
    for b in batches:
        batch_file = f"data/eval/predictions_batch_{b['batch_id']:03d}.jsonl"
        if os.path.exists(batch_file):
            print(f"Batch {b['batch_id']} already done locally, skipping.")
            start_batch = b['batch_id'] + 1
        else:
            break

    for batch in batches[start_batch:]:
        batch_id = batch["batch_id"]
        print(f"\n{'='*60}")
        print(f"Starting batch {batch_id} ({len(batch['articles'])} articles)")
        print(f"{'='*60}")

        result_json = run_batch.remote(json.dumps(batch))
        result = json.loads(result_json)
        print(f"Batch {batch_id} result: {result}")

    print(f"\n{'='*60}")
    print(f"ALL BATCHES COMPLETE")
    print(f"Download all results:")
    print(f"  for i in $(seq 0 {len(batches)-1}); do")
    print(f"    modal volume get alpha-signal-outputs predictions_batch_$(printf '%03d' $i).jsonl data/eval/predictions_batch_$(printf '%03d' $i).jsonl")
    print(f"  done")
