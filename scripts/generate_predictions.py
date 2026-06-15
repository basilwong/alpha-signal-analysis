"""
Batch Prediction Script V3: Fully remote execution.

The entire pipeline runs on Modal's GPU. Articles are uploaded to the volume,
processed remotely, and results saved to the volume. No dependency on local
client staying connected.

Usage:
    # Upload articles to volume first:
    modal volume put alpha-signal-outputs data/raw/articles.jsonl articles.jsonl --force

    # Run predictions (detached - survives local disconnect):
    modal run --detach scripts/generate_predictions.py

    # Check progress:
    modal volume ls alpha-signal-outputs

    # Download results when done:
    modal volume get alpha-signal-outputs predictions_v2.jsonl data/eval/predictions_v2.jsonl
"""

import modal
import json
import traceback
import time

app = modal.App("alpha-signal-predictions")

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

Output ONLY the JSON object."""

SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess novelty and likely decay speed.",
    "arxiv": "This is an academic paper abstract. Most are incremental. Only significant scores for genuine breakthroughs. Slow decay.",
    "sec_filing": "This is a regulatory filing. High reliability. Fast decay.",
    "press_release": "Company press release. Be skeptical.",
    "social_media": "Social media post. High noise, low reliability.",
    "earnings_call": "Earnings call. Forward guidance matters most.",
}


@app.function(
    image=predict_image,
    gpu="A100",
    timeout=7200,  # 2 hours
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_all_predictions():
    """
    Fully remote: loads articles from volume, runs predictions, saves results to volume.
    No dependency on local client.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load articles from volume
    articles_path = "/outputs/articles.jsonl"
    output_path = "/outputs/predictions_v2.jsonl"

    articles = []
    with open(articles_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip() and i >= 200:  # Skip first 200 (training set)
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)

    print(f"Loaded {len(articles)} evaluation articles (indices 200+)")

    # Check for existing progress
    completed_indices = set()
    try:
        with open(output_path, "r") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    completed_indices.add(result.get("article_idx"))
        print(f"Resuming: {len(completed_indices)} already completed")
    except FileNotFoundError:
        print("Starting fresh (no previous results)")

    remaining = [a for a in articles if a["idx"] not in completed_indices]
    print(f"Remaining to process: {len(remaining)}")

    if not remaining:
        print("All articles already processed!")
        return json.dumps({"status": "complete", "total": len(articles)})

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
    start_time = time.time()

    with open(output_path, "a") as f_out:
        for i, article in enumerate(remaining):
            text = article.get("text", "")
            source = article.get("source", "news")
            idx = article.get("idx", -1)
            title = article.get("title", "")
            date = article.get("date", "")

            # Skip very short articles
            if len(text.strip()) < 30:
                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "skipped", "reason": "text too short"
                }
                f_out.write(json.dumps(result) + "\n")
                skipped += 1
                continue

            source_instruction = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
            user_message = f"{source_instruction}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text}"

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

                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "success", "signal": signal,
                    "input_tokens": input_length,
                }
                f_out.write(json.dumps(result) + "\n")
                success += 1

            except Exception as e:
                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "error",
                    "error": str(e)[:300],
                    "input_tokens": input_length if 'input_length' in locals() else None,
                }
                f_out.write(json.dumps(result) + "\n")
                errors += 1
                print(f"  ERROR [{idx}] {title[:50]}: {str(e)[:100]}")

            # Flush every 10 articles and print progress
            if (i + 1) % 10 == 0:
                f_out.flush()
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                eta = (len(remaining) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1}/{len(remaining)}] success={success} errors={errors} "
                      f"skipped={skipped} | {rate:.1f} art/s | ETA: {eta/60:.1f} min")

    elapsed = time.time() - start_time
    summary = {
        "status": "complete",
        "total_processed": success + errors + skipped,
        "success": success,
        "errors": errors,
        "skipped": skipped,
        "elapsed_seconds": int(elapsed),
        "output_path": output_path,
    }
    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed/60:.1f} minutes")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")
    print(f"Results saved to volume: {output_path}")

    # Commit volume changes
    output_vol.commit()

    return json.dumps(summary)


@app.local_entrypoint()
def main():
    """Just triggers the remote function. Can disconnect safely."""
    print("Starting fully-remote prediction run...")
    print("Safe to disconnect. Results will be saved to Modal volume.")
    print("Check progress: modal app logs alpha-signal-predictions")
    print("Download results: modal volume get alpha-signal-outputs predictions_v2.jsonl data/eval/predictions_v2.jsonl")
    result = run_all_predictions.remote()
    print(f"\nResult: {result}")
