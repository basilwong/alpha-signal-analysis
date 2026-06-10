"""
Generate predictions using Qwen Cloud API (base models, no fine-tuning).

Runs qwen3-8b and qwen3-32b through the DashScope API for comparison
against our fine-tuned model. Free tier: 1M tokens per model.

Usage:
    python scripts/generate_predictions_api.py --model qwen3-8b --output data/eval/predictions_qwen3_8b_base.jsonl
    python scripts/generate_predictions_api.py --model qwen3-32b --output data/eval/predictions_qwen3_32b_base.jsonl
"""

import json
import re
import time
import argparse
from pathlib import Path
from openai import OpenAI

# API Configuration (Singapore region)
API_KEY = "sk-ws-H.IIMPYP.OVEd.MEYCIQCgnJiyfu3TI7aOMuMio4dSrWTf5zbFNrCpKP-NTyUGagIhAJQ6AGEG4uC8C9LmDEqJCLQGSUnilOLV6lQ1QR7QvVBi"
BASE_URL = "https://ws-wuyspztgv1cyxvbr.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"

MODEL_MAP = {
    "qwen3-8b": "qwen3-8b",
    "qwen3-32b": "qwen3-32b",
    "qwen3-max": "qwen3-max",
    "qwen3.7-max": "qwen3.7-max-2026-05-20",
}

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
    """Strip HTML tags and URLs."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def run_inference(client, model_name, text, source):
    """Run inference through the API. Returns (signal_dict, latency_ms)."""
    cleaned = clean_article_text(text)
    if len(cleaned.strip()) < 30:
        return None, 0

    instruction = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
    user_message = f"{instruction}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{cleaned}"

    start = time.time()
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=0.3,
            max_tokens=1024,
            extra_body={"enable_thinking": False},
        )

        raw_output = response.choices[0].message.content
        latency_ms = int((time.time() - start) * 1000)

        # Strip thinking tags if present
        if "<think>" in raw_output:
            parts = raw_output.split("</think>")
            if len(parts) > 1:
                raw_output = parts[-1].strip()

        # Parse JSON
        s = raw_output.find("{")
        e = raw_output.rfind("}") + 1
        if s != -1 and e > s:
            signal = json.loads(raw_output[s:e])
        else:
            signal = json.loads(raw_output)

        return signal, latency_ms

    except Exception as ex:
        latency_ms = int((time.time() - start) * 1000)
        return {"error": str(ex)[:200]}, latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(MODEL_MAP.keys()))
    parser.add_argument("--output", required=True)
    parser.add_argument("--input", default="data/raw/articles_eval.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Skip already-processed articles")
    args = parser.parse_args()

    model_name = MODEL_MAP[args.model]
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    # Load articles
    articles = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)

    if args.limit:
        articles = articles[:args.limit]

    print(f"Model: {model_name}")
    print(f"Articles: {len(articles)}")
    print(f"Output: {args.output}")

    # Resume support
    completed_indices = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("status") == "success":
                        completed_indices.add(r.get("article_idx"))
        print(f"Resuming: {len(completed_indices)} already done")

    remaining = [a for a in articles if a["idx"] not in completed_indices]
    print(f"Remaining: {len(remaining)}")

    # Process
    success = 0
    errors = 0
    skipped = 0
    start_time = time.time()

    mode = "a" if args.resume else "w"
    with open(args.output, mode) as f_out:
        for i, article in enumerate(remaining):
            text = article.get("text", "")
            source = article.get("source", "news")
            idx = article.get("idx", -1)
            title = article.get("title", "")
            date = article.get("date", "")

            signal, latency_ms = run_inference(client, model_name, text, source)

            if signal is None:
                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "skipped", "reason": "text too short",
                }
                skipped += 1
            elif "error" in signal:
                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "error",
                    "error": signal["error"], "time_ms": latency_ms,
                }
                errors += 1
            else:
                result = {
                    "article_idx": idx, "date": date, "title": title,
                    "source": source, "status": "success", "signal": signal,
                    "time_ms": latency_ms,
                }
                success += 1

            f_out.write(json.dumps(result) + "\n")

            if (i + 1) % 10 == 0:
                f_out.flush()
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                print(f"  [{i+1}/{len(remaining)}] success={success} errors={errors} "
                      f"skipped={skipped} | {rate:.0f} art/min")

            # Rate limit: 1 second between calls
            time.sleep(1)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COMPLETE ({model_name}): {elapsed/60:.1f} min")
    print(f"  Success: {success}, Errors: {errors}, Skipped: {skipped}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
