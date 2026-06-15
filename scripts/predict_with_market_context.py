"""
Prediction script with live market context.

At inference time, fetches the prior 5-day returns for each ticker
and prepends them to the user message, matching the training format.

Usage (on Modal):
    modal run scripts/predict_with_market_context.py

Usage (locally with llama.cpp):
    python scripts/predict_with_market_context.py --local --server http://localhost:8080
"""

import json
import re
import time
import os
import argparse
from pathlib import Path
from datetime import datetime, timedelta

try:
    import yfinance as yf
    import pandas as pd
except ImportError:
    print("Install: pip install yfinance pandas")
    exit(1)


TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"]

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

**Inactive (always 0.0, but reason about their impact on active tickers):**
- MSFT: Microsoft (topological approach)
- GOOGL: Alphabet/Google (superconducting approach)
- NVDA: NVIDIA (quantum hardware enabler)

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


def fetch_market_context(article_date: str, market_cache: dict = None) -> str:
    """
    Fetch prior 5-trading-day returns for all tickers.
    
    Uses yfinance for live data. Caches results to avoid repeated API calls.
    For historical articles, fetches the data as of the article date.
    """
    try:
        date = pd.Timestamp(article_date)
    except:
        return ""

    # Fetch 10 calendar days before the article to get 5 trading days
    start = (date - timedelta(days=12)).strftime("%Y-%m-%d")
    end = date.strftime("%Y-%m-%d")

    parts = []
    for ticker in TICKERS:
        try:
            # Check cache first
            cache_key = f"{ticker}_{start}_{end}"
            if market_cache is not None and cache_key in market_cache:
                ret_5d = market_cache[cache_key]
            else:
                df = yf.download(ticker, start=start, end=end, progress=False)
                if len(df) >= 2:
                    ret_5d = (df["Close"].iloc[-1] / df["Close"].iloc[0] - 1) * 100
                    if market_cache is not None:
                        market_cache[cache_key] = ret_5d
                else:
                    parts.append(f"{ticker}: N/A")
                    continue

            parts.append(f"{ticker}: {ret_5d:+.1f}%")
        except Exception:
            parts.append(f"{ticker}: N/A")

    if not parts:
        return ""

    return "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\n" + " | ".join(parts)


def fetch_market_context_from_parquet(article_date: str, market_dir: str = "data/market") -> str:
    """
    Compute market context from local parquet files (for offline/eval use).
    """
    try:
        date = pd.Timestamp(article_date)
    except:
        return ""

    parts = []
    for ticker in TICKERS:
        path = Path(market_dir) / f"{ticker}.parquet"
        if not path.exists():
            parts.append(f"{ticker}: N/A")
            continue

        df = pd.read_parquet(path)
        close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
        prices = df[close_col].sort_index()
        prior = prices[prices.index < date].tail(6)

        if len(prior) >= 2:
            ret_5d = (prior.iloc[-1] / prior.iloc[0] - 1) * 100
            parts.append(f"{ticker}: {ret_5d:+.1f}%")
        else:
            parts.append(f"{ticker}: N/A")

    return "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\n" + " | ".join(parts)


def build_user_message(article: dict, market_context: str) -> str:
    """Construct the full user message with market context prepended."""
    text = article.get("text", "")
    # Clean text
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text[:2500]  # Truncate to fit context window

    source = article.get("source", "news")
    source_inst = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])

    parts = []
    if market_context:
        parts.append(market_context)
    parts.append(f"{source_inst}")
    parts.append(f"\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text}")

    return "\n\n".join(parts)


def extract_signal(raw_output: str) -> dict:
    """Extract JSON signal, handling optional thinking blocks."""
    if "<think>" in raw_output:
        think_end = raw_output.find("</think>")
        if think_end != -1:
            raw_output = raw_output[think_end + len("</think>"):].strip()
        else:
            first_brace = raw_output.find("{")
            if first_brace > 0:
                raw_output = raw_output[first_brace:]

    start = raw_output.find("{")
    end = raw_output.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")

    json_str = re.sub(r',\s*([}\]])', r'\1', raw_output[start:end])
    return json.loads(json_str)


def apply_threshold(signal_vector: dict, threshold: float = 0.5) -> dict:
    """Zero out scores below confidence threshold. Applied post-prediction."""
    filtered = {}
    for ticker, data in signal_vector.items():
        score = data.get("score", 0)
        if isinstance(score, (int, float)) and abs(score) < threshold:
            filtered[ticker] = {"score": 0.0, "reasoning": data.get("reasoning", "")}
        else:
            filtered[ticker] = data
    return filtered


# =========================================================
# LOCAL INFERENCE MODE (llama.cpp server)
# =========================================================

def predict_local(articles: list, server_url: str, output_path: str, use_parquet: bool = True):
    """Run predictions against a local llama.cpp server."""
    import requests

    market_cache = {}
    success = 0
    errors = 0

    with open(output_path, "w") as f_out:
        for i, article in enumerate(articles):
            date = article.get("date", "")
            title = article.get("title", "")
            source = article.get("source", "news")

            # Get market context
            if use_parquet:
                market_context = fetch_market_context_from_parquet(date)
            else:
                market_context = fetch_market_context(date, market_cache)

            user_msg = build_user_message(article, market_context)

            if len(user_msg.strip()) < 50:
                result = {"article_idx": i, "date": date, "title": title,
                          "source": source, "status": "skipped", "reason": "too short"}
                f_out.write(json.dumps(result) + "\n")
                continue

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ]

            start_time = time.time()
            try:
                resp = requests.post(
                    f"{server_url}/v1/chat/completions",
                    json={"messages": messages, "max_tokens": 3000, "temperature": 0.3},
                    timeout=300,
                )
                resp.raise_for_status()
                raw = resp.json()["choices"][0]["message"]["content"]
                signal = extract_signal(raw)
                elapsed = time.time() - start_time

                result = {"article_idx": i, "date": date, "title": title,
                          "source": source, "status": "success", "signal": signal,
                          "time_seconds": round(elapsed, 2)}
                success += 1
            except Exception as e:
                elapsed = time.time() - start_time
                result = {"article_idx": i, "date": date, "title": title,
                          "source": source, "status": "error", "error": str(e)[:200],
                          "time_seconds": round(elapsed, 2)}
                errors += 1

            f_out.write(json.dumps(result) + "\n")
            f_out.flush()

            processed = success + errors
            if processed % 10 == 0 or processed == 1:
                avg = (time.time() - start_time) if processed == 1 else elapsed
                print(f"[{processed}/{len(articles)}] success={success} errors={errors}")

    print(f"\nDone! success={success} errors={errors}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true", help="Run against local llama.cpp server")
    parser.add_argument("--server", default="http://localhost:8080", help="llama.cpp server URL")
    parser.add_argument("--input", default="data/raw/articles_eval.jsonl", help="Input articles")
    parser.add_argument("--output", default="data/eval/predictions_v5_with_context.jsonl", help="Output file")
    parser.add_argument("--limit", type=int, default=0, help="Limit articles (0=all)")
    parser.add_argument("--live-market", action="store_true", help="Use yfinance instead of parquet files")
    args = parser.parse_args()

    # Load articles
    articles = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    if args.limit > 0:
        articles = articles[:args.limit]

    print(f"Loaded {len(articles)} articles")

    if args.local:
        predict_local(articles, args.server, args.output, use_parquet=not args.live_market)
    else:
        print("For Modal inference, use scripts/predict_vllm_minicpm.py or scripts/merge_predict_nemotron7b.py")
        print("This script is for local llama.cpp inference.")
