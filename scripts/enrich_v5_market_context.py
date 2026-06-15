"""
Enrich V5 training data with pre-event market context.

For each training example, computes the prior 5-day return for each ticker
and prepends it to the user message. This gives the model information about
recent price action so it can judge whether news is already priced in.

Usage:
    python scripts/enrich_v5_market_context.py

Input:  data/training/quantum_alpha_train_v5.jsonl
Output: data/training/quantum_alpha_train_v5_enriched.jsonl
"""

import json
import pandas as pd
import re
from pathlib import Path

MARKET_DIR = Path("data/market")
INPUT_FILE = "data/training/quantum_alpha_train_v5.jsonl"
OUTPUT_FILE = "data/training/quantum_alpha_train_v5_enriched.jsonl"

TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"]


def load_market_data():
    """Load all ticker price data into memory."""
    market = {}
    for ticker in TICKERS:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            market[ticker] = df[close_col].sort_index()
        else:
            print(f"  WARNING: No market data for {ticker}")
    return market


def compute_market_context(article_date: str, market: dict) -> str:
    """Compute prior 5-trading-day returns for all tickers."""
    try:
        date = pd.Timestamp(article_date)
    except:
        return ""

    parts = []
    for ticker in TICKERS:
        if ticker not in market:
            parts.append(f"{ticker}: N/A")
            continue

        prices = market[ticker]
        prior = prices[prices.index < date].tail(6)

        if len(prior) >= 2:
            ret_5d = (prior.iloc[-1] / prior.iloc[0] - 1) * 100
            parts.append(f"{ticker}: {ret_5d:+.1f}%")
        else:
            parts.append(f"{ticker}: N/A")

    return "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\n" + " | ".join(parts)


def extract_date_from_user_message(user_content: str) -> str:
    """Try to extract an article date from the user message."""
    # Look for ISO date format
    match = re.search(r'(\d{4}-\d{2}-\d{2})', user_content)
    if match:
        return match.group(1)
    return ""


def main():
    print("Loading market data...")
    market = load_market_data()
    print(f"  Loaded {len(market)} tickers")

    print(f"\nProcessing {INPUT_FILE}...")
    enriched = 0
    skipped = 0
    total = 0

    with open(INPUT_FILE) as f_in, open(OUTPUT_FILE, "w") as f_out:
        for line in f_in:
            if not line.strip():
                continue
            total += 1
            example = json.loads(line)
            msgs = example["messages"]

            user_content = msgs[1]["content"]

            # Extract date from user message
            date = extract_date_from_user_message(user_content)

            if not date:
                # Try to find date in the article text itself
                date_match = re.search(r'(20\d{2}-\d{2}-\d{2})', user_content)
                if date_match:
                    date = date_match.group(1)

            if date:
                context = compute_market_context(date, market)
                if context and "N/A" not in context:
                    # Prepend market context to user message
                    msgs[1]["content"] = context + "\n\n" + user_content
                    enriched += 1
                else:
                    skipped += 1
            else:
                skipped += 1

            f_out.write(json.dumps(example) + "\n")

    print(f"\nDone!")
    print(f"  Total: {total}")
    print(f"  Enriched with market context: {enriched} ({enriched/total*100:.1f}%)")
    print(f"  Skipped (no date or no market data): {skipped} ({skipped/total*100:.1f}%)")
    print(f"  Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
