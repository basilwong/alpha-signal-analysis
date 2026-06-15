"""
Prepare GRPO training data from TRAINING articles only (NOT eval articles).

Extracts dates and article text from the V4 training data,
computes actual post-event returns, and saves for GRPO/rejection/DPO.

Usage:
    python scripts/prepare_grpo_data_from_train.py
"""
import json
import re
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from datetime import timedelta

DATA_DIR = Path("data")
MARKET_DIR = DATA_DIR / "market"
TRAIN_FILE = DATA_DIR / "training" / "alpha_signal_train_v4.jsonl"
OUTPUT_FILE = DATA_DIR / "training" / "grpo_train_articles_with_returns.jsonl"

QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "HON"]
MARKET_TICKER = "SPY"


def load_returns():
    returns = {}
    for ticker in QUANTUM_TICKERS + [MARKET_TICKER]:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            returns[ticker] = df[close_col].pct_change().dropna()
    return pd.DataFrame(returns)


def compute_car(event_date, ticker, returns_df, horizon=5):
    """Compute cumulative abnormal return."""
    stock_returns = returns_df[ticker]
    market_returns = returns_df[MARKET_TICKER]

    pre_event = stock_returns[stock_returns.index < event_date]
    if len(pre_event) < 60:
        return None

    gap_date = event_date - timedelta(days=14)
    est_stock = pre_event[pre_event.index < gap_date].tail(180)
    est_market = market_returns[market_returns.index < gap_date].tail(180)

    aligned = pd.concat([est_stock, est_market], axis=1).dropna()
    aligned.columns = ["stock", "market"]
    if len(aligned) < 60:
        return None

    X = sm.add_constant(aligned["market"])
    try:
        model = sm.OLS(aligned["stock"], X).fit()
        alpha, beta = model.params.iloc[0], model.params.iloc[1]
    except:
        return None

    post_stock = stock_returns[stock_returns.index > event_date]
    post_market = market_returns[market_returns.index > event_date]

    if len(post_stock) < horizon:
        return None
    ws = post_stock.iloc[:horizon]
    wm = post_market.iloc[:horizon]
    idx = ws.index.intersection(wm.index)
    if len(idx) < 1:
        return None
    expected = alpha + beta * wm.loc[idx]
    ar = ws.loc[idx] - expected
    return ar.sum()


def extract_article_text(user_content):
    """Extract the article text from the user message (after source instruction)."""
    # The user message typically has a source instruction then the article
    # Look for the article text after common markers
    markers = [
        "Analyze the following content and generate a cross-sectional signal vector:",
        "Analyze the following",
        "[ARTICLE]",
    ]
    for marker in markers:
        if marker in user_content:
            idx = user_content.index(marker) + len(marker)
            return user_content[idx:].strip()
    # Fallback: return everything after the first newline
    return user_content.split("\n", 1)[-1].strip() if "\n" in user_content else user_content


def main():
    print("Loading market data...")
    returns_df = load_returns()

    print(f"Loading training articles from {TRAIN_FILE}...")
    output_count = 0
    skipped_no_date = 0
    skipped_no_returns = 0

    with open(TRAIN_FILE) as f_in, open(OUTPUT_FILE, "w") as f_out:
        for i, line in enumerate(f_in):
            if not line.strip():
                continue
            example = json.loads(line)
            user_content = example["messages"][1]["content"]

            # Extract date
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', user_content)
            if not date_match:
                skipped_no_date += 1
                continue

            date_str = date_match.group(1)
            try:
                event_date = pd.Timestamp(date_str)
            except:
                skipped_no_date += 1
                continue

            # Compute actual returns
            actual_returns = {}
            has_data = False
            for ticker in QUANTUM_TICKERS:
                car = compute_car(event_date, ticker, returns_df, horizon=5)
                if car is not None:
                    actual_returns[ticker] = round(float(car), 6)
                    has_data = True
                else:
                    actual_returns[ticker] = None

            if not has_data:
                skipped_no_returns += 1
                continue

            # Extract article text for the prompt
            article_text = extract_article_text(user_content)
            if len(article_text) < 30:
                skipped_no_returns += 1
                continue

            # Clean
            article_text = re.sub(r'<[^>]+>', ' ', article_text)
            article_text = re.sub(r'https?://\S+', '', article_text)
            article_text = re.sub(r'&\w+;', ' ', article_text)
            article_text = re.sub(r'\s+', ' ', article_text).strip()
            article_text = article_text[:2500]

            source_inst = "This is a financial news article. Assess novelty and likely decay speed."
            prompt = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{article_text}"

            output = {
                "prompt": prompt,
                "actual_returns_5d": actual_returns,
                "date": date_str,
                "train_idx": i,
            }
            f_out.write(json.dumps(output) + "\n")
            output_count += 1

    print(f"\nDone!")
    print(f"  Output: {output_count} training articles with return data")
    print(f"  Skipped (no date): {skipped_no_date}")
    print(f"  Skipped (no returns): {skipped_no_returns}")
    print(f"  File: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
