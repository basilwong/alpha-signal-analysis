"""
Prepare GRPO training data: articles paired with actual post-event returns.

For each evaluation article that has 5-day return data, create a GRPO prompt.
The reward function will score model outputs against these actual returns.

Output: data/training/grpo_articles_with_returns.jsonl
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
import statsmodels.api as sm
from datetime import timedelta

DATA_DIR = Path("data")
MARKET_DIR = DATA_DIR / "market"
EVAL_FILE = DATA_DIR / "raw" / "articles_eval.jsonl"
OUTPUT_FILE = DATA_DIR / "training" / "grpo_articles_with_returns.jsonl"

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
    """Compute cumulative abnormal return for a ticker after event_date."""
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


def main():
    print("Loading market data...")
    returns_df = load_returns()

    print("Loading evaluation articles...")
    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)

    print(f"Processing {len(articles)} articles...")
    output_count = 0
    skipped = 0

    with open(OUTPUT_FILE, "w") as f_out:
        for article in articles:
            date_str = article.get("date", "")
            if not date_str:
                skipped += 1
                continue

            try:
                event_date = pd.Timestamp(date_str)
            except:
                skipped += 1
                continue

            # Compute actual returns for all tickers
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
                skipped += 1
                continue

            # Build the prompt (same format as inference)
            text = article.get("text", "")
            if len(text) < 30:
                skipped += 1
                continue

            source = article.get("source", "news")
            source_instructions = {
                "news": "This is a financial news article. Assess novelty and likely decay speed.",
                "arxiv": "This is an academic paper abstract. Most are incremental. Only significant scores for genuine breakthroughs.",
            }
            source_inst = source_instructions.get(source, source_instructions["news"])

            # Clean text
            import re
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'https?://\S+', '', text)
            text = re.sub(r'&\w+;', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            text = text[:2500]

            user_msg = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text}"

            output = {
                "prompt": user_msg,
                "actual_returns_5d": actual_returns,
                "date": date_str,
                "source": source,
                "article_idx": article["idx"],
            }

            f_out.write(json.dumps(output) + "\n")
            output_count += 1

    print(f"\nDone!")
    print(f"  Output: {output_count} articles with return data")
    print(f"  Skipped: {skipped} (no date, no return data, or too short)")
    print(f"  File: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
