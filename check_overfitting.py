"""
Check if V7b/c results are due to overfitting.

The V7b/c training data was derived from the evaluation articles (the 270 that have return data).
If the IC improvement only appears on those 270 articles and NOT on the remaining ~151,
then we have overfitting.

We split the evaluation into:
- "In-training": articles that were used to generate V7b/c training data (270)
- "Out-of-sample": articles that were NOT used (the rest)
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from datetime import timedelta

DATA_DIR = Path("data")
EVAL_DIR = DATA_DIR / "eval"
MARKET_DIR = DATA_DIR / "market"

QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]
MARKET_TICKER = "SPY"

# Load the GRPO training article indices (these were used to train V7b/c)
training_indices = set()
with open("data/training/grpo_articles_with_returns.jsonl") as f:
    for line in f:
        if line.strip():
            d = json.loads(line)
            training_indices.add(d["article_idx"])

print(f"Training articles: {len(training_indices)}")
print(f"Total eval articles: 421")
print(f"Out-of-sample articles: {421 - len(training_indices)}")


def load_returns():
    returns = {}
    for ticker in QUANTUM_TICKERS + [MARKET_TICKER]:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            returns[ticker] = df[close_col].pct_change().dropna()
    return pd.DataFrame(returns)


def compute_ic_for_subset(predictions_file, returns_df, article_indices, horizon=10):
    """Compute IC only for articles in the given index set."""
    results = []
    with open(predictions_file) as f:
        for line in f:
            if not line.strip():
                continue
            p = json.loads(line)
            if p.get("status") != "success":
                continue
            idx = p.get("article_idx")
            if idx not in article_indices:
                continue

            date_str = p.get("date", "")
            signal = p.get("signal", {})
            sv = signal.get("signal_vector", {})
            if not date_str or not sv:
                continue
            try:
                event_date = pd.Timestamp(date_str)
            except:
                continue

            for ticker in QUANTUM_TICKERS:
                if ticker not in sv or ticker not in returns_df.columns:
                    continue
                val = sv[ticker]
                score = val.get("score", 0) if isinstance(val, dict) else val if isinstance(val, (int, float)) else 0
                if score == 0:
                    continue

                stock_returns = returns_df[ticker]
                market_returns = returns_df[MARKET_TICKER]
                pre_event = stock_returns[stock_returns.index < event_date]
                if len(pre_event) < 60:
                    continue
                gap_date = event_date - timedelta(days=14)
                est_stock = pre_event[pre_event.index < gap_date].tail(180)
                est_market = market_returns[market_returns.index < gap_date].tail(180)
                aligned = pd.concat([est_stock, est_market], axis=1).dropna()
                aligned.columns = ["stock", "market"]
                if len(aligned) < 60:
                    continue
                X = sm.add_constant(aligned["market"])
                try:
                    model = sm.OLS(aligned["stock"], X).fit()
                    alpha, beta = model.params.iloc[0], model.params.iloc[1]
                except:
                    continue
                post_stock = stock_returns[stock_returns.index > event_date]
                post_market = market_returns[market_returns.index > event_date]
                if len(post_stock) < horizon:
                    continue
                ws = post_stock.iloc[:horizon]
                wm = post_market.iloc[:horizon]
                idx_dates = ws.index.intersection(wm.index)
                if len(idx_dates) < 1:
                    continue
                expected = alpha + beta * wm.loc[idx_dates]
                ar = ws.loc[idx_dates] - expected
                car = ar.sum()
                results.append({"score": score, "car": car})

    if len(results) < 10:
        return None, None, len(results)
    df = pd.DataFrame(results)
    ic, pval = stats.pearsonr(df["score"], df["car"])
    return ic, pval, len(results)


returns_df = load_returns()
all_indices = set(range(421))
oos_indices = all_indices - training_indices

print("\n" + "=" * 70)
print("OVERFITTING CHECK: IC on in-training vs out-of-sample articles")
print("=" * 70)

models = {
    "V7b Rejection": EVAL_DIR / "predictions_v7b_rejection.jsonl",
    "V7c DPO": EVAL_DIR / "predictions_v7c_dpo.jsonl",
    "V4 Baseline": EVAL_DIR / "predictions_openreasoning7b_v4.jsonl",
}

for horizon in [5, 10, 20]:
    print(f"\n  Horizon: +{horizon}d")
    print(f"  {'Model':<22} {'In-Training IC':>15} {'OOS IC':>10} {'In-Train N':>11} {'OOS N':>7}")
    print(f"  {'-'*70}")

    for name, path in models.items():
        ic_in, p_in, n_in = compute_ic_for_subset(str(path), returns_df, training_indices, horizon)
        ic_oos, p_oos, n_oos = compute_ic_for_subset(str(path), returns_df, oos_indices, horizon)

        in_str = f"{ic_in:+.4f} (p={p_in:.3f})" if ic_in is not None else "N/A"
        oos_str = f"{ic_oos:+.4f} (p={p_oos:.3f})" if ic_oos is not None else "N/A"
        print(f"  {name:<22} {in_str:>15} {oos_str:>10} {n_in:>9} {n_oos:>7}")
