"""
Analyze IC by article source type for OpenReasoning-Nemotron-7B.
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import statsmodels.api as sm
from datetime import timedelta
from collections import defaultdict

DATA_DIR = Path("data")
EVAL_DIR = DATA_DIR / "eval"
MARKET_DIR = DATA_DIR / "market"

QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]
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


def load_predictions(path):
    preds = []
    with open(path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p.get("status") == "success" or p.get("success") == True:
                    preds.append(p)
    return preds


def compute_observations(predictions, returns_df, horizon=5):
    """Compute predicted score vs CAR for each prediction at given horizon."""
    results = []
    for pred in predictions:
        date_str = pred.get("date", "")
        signal = pred.get("signal", {})
        signal_vector = signal.get("signal_vector", {})
        source = pred.get("source", "unknown")
        if not date_str or not signal_vector:
            continue
        try:
            event_date = pd.Timestamp(date_str)
        except:
            continue

        for ticker in QUANTUM_TICKERS:
            if ticker not in signal_vector or ticker not in returns_df.columns:
                continue
            predicted_score = signal_vector[ticker].get("score", 0)
            if predicted_score == 0:
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
            idx = ws.index.intersection(wm.index)
            if len(idx) < 1:
                continue
            expected = alpha + beta * wm.loc[idx]
            ar = ws.loc[idx] - expected
            car = ar.sum()

            results.append({
                "source": source,
                "ticker": ticker,
                "predicted_score": predicted_score,
                "car": car,
                "date": date_str,
                "article_idx": pred.get("article_idx"),
            })

    return pd.DataFrame(results)


# Load data
print("Loading market data...")
returns_df = load_returns()

# Analyze Nemotron
print("Loading Nemotron predictions...")
nemotron_preds = load_predictions(EVAL_DIR / "predictions_openreasoning7b_v4.jsonl")

print("Computing observations at +5d, +10d, +20d...")
obs_5d = compute_observations(nemotron_preds, returns_df, horizon=5)
obs_10d = compute_observations(nemotron_preds, returns_df, horizon=10)
obs_20d = compute_observations(nemotron_preds, returns_df, horizon=20)

print(f"\nTotal observations: {len(obs_5d)} at 5d, {len(obs_10d)} at 10d, {len(obs_20d)} at 20d")

# IC by source
print("\n" + "=" * 70)
print("IC BY SOURCE TYPE (OpenReasoning-Nemotron-7B)")
print("=" * 70)

for horizon_label, obs in [("5d", obs_5d), ("10d", obs_10d), ("20d", obs_20d)]:
    print(f"\n  Horizon: +{horizon_label}")
    print(f"  {'Source':<12} {'IC':>8} {'p-value':>10} {'N obs':>8} {'N articles':>12} {'Avg |score|':>12}")
    print(f"  {'-'*60}")
    
    for source in sorted(obs["source"].unique()):
        subset = obs[obs["source"] == source]
        if len(subset) < 10:
            continue
        corr, pval = stats.pearsonr(subset["predicted_score"], subset["car"])
        n_articles = subset["article_idx"].nunique()
        avg_mag = subset["predicted_score"].abs().mean()
        sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
        print(f"  {source:<12} {corr:>+8.4f} {pval:>10.4f}{sig:>3} {len(subset):>6} {n_articles:>10} {avg_mag:>12.3f}")
    
    # Overall
    corr, pval = stats.pearsonr(obs["predicted_score"], obs["car"])
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"  {'OVERALL':<12} {corr:>+8.4f} {pval:>10.4f}{sig:>3} {len(obs):>6} {obs['article_idx'].nunique():>10}")

# IC by ticker
print("\n" + "=" * 70)
print("IC BY TICKER (at +10d horizon)")
print("=" * 70)
print(f"\n  {'Ticker':<8} {'IC':>8} {'p-value':>10} {'N':>6} {'Avg Pred':>10} {'Avg CAR':>10}")
print(f"  {'-'*55}")
for ticker in QUANTUM_TICKERS:
    subset = obs_10d[obs_10d["ticker"] == ticker]
    if len(subset) < 10:
        continue
    corr, pval = stats.pearsonr(subset["predicted_score"], subset["car"])
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"  {ticker:<8} {corr:>+8.4f} {pval:>10.4f}{sig:>3} {len(subset):>5} {subset['predicted_score'].mean():>+10.3f} {subset['car'].mean():>+10.4f}")

# High conviction vs low conviction
print("\n" + "=" * 70)
print("IC BY CONVICTION LEVEL (at +10d)")
print("=" * 70)
for label, low, high in [("Low (0-0.5)", 0, 0.5), ("Medium (0.5-1.0)", 0.5, 1.0), ("High (1.0+)", 1.0, 999)]:
    subset = obs_10d[(obs_10d["predicted_score"].abs() >= low) & (obs_10d["predicted_score"].abs() < high)]
    if len(subset) < 10:
        print(f"  {label:<20} N={len(subset)} (too few)")
        continue
    corr, pval = stats.pearsonr(subset["predicted_score"], subset["car"])
    sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
    print(f"  {label:<20} IC={corr:>+.4f} p={pval:.4f}{sig:>3} N={len(subset)}")
