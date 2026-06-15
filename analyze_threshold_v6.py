"""
Threshold sweep on V6 predictions to see if filtering improves IC.
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
                if p.get("status") == "success":
                    preds.append(p)
    return preds


def compute_observations(predictions, returns_df, horizon):
    results = []
    for pred in predictions:
        date_str = pred.get("date", "")
        signal = pred.get("signal", {})
        signal_vector = signal.get("signal_vector", {})
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
            if not isinstance(predicted_score, (int, float)) or predicted_score == 0:
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
                "predicted_score": predicted_score,
                "abs_score": abs(predicted_score),
                "car": car,
            })

    return pd.DataFrame(results)


# Load
print("Loading data...")
returns_df = load_returns()
v6_preds = load_predictions(EVAL_DIR / "predictions_openreasoning7b_v6.jsonl")

print("Computing observations at multiple horizons...")
horizons = {"1d": 1, "5d": 5, "10d": 10, "20d": 20}
obs_by_horizon = {}
for label, h in horizons.items():
    obs_by_horizon[label] = compute_observations(v6_preds, returns_df, h)
    print(f"  {label}: {len(obs_by_horizon[label])} observations")

# Sweep thresholds
thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

print(f"\n{'='*90}")
print(f"THRESHOLD SWEEP: V6 OpenReasoning-Nemotron-7B")
print(f"{'='*90}")
print(f"\n  {'Thresh':<8} {'IC@1d':>8} {'IC@5d':>8} {'IC@10d':>8} {'IC@20d':>8} {'N@5d':>6} {'DirAcc@5d':>10}")
print(f"  {'-'*60}")

for thresh in thresholds:
    row = f"  >={thresh:<5.1f}"
    for label in ["1d", "5d", "10d", "20d"]:
        obs = obs_by_horizon[label]
        subset = obs[obs["abs_score"] >= thresh]
        if len(subset) < 20:
            row += f" {'N/A':>8}"
        else:
            ic, pval = stats.pearsonr(subset["predicted_score"], subset["car"])
            sig = "***" if pval < 0.01 else "**" if pval < 0.05 else "*" if pval < 0.1 else ""
            row += f" {ic:>+6.4f}{sig:<2}"

    # N and DirAcc at 5d
    obs5 = obs_by_horizon["5d"]
    subset5 = obs5[obs5["abs_score"] >= thresh]
    if len(subset5) >= 20:
        correct = ((subset5["predicted_score"] > 0) & (subset5["car"] > 0)) | \
                  ((subset5["predicted_score"] < 0) & (subset5["car"] < 0))
        row += f" {len(subset5):>6} {correct.mean():>9.1%}"
    else:
        row += f" {'N/A':>6} {'N/A':>9}"

    print(row)

# Also show bullish-only and bearish-only at best threshold
print(f"\n{'='*90}")
print(f"DIRECTION SPLIT at |score| >= 0.5 (5d horizon)")
print(f"{'='*90}")

obs5 = obs_by_horizon["5d"]
subset = obs5[obs5["abs_score"] >= 0.5]
bullish = subset[subset["predicted_score"] > 0]
bearish = subset[subset["predicted_score"] < 0]

if len(bullish) >= 10:
    ic_b, p_b = stats.pearsonr(bullish["predicted_score"], bullish["car"])
    print(f"  Bullish (N={len(bullish)}): IC={ic_b:+.4f} p={p_b:.4f} AvgCAR={bullish['car'].mean():+.4f}")

if len(bearish) >= 10:
    ic_bear, p_bear = stats.pearsonr(bearish["predicted_score"], bearish["car"])
    print(f"  Bearish (N={len(bearish)}): IC={ic_bear:+.4f} p={p_bear:.4f} AvgCAR={bearish['car'].mean():+.4f}")
