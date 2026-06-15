"""
Find the optimal "no trade" threshold by sweeping cutoffs and computing IC.
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
                if p.get("status") == "success" or p.get("success") == True:
                    preds.append(p)
    return preds


def compute_observations_all(predictions, returns_df, horizon=10):
    """Include ALL predictions including zeros for threshold analysis."""
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
            if not isinstance(predicted_score, (int, float)):
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
                "ticker": ticker,
                "predicted_score": predicted_score,
                "abs_score": abs(predicted_score),
                "car": car,
            })

    return pd.DataFrame(results)


# Load
returns_df = load_returns()
nemotron_preds = load_predictions(EVAL_DIR / "predictions_openreasoning7b_v4.jsonl")

print("Computing observations...")
obs_10d = compute_observations_all(nemotron_preds, returns_df, horizon=10)
obs_5d = compute_observations_all(nemotron_preds, returns_df, horizon=5)

# Filter out zeros (model already filtered these)
obs_10d_nonzero = obs_10d[obs_10d["predicted_score"] != 0]
obs_5d_nonzero = obs_5d[obs_5d["predicted_score"] != 0]

print(f"Non-zero observations: {len(obs_10d_nonzero)} at 10d, {len(obs_5d_nonzero)} at 5d")

# Sweep thresholds
print("\n" + "=" * 70)
print("THRESHOLD SWEEP: IC at different minimum |score| cutoffs")
print("=" * 70)

thresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]

print(f"\n  {'Threshold':<12} {'IC@5d':>8} {'p@5d':>8} {'N@5d':>6} {'IC@10d':>8} {'p@10d':>8} {'N@10d':>6} {'DirAcc@10d':>11}")
print(f"  {'-'*75}")

for thresh in thresholds:
    # Filter: keep only predictions where |score| >= threshold
    subset_5d = obs_5d_nonzero[obs_5d_nonzero["abs_score"] >= thresh]
    subset_10d = obs_10d_nonzero[obs_10d_nonzero["abs_score"] >= thresh]
    
    if len(subset_5d) < 20 or len(subset_10d) < 20:
        print(f"  {thresh:<12} {'(too few observations)':>50}")
        continue
    
    ic5, p5 = stats.pearsonr(subset_5d["predicted_score"], subset_5d["car"])
    ic10, p10 = stats.pearsonr(subset_10d["predicted_score"], subset_10d["car"])
    
    # Direction accuracy at 10d
    correct = ((subset_10d["predicted_score"] > 0) & (subset_10d["car"] > 0)) | \
              ((subset_10d["predicted_score"] < 0) & (subset_10d["car"] < 0))
    dir_acc = correct.mean()
    
    sig5 = "***" if p5 < 0.01 else "**" if p5 < 0.05 else "*" if p5 < 0.1 else ""
    sig10 = "***" if p10 < 0.01 else "**" if p10 < 0.05 else "*" if p10 < 0.1 else ""
    
    print(f"  |score|>={thresh:<4.1f} {ic5:>+7.4f}{sig5:<3} {p5:>7.4f} {len(subset_5d):>5} {ic10:>+7.4f}{sig10:<3} {p10:>7.4f} {len(subset_10d):>5} {dir_acc:>9.1%}")

print("\n" + "=" * 70)
print("INTERPRETATION")
print("=" * 70)
print("""
  The optimal threshold maximizes IC while retaining enough observations
  for statistical significance. Look for:
  - Highest IC with p < 0.05
  - At least 100+ observations (for robustness)
  - Improving direction accuracy
""")
