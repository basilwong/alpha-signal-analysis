"""
Compute both Pearson and Spearman IC for all models to provide clarity in reporting.
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

MODELS = {
    "V7d GRPO (clean)": EVAL_DIR / "predictions_v7d_grpo_clean.jsonl",
    "V7b Rejection (clean)": EVAL_DIR / "predictions_v7b_clean.jsonl",
    "V7c DPO (clean)": EVAL_DIR / "predictions_v7c_clean.jsonl",
    "V4 Baseline": EVAL_DIR / "predictions_openreasoning7b_v4.jsonl",
}


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
    predictions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p.get("status") == "success":
                    predictions.append(p)
    return predictions


def compute_observations(predictions, returns_df, horizon):
    results = []
    for pred in predictions:
        date_str = pred.get("date", "")
        signal = pred.get("signal", {})
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
            idx = ws.index.intersection(wm.index)
            if len(idx) < 1:
                continue
            expected = alpha + beta * wm.loc[idx]
            ar = ws.loc[idx] - expected
            car = ar.sum()

            results.append({"score": score, "car": car})

    return pd.DataFrame(results)


returns_df = load_returns()

print("=" * 90)
print("IC COMPARISON: Pearson vs Spearman (all models, all horizons)")
print("=" * 90)

for horizon in [5, 10, 20]:
    print(f"\n  Horizon: +{horizon}d")
    print(f"  {'Model':<25} {'Pearson IC':>12} {'p-val':>8} {'Spearman IC':>13} {'p-val':>8} {'N':>6}")
    print(f"  {'-'*75}")

    for name, path in MODELS.items():
        if not path.exists():
            continue
        preds = load_predictions(path)
        obs = compute_observations(preds, returns_df, horizon)

        if len(obs) < 10:
            print(f"  {name:<25} {'N/A':>12} {'':>8} {'N/A':>13} {'':>8} {len(obs):>6}")
            continue

        pearson_ic, pearson_p = stats.pearsonr(obs["score"], obs["car"])
        spearman_ic, spearman_p = stats.spearmanr(obs["score"], obs["car"])

        p_sig = "***" if pearson_p < 0.01 else "**" if pearson_p < 0.05 else "*" if pearson_p < 0.1 else ""
        s_sig = "***" if spearman_p < 0.01 else "**" if spearman_p < 0.05 else "*" if spearman_p < 0.1 else ""

        print(f"  {name:<25} {pearson_ic:>+.4f}{p_sig:<3} {pearson_p:>7.4f}  {spearman_ic:>+.4f}{s_sig:<3} {spearman_p:>7.4f}  {len(obs):>5}")
