"""
Repair base model predictions by applying aggressive JSON extraction.

The base reasoning models output <think>...</think> blocks followed by JSON,
but often the JSON is malformed (unescaped quotes, trailing commas, truncated).

This script:
1. Re-downloads raw outputs if available (they're not, we only have parsed results)
2. For "No JSON found" errors: the model likely produced only thinking with no JSON
3. For "JSON parse error": try aggressive repair (trailing commas, newlines, partial JSON)

Since we don't have the raw outputs saved, we'll need to re-run the base models
with a more robust parser. But first, let's see what we CAN extract from the 14B
successes and check if there's signal.
"""
import json
import re
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


def compute_ic(predictions, returns_df, horizon=5):
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
            car = (ws.loc[idx] - expected).sum()
            results.append({"score": score, "car": car})

    if len(results) < 10:
        return None, None, len(results)
    df = pd.DataFrame(results)
    sp_ic, sp_p = stats.spearmanr(df["score"], df["car"])
    pe_ic, pe_p = stats.pearsonr(df["score"], df["car"])
    return sp_ic, sp_p, pe_ic, pe_p, len(results)


# Evaluate the 45 successful 14B predictions
print("=" * 70)
print("EVALUATING BASE MODEL PREDICTIONS (what we have)")
print("=" * 70)

returns_df = load_returns()

for model_file, model_name in [
    ("predictions_base_14b.jsonl", "14B Base (45 successes)"),
    ("predictions_openreasoning7b_v4.jsonl", "7B V4 Fine-tuned (388 successes)"),
    ("predictions_v7d_grpo_clean.jsonl", "7B V7d GRPO (342 successes)"),
]:
    path = EVAL_DIR / model_file
    if not path.exists():
        print(f"\n  {model_name}: FILE NOT FOUND")
        continue

    preds = [json.loads(l) for l in open(path) if json.loads(l).get("status") == "success"]
    print(f"\n  {model_name}: {len(preds)} predictions")

    for horizon in [5, 10, 20]:
        result = compute_ic(preds, returns_df, horizon)
        if result is None:
            print(f"    +{horizon}d: N/A (too few observations)")
        else:
            sp_ic, sp_p, pe_ic, pe_p, n = result
            sp_sig = "***" if sp_p < 0.01 else "**" if sp_p < 0.05 else "*" if sp_p < 0.1 else ""
            print(f"    +{horizon}d: Spearman={sp_ic:+.4f}{sp_sig} (p={sp_p:.4f}) N={n}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)
print("""
The 14B base model only produced 45 valid predictions (11% success rate).
With only 45 predictions, we likely have <50 observations for IC computation,
which is too few for statistical significance.

To properly compare base vs fine-tuned, we need to either:
1. Re-run base models with max_tokens=6000 (so thinking + JSON both fit)
2. Use a two-pass approach: first generate thinking, then prompt for JSON only
3. Accept that base models can't produce structured output without fine-tuning

Option 1 is the simplest fix. The issue is that max_tokens=3000 isn't enough
for the base model's thinking chain + full JSON output.
""")
