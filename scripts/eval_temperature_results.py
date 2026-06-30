"""Evaluate IC across all temperature experiments."""
import json
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, '.')
from agent.config import QUANTUM_TICKERS

TICKERS_TO_EVAL = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "HON", "IBM"]

def load_market_data():
    market = {}
    for ticker in QUANTUM_TICKERS + ["SPY"]:
        path = f"data/market/{ticker}.parquet"
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if close_col in df.columns:
                series = df[close_col].dropna()
                market[ticker] = {
                    "dates": [str(d.date()) for d in series.index],
                    "values": [float(v) for v in series.values]
                }
    return market

def get_forward_return(market, ticker, event_date, horizon):
    if ticker not in market:
        return None
    dates = market[ticker]["dates"]
    values = market[ticker]["values"]
    try:
        start_idx = next(i for i, d in enumerate(dates) if d >= event_date)
    except StopIteration:
        return None
    end_idx = min(start_idx + horizon, len(values) - 1)
    if end_idx <= start_idx or values[start_idx] == 0:
        return None
    return (values[end_idx] - values[start_idx]) / values[start_idx]

def compute_ic(predictions, market, horizon=5, min_score=0.3):
    scores = []
    returns = []
    for pred in predictions:
        date = pred.get("date", "")
        sv = pred.get("signal_vector_clean", {})
        if not sv or not date:
            continue
        for ticker in TICKERS_TO_EVAL:
            score = sv.get(ticker, 0)
            if isinstance(score, str):
                try: score = float(score)
                except: continue
            if abs(score) < min_score:
                continue
            ret = get_forward_return(market, ticker, date, horizon)
            if ret is not None:
                scores.append(score)
                returns.append(ret)
    if len(scores) >= 10:
        ic, p = spearmanr(scores, returns)
        return {"ic": round(float(ic), 4), "p": round(float(p), 4), "n": len(scores)}
    return {"ic": None, "p": None, "n": len(scores)}

def compute_score_stats(predictions):
    all_scores = []
    for pred in predictions:
        sv = pred.get("signal_vector_clean", {})
        for ticker in TICKERS_TO_EVAL:
            score = sv.get(ticker, 0)
            if isinstance(score, str):
                try: score = float(score)
                except: continue
            all_scores.append(abs(score))
    if all_scores:
        return {
            "mean_abs": round(np.mean(all_scores), 3),
            "median_abs": round(np.median(all_scores), 3),
            "pct_above_03": round(sum(1 for s in all_scores if s > 0.3) / len(all_scores) * 100, 1),
            "pct_above_10": round(sum(1 for s in all_scores if s > 1.0) / len(all_scores) * 100, 1),
            "total_scores": len(all_scores),
        }
    return {}

def load_preds(filepath):
    preds = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                if p.get("status") == "success":
                    preds.append(p)
            except:
                continue
    return preds

# Main
market = load_market_data()

# Temperature experiment files
files = {
    "14B Base t=0.3": "data/eval/predictions_14b_base.jsonl",
    "14B Base t=0.5": "data/eval/temp_exp/14b_base_t0.5.jsonl",
    "14B Base t=0.7": "data/eval/temp_exp/14b_base_t0.7.jsonl",
    "14B Base t=1.0": "data/eval/temp_exp/14b_base_t1.0.jsonl",
    "14B FT t=0.3": "data/eval/predictions_14b_ft_base.jsonl",
    "14B FT t=0.5": "data/eval/temp_exp/14b_ft_t0.5.jsonl",
    "14B FT t=0.7": "data/eval/temp_exp/14b_ft_t0.7.jsonl",
    "14B FT t=1.0": "data/eval/temp_exp/14b_ft_t1.0.jsonl",
    "8B Base t=0.3": "data/eval/predictions_8b_base.jsonl",
}

print("=" * 100)
print("TEMPERATURE EXPERIMENT RESULTS")
print("=" * 100)
print(f"\n{'Config':<20} {'IC @5d':<12} {'p-value':<10} {'N':<6} {'Mean|Score|':<12} {'%>0.3':<8} {'%>1.0':<8} {'Preds':<6}")
print("-" * 100)

for name, filepath in files.items():
    if not os.path.exists(filepath):
        print(f"{name:<20} FILE NOT FOUND")
        continue
    preds = load_preds(filepath)
    ic = compute_ic(preds, market, horizon=5)
    stats = compute_score_stats(preds)
    
    ic_str = f"{ic['ic']:+.4f}" if ic['ic'] is not None else "N/A"
    p_str = f"{ic['p']:.4f}" if ic['p'] is not None else "N/A"
    sig = ""
    if ic['p'] is not None:
        if ic['p'] < 0.01: sig = "***"
        elif ic['p'] < 0.05: sig = "**"
        elif ic['p'] < 0.10: sig = "*"
    
    print(f"{name:<20} {ic_str}{sig:<7} {p_str:<10} {ic.get('n',0):<6} {stats.get('mean_abs','?'):<12} {stats.get('pct_above_03','?'):<8} {stats.get('pct_above_10','?'):<8} {len(preds):<6}")

print("-" * 100)
print("\nKey: *** p<0.01, ** p<0.05, * p<0.10")
print("\nIf model collapse is the issue, higher temperatures should show:")
print("  - Higher mean |score| (more diverse outputs)")
print("  - Higher % above thresholds")
print("  - Potentially better IC (if collapse was causing degenerate predictions)")
