"""
6-Way IC Evaluation: Compute Information Coefficient for all model configurations.
"""
import json
import os
import sys
import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, '.')
from agent.config import QUANTUM_TICKERS

# Configs to evaluate
CONFIGS = {
    "1. 8B Base": "data/eval/predictions_8b_base.jsonl",
    "2. 8B + Memory": "data/eval/predictions_iterative_memory.jsonl",
    "3. 14B Base": "data/eval/predictions_14b_base.jsonl",
    "4. 14B + Memory": "data/eval/predictions_14b_memory.jsonl",
    "5. 14B Fine-tuned": "data/eval/predictions_14b_ft_base.jsonl",
    "6. 14B FT + Memory": "data/eval/predictions_14b_ft_memory.jsonl",
}

HORIZONS = [1, 2, 5, 10, 20]
TICKERS_TO_EVAL = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "HON", "IBM"]  # Exclude GOOGL/MSFT/NVDA (always 0)

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

def load_predictions(filepath):
    preds = []
    with open(filepath) as f:
        for line in f:
            if not line.strip():
                continue
            try:
                p = json.loads(line)
                if p.get("status") == "success":
                    preds.append(p)
            except json.JSONDecodeError:
                continue
    return preds

def compute_ic(predictions, market, horizon, min_score=0.3):
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
                try:
                    score = float(score)
                except:
                    continue
            if abs(score) < min_score:
                continue
            ret = get_forward_return(market, ticker, date, horizon)
            if ret is not None:
                scores.append(score)
                returns.append(ret)
    
    if len(scores) >= 10:
        ic, p_value = spearmanr(scores, returns)
        return {"ic": round(float(ic), 4), "p_value": round(float(p_value), 4), "n": len(scores)}
    return {"ic": None, "p_value": None, "n": len(scores)}

def compute_direction_accuracy(predictions, market, horizon=5, min_score=0.3):
    correct = 0
    total = 0
    for pred in predictions:
        date = pred.get("date", "")
        sv = pred.get("signal_vector_clean", {})
        if not sv or not date:
            continue
        for ticker in TICKERS_TO_EVAL:
            score = sv.get(ticker, 0)
            if isinstance(score, str):
                try:
                    score = float(score)
                except:
                    continue
            if abs(score) < min_score:
                continue
            ret = get_forward_return(market, ticker, date, horizon)
            if ret is not None:
                total += 1
                predicted_dir = 1 if score > 0 else -1
                actual_dir = 1 if ret > 0 else -1
                if predicted_dir == actual_dir:
                    correct += 1
    
    if total > 0:
        return {"accuracy": round(correct / total, 4), "correct": correct, "total": total}
    return {"accuracy": None, "correct": 0, "total": 0}

# Main
print("Loading market data...")
market = load_market_data()
print(f"Loaded {len(market)} tickers\n")

results = {}

print("=" * 90)
print("6-WAY MODEL COMPARISON: INFORMATION COEFFICIENT")
print("=" * 90)

# Header
header = f"{'Config':<25}"
for h in HORIZONS:
    header += f"{'IC @' + str(h) + 'd':<12}"
header += f"{'Dir Acc @5d':<12} {'N':<6}"
print(header)
print("-" * 90)

for name, filepath in CONFIGS.items():
    if not os.path.exists(filepath):
        print(f"{name:<25} FILE NOT FOUND")
        continue
    
    preds = load_predictions(filepath)
    
    row = f"{name:<25}"
    ic_values = {}
    for h in HORIZONS:
        ic_result = compute_ic(preds, market, h)
        ic_values[h] = ic_result
        if ic_result["ic"] is not None:
            sig = "*" if ic_result["p_value"] < 0.10 else ""
            sig = "**" if ic_result["p_value"] < 0.05 else sig
            sig = "***" if ic_result["p_value"] < 0.01 else sig
            row += f"{ic_result['ic']:+.4f}{sig:<7}"
        else:
            row += f"{'N/A':<12}"
    
    dir_acc = compute_direction_accuracy(preds, market, horizon=5)
    if dir_acc["accuracy"] is not None:
        row += f"{dir_acc['accuracy']*100:.1f}%{'':<6}"
    else:
        row += f"{'N/A':<12}"
    row += f"{ic_values[5]['n'] if ic_values[5]['ic'] is not None else 0:<6}"
    
    print(row)
    
    results[name] = {
        "predictions": len(preds),
        "ic_by_horizon": ic_values,
        "direction_accuracy_5d": dir_acc,
    }

print("-" * 90)

# Summary analysis
print("\n" + "=" * 90)
print("ANALYSIS: MARGINAL CONTRIBUTION OF EACH VARIABLE")
print("=" * 90)

# Get IC @5d for each config
ics = {}
for name in CONFIGS:
    if name in results and results[name]["ic_by_horizon"][5]["ic"] is not None:
        ics[name] = results[name]["ic_by_horizon"][5]["ic"]

if len(ics) >= 4:
    print(f"\nIC @5d values:")
    for name, ic in sorted(ics.items()):
        print(f"  {name}: {ic:+.4f}")
    
    # Marginal contributions
    base_8b = ics.get("1. 8B Base", 0)
    mem_8b = ics.get("2. 8B + Memory", 0)
    base_14b = ics.get("3. 14B Base", 0)
    mem_14b = ics.get("4. 14B + Memory", 0)
    ft_14b = ics.get("5. 14B Fine-tuned", 0)
    ft_mem_14b = ics.get("6. 14B FT + Memory", 0)
    
    print(f"\n  Marginal value of MEMORY (8B):  {mem_8b - base_8b:+.4f} (8B+mem - 8B base)")
    print(f"  Marginal value of MEMORY (14B): {mem_14b - base_14b:+.4f} (14B+mem - 14B base)")
    print(f"  Marginal value of MEMORY (FT):  {ft_mem_14b - ft_14b:+.4f} (14B FT+mem - 14B FT)")
    print(f"  Marginal value of SIZE:         {base_14b - base_8b:+.4f} (14B base - 8B base)")
    print(f"  Marginal value of FINE-TUNING:  {ft_14b - base_14b:+.4f} (14B FT - 14B base)")
    print(f"  BEST config vs WORST config:    {max(ics.values()) - min(ics.values()):+.4f}")
    print(f"\n  Best overall: {max(ics, key=ics.get)} ({max(ics.values()):+.4f})")
    print(f"  Worst overall: {min(ics, key=ics.get)} ({min(ics.values()):+.4f})")

# Save results
output_file = "data/eval/results_6way_comparison.json"
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"\nResults saved to: {output_file}")
