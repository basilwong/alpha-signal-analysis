"""
Fix the 59 'could not convert string to float' errors by interpreting
string scores as numeric values, then run IC evaluation.
"""
import json
import os
import sys
import re
from scipy.stats import spearmanr
import pandas as pd
import numpy as np

sys.path.insert(0, '.')

PREDICTIONS_FILE = "data/eval/predictions_memory_agent.jsonl"
FIXED_FILE = "data/eval/predictions_memory_agent_fixed.jsonl"
EVAL_ARTICLES = "data/raw/articles_eval.jsonl"
MARKET_DIR = "data/market"
RESULTS_FILE = "data/eval/results_memory_agent.json"

# String-to-score mapping
SCORE_MAP = {
    "strongly bullish": 2.0, "very bullish": 2.0,
    "bullish": 1.5, "moderately bullish": 1.0, "slightly bullish": 0.5,
    "mildly bullish": 0.5, "marginally bullish": 0.3,
    "neutral": 0.0, "none": 0.0, "n/a": 0.0, "negligible": 0.0,
    "slightly bearish": -0.5, "mildly bearish": -0.5, "marginally bearish": -0.3,
    "bearish": -1.5, "moderately bearish": -1.0,
    "strongly bearish": -2.0, "very bearish": -2.0,
    "moderate": 0.8, "significant": 1.5, "minor": 0.3,
    "positive": 1.0, "negative": -1.0,
}

def string_to_score(val):
    """Convert a string score to a numeric value."""
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        val_lower = val.strip().lower()
        # Try direct lookup
        if val_lower in SCORE_MAP:
            return SCORE_MAP[val_lower]
        # Try to extract a number from the string
        numbers = re.findall(r'[-+]?\d*\.?\d+', val)
        if numbers:
            return float(numbers[0])
        # Check for partial matches
        for key, score in SCORE_MAP.items():
            if key in val_lower:
                return score
        return 0.0
    return 0.0

def extract_signal_vector(signal):
    """Extract a clean {ticker: float_score} dict from various signal formats."""
    sv = signal.get('signal_vector', signal)
    result = {}
    if isinstance(sv, dict):
        for ticker, data in sv.items():
            if isinstance(data, (int, float)):
                result[ticker] = float(data)
            elif isinstance(data, dict):
                score = data.get('score', data.get('signal', 0))
                result[ticker] = string_to_score(score)
            elif isinstance(data, str):
                result[ticker] = string_to_score(data)
    return result

def fix_predictions():
    """Fix the error predictions by re-parsing with string-to-score conversion."""
    with open(PREDICTIONS_FILE) as f:
        preds = [json.loads(l) for l in f if l.strip()]
    
    fixed = []
    recovered = 0
    still_failed = 0
    
    for pred in preds:
        if pred.get('status') == 'success':
            # Re-extract signal vector to ensure all scores are numeric
            signal = pred.get('signal', {})
            sv = extract_signal_vector(signal)
            if sv:
                pred['signal_vector_clean'] = sv
            fixed.append(pred)
        elif pred.get('status') == 'error':
            error_msg = pred.get('error', '')
            if 'could not convert string to float' in error_msg:
                # Try to recover from the raw signal data if available
                # The error happened during score extraction, but the signal might be in the pred
                signal = pred.get('signal', {})
                if signal:
                    sv = extract_signal_vector(signal)
                    if sv and any(v != 0 for v in sv.values()):
                        pred['status'] = 'recovered'
                        pred['signal_vector_clean'] = sv
                        recovered += 1
                        fixed.append(pred)
                        continue
                still_failed += 1
                fixed.append(pred)
            else:
                still_failed += 1
                fixed.append(pred)
    
    # Save fixed predictions
    with open(FIXED_FILE, 'w') as f:
        for pred in fixed:
            f.write(json.dumps(pred) + '\n')
    
    usable = sum(1 for p in fixed if p.get('status') in ('success', 'recovered'))
    print(f"Fix results:")
    print(f"  Original successes: {sum(1 for p in preds if p.get('status') == 'success')}")
    print(f"  Recovered from errors: {recovered}")
    print(f"  Still failed: {still_failed}")
    print(f"  Total usable: {usable}")
    return fixed

def load_market_data():
    """Load market price data for IC computation."""
    market = {}
    for ticker in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "GOOGL", "MSFT", "HON", "NVDA", "SPY"]:
        path = os.path.join(MARKET_DIR, f"{ticker}.parquet")
        if os.path.exists(path):
            df = pd.read_parquet(path)
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            if close_col in df.columns:
                series = df[close_col].dropna()
                # Flatten any remaining nested structures
                values = [float(v) for v in series.values]
                market[ticker] = {
                    "dates": [str(d.date()) if hasattr(d, 'date') else str(d)[:10] for d in series.index],
                    "values": values
                }
    return market

def compute_forward_returns(market, ticker, event_date, horizon=5):
    """Compute forward return for a ticker from event_date over horizon days."""
    if ticker not in market:
        return None
    dates = market[ticker]["dates"]
    values = market[ticker]["values"]
    try:
        start_idx = next(i for i, d in enumerate(dates) if d >= event_date)
    except StopIteration:
        return None
    end_idx = min(start_idx + horizon, len(values) - 1)
    if end_idx <= start_idx:
        return None
    return (values[end_idx] - values[start_idx]) / values[start_idx]

def compute_abnormal_return(market, ticker, event_date, horizon=5):
    """Compute abnormal return (ticker return - SPY return)."""
    ticker_ret = compute_forward_returns(market, ticker, event_date, horizon)
    spy_ret = compute_forward_returns(market, "SPY", event_date, horizon)
    if ticker_ret is None or spy_ret is None:
        return None
    return ticker_ret - spy_ret

def run_evaluation(fixed_preds, market):
    """Run IC evaluation on the fixed predictions."""
    usable = [p for p in fixed_preds if p.get('status') in ('success', 'recovered')]
    
    results = {}
    pure_play = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT"]
    
    for horizon in [1, 2, 5, 10, 20]:
        scores = []
        returns = []
        
        for pred in usable:
            date = pred.get('date', '')
            if not date:
                continue
            
            sv = pred.get('signal_vector_clean', {})
            if not sv:
                signal = pred.get('signal', {})
                sv = extract_signal_vector(signal)
            
            for ticker in pure_play:
                score = sv.get(ticker, 0)
                if score == 0:
                    continue
                ar = compute_abnormal_return(market, ticker, date, horizon)
                if ar is not None:
                    scores.append(score)
                    returns.append(ar)
        
        if len(scores) >= 20:
            ic, p_value = spearmanr(scores, returns)
            results[f"ic_{horizon}d"] = {
                "ic": round(ic, 4),
                "p_value": round(p_value, 4),
                "n": len(scores),
                "significant": p_value < 0.05
            }
        else:
            results[f"ic_{horizon}d"] = {"ic": None, "n": len(scores), "note": "insufficient data"}
    
    # Direction accuracy
    for horizon in [1, 5, 10]:
        correct = 0
        total = 0
        for pred in usable:
            date = pred.get('date', '')
            if not date:
                continue
            sv = pred.get('signal_vector_clean', {})
            if not sv:
                signal = pred.get('signal', {})
                sv = extract_signal_vector(signal)
            for ticker in pure_play:
                score = sv.get(ticker, 0)
                if abs(score) < 0.3:  # Only count non-trivial predictions
                    continue
                ar = compute_abnormal_return(market, ticker, date, horizon)
                if ar is not None:
                    total += 1
                    if (score > 0 and ar > 0) or (score < 0 and ar < 0):
                        correct += 1
        
        results[f"direction_accuracy_{horizon}d"] = {
            "accuracy": round(correct / total, 4) if total > 0 else None,
            "n": total
        }
    
    return results

def main():
    print("=" * 60)
    print("MEMORY AGENT EVALUATION")
    print("=" * 60)
    
    # Step 1: Fix predictions
    print("\n--- Step 1: Fix string-to-float errors ---")
    fixed = fix_predictions()
    
    # Step 2: Load market data
    print("\n--- Step 2: Load market data ---")
    market = load_market_data()
    print(f"  Loaded {len(market)} tickers")
    
    # Step 3: Run evaluation
    print("\n--- Step 3: Compute IC and direction accuracy ---")
    results = run_evaluation(fixed, market)
    
    # Step 4: Print results
    print("\n" + "=" * 60)
    print("RESULTS: Memory Agent (qwen-plus + persistent memory)")
    print("=" * 60)
    for key, val in results.items():
        if 'ic' in key and val.get('ic') is not None:
            sig = " ***" if val.get('significant') else ""
            print(f"  {key}: IC = {val['ic']:+.4f} (p={val['p_value']:.4f}, n={val['n']}){sig}")
        elif 'direction' in key and val.get('accuracy') is not None:
            print(f"  {key}: {val['accuracy']*100:.1f}% (n={val['n']})")
    
    # Step 5: Compare with Nemotron results
    print("\n--- Comparison with Fine-tuned Nemotron-7B ---")
    print("  Nemotron-7B (SFT + GRPO): Direction Accuracy @5d = 58.6% (n=261)")
    mem_dir5 = results.get('direction_accuracy_5d', {})
    if mem_dir5.get('accuracy'):
        print(f"  Memory Agent (qwen-plus):  Direction Accuracy @5d = {mem_dir5['accuracy']*100:.1f}% (n={mem_dir5['n']})")
        diff = mem_dir5['accuracy'] * 100 - 58.6
        print(f"  Difference: {diff:+.1f}pp")
    
    # Step 6: Save results
    results["model"] = "Memory Agent (qwen-plus + persistent memory)"
    results["total_predictions"] = len(fixed)
    results["usable_predictions"] = sum(1 for p in fixed if p.get('status') in ('success', 'recovered'))
    
    # Convert numpy types to native Python for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    
    with open(RESULTS_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    print(f"\n  Results saved to: {RESULTS_FILE}")

if __name__ == "__main__":
    main()
