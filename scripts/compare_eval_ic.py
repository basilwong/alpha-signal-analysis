"""
Compare IC between v1 (old prompt) and v2 (new prompt) eval predictions.
Measures the improvement from prompt changes alone (no fine-tuning).

Usage:
    python scripts/compare_eval_ic.py
"""

import json
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

PROJECT_ROOT = Path(__file__).parent.parent
DATA_EVAL = PROJECT_ROOT / "data" / "eval"
MARKET_DIR = PROJECT_ROOT / "data" / "market"

ACTIVE_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "HON"]
ALL_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"]


def load_predictions(filepath):
    """Load predictions from JSONL."""
    preds = []
    with open(filepath) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get("status") == "success" or r.get("success"):
                    preds.append(r)
    return preds


def load_returns():
    """Load daily returns for all tickers."""
    returns = {}
    for ticker in ALL_TICKERS + ["SPY"]:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close = df["Close"]
            returns[ticker] = close.pct_change().dropna()
    return pd.DataFrame(returns)


def compute_ic(predictions, returns_df, horizon=5, tickers=None):
    """
    Compute Information Coefficient (Spearman rank correlation) between
    predicted scores and forward returns at the given horizon.
    """
    if tickers is None:
        tickers = ACTIVE_TICKERS
    
    all_predicted = []
    all_actual = []
    
    for pred in predictions:
        date = pred.get("date")
        signal = pred.get("signal")
        if not date or not signal:
            continue
        
        sv = signal.get("signal_vector", {})
        target_date = pd.Timestamp(date)
        
        for ticker in tickers:
            if ticker not in sv or ticker not in returns_df.columns:
                continue
            
            score = sv[ticker].get("score", 0)
            
            # Compute forward return
            ticker_returns = returns_df[ticker]
            future = ticker_returns.loc[target_date:]
            
            if len(future) < horizon + 1:
                continue
            
            # Cumulative return over horizon days
            forward_ret = (1 + future.iloc[1:horizon+1]).prod() - 1
            
            all_predicted.append(score)
            all_actual.append(forward_ret)
    
    if len(all_predicted) < 30:
        return None, None, len(all_predicted)
    
    ic, p_value = stats.spearmanr(all_predicted, all_actual)
    return ic, p_value, len(all_predicted)


def compute_ic_by_ticker(predictions, returns_df, horizon=5):
    """Compute IC for each ticker separately."""
    results = {}
    for ticker in ACTIVE_TICKERS:
        ic, p, n = compute_ic(predictions, returns_df, horizon, tickers=[ticker])
        if ic is not None:
            results[ticker] = {"ic": ic, "p_value": p, "n": n}
    return results


def compute_direction_accuracy(predictions, returns_df, horizon=5, tickers=None):
    """Compute direction accuracy (% of non-zero predictions with correct sign)."""
    if tickers is None:
        tickers = ACTIVE_TICKERS
    
    correct = 0
    total = 0
    
    for pred in predictions:
        date = pred.get("date")
        signal = pred.get("signal")
        if not date or not signal:
            continue
        
        sv = signal.get("signal_vector", {})
        target_date = pd.Timestamp(date)
        
        for ticker in tickers:
            if ticker not in sv or ticker not in returns_df.columns:
                continue
            
            score = sv[ticker].get("score", 0)
            if abs(score) < 0.01:  # Skip zero predictions
                continue
            
            ticker_returns = returns_df[ticker]
            future = ticker_returns.loc[target_date:]
            
            if len(future) < horizon + 1:
                continue
            
            forward_ret = (1 + future.iloc[1:horizon+1]).prod() - 1
            if abs(forward_ret) < 0.005:  # Skip trivial moves
                continue
            
            total += 1
            if (score > 0 and forward_ret > 0) or (score < 0 and forward_ret < 0):
                correct += 1
    
    if total == 0:
        return None, 0
    return correct / total, total


def compute_ic_by_source(predictions, returns_df, horizon=5):
    """Compute IC split by source type."""
    by_source = {}
    for pred in predictions:
        source = pred.get("source", "unknown")
        if source not in by_source:
            by_source[source] = []
        by_source[source].append(pred)
    
    results = {}
    for source, preds in by_source.items():
        ic, p, n = compute_ic(preds, returns_df, horizon)
        if ic is not None:
            results[source] = {"ic": ic, "p_value": p, "n": n}
    return results


def main():
    print("=" * 70)
    print("EVAL IC COMPARISON: v1 (old prompt) vs v2 (new prompt)")
    print("=" * 70)
    
    # Load predictions
    v1_file = DATA_EVAL / "predictions_manus_teacher.jsonl"
    v2_file = DATA_EVAL / "predictions_manus_teacher_v2.jsonl"
    
    v1_preds = load_predictions(v1_file)
    v2_preds = load_predictions(v2_file)
    
    print(f"\nv1 predictions: {len(v1_preds)} successful")
    print(f"v2 predictions: {len(v2_preds)} successful")
    
    # Load returns
    returns_df = load_returns()
    print(f"Market data: {len(returns_df)} days, {list(returns_df.columns)}")
    
    # ============================================================
    print(f"\n{'='*70}")
    print("OVERALL IC (5-day horizon, active tickers only)")
    print(f"{'='*70}")
    
    v1_ic, v1_p, v1_n = compute_ic(v1_preds, returns_df, horizon=5)
    v2_ic, v2_p, v2_n = compute_ic(v2_preds, returns_df, horizon=5)
    
    print(f"\n  {'Metric':<20} {'v1 (old prompt)':<20} {'v2 (new prompt)':<20} {'Change':<15}")
    print(f"  {'-'*75}")
    print(f"  {'IC (5d)':<20} {v1_ic:+.4f}{'':<12} {v2_ic:+.4f}{'':<12} {v2_ic-v1_ic:+.4f}")
    print(f"  {'p-value':<20} {v1_p:.4f}{'':<13} {v2_p:.4f}{'':<13}")
    print(f"  {'n observations':<20} {v1_n:<20} {v2_n:<20}")
    
    # Direction accuracy
    v1_acc, v1_acc_n = compute_direction_accuracy(v1_preds, returns_df)
    v2_acc, v2_acc_n = compute_direction_accuracy(v2_preds, returns_df)
    
    if v1_acc and v2_acc:
        print(f"  {'Direction acc':<20} {v1_acc:.1%} (n={v1_acc_n}){'':<5} {v2_acc:.1%} (n={v2_acc_n}){'':<5} {v2_acc-v1_acc:+.1%}")
    
    # ============================================================
    print(f"\n{'='*70}")
    print("IC BY TICKER (5-day horizon)")
    print(f"{'='*70}")
    
    v1_by_ticker = compute_ic_by_ticker(v1_preds, returns_df)
    v2_by_ticker = compute_ic_by_ticker(v2_preds, returns_df)
    
    print(f"\n  {'Ticker':<8} {'v1 IC':<12} {'v2 IC':<12} {'Change':<12} {'v2 p-value':<12} {'Signal'}")
    print(f"  {'-'*65}")
    
    for ticker in ACTIVE_TICKERS:
        v1_t = v1_by_ticker.get(ticker, {})
        v2_t = v2_by_ticker.get(ticker, {})
        v1_ic_t = v1_t.get("ic", 0)
        v2_ic_t = v2_t.get("ic", 0)
        v2_p_t = v2_t.get("p_value", 1)
        change = v2_ic_t - v1_ic_t
        sig = "***" if v2_p_t < 0.01 else "**" if v2_p_t < 0.05 else "*" if v2_p_t < 0.1 else ""
        improved = "↑" if change > 0.02 else "↓" if change < -0.02 else "→"
        print(f"  {ticker:<8} {v1_ic_t:+.4f}{'':<5} {v2_ic_t:+.4f}{'':<5} {change:+.4f} {improved:<3} {v2_p_t:.4f}{'':<5} {sig}")
    
    # Also show inactive tickers for v1 (to confirm they were noise)
    print(f"\n  Inactive tickers (v1 only, for reference):")
    for ticker in ["MSFT", "GOOGL", "NVDA"]:
        v1_t = v1_by_ticker.get(ticker, {})
        if not v1_t:
            # Compute manually for v1
            ic, p, n = compute_ic(v1_preds, returns_df, horizon=5, tickers=[ticker])
            if ic is not None:
                print(f"  {ticker:<8} {ic:+.4f} (p={p:.4f}) — now inactive in v2")
    
    # ============================================================
    print(f"\n{'='*70}")
    print("IC BY SOURCE")
    print(f"{'='*70}")
    
    v1_by_source = compute_ic_by_source(v1_preds, returns_df)
    v2_by_source = compute_ic_by_source(v2_preds, returns_df)
    
    print(f"\n  {'Source':<12} {'v1 IC':<12} {'v2 IC':<12} {'Change':<12}")
    print(f"  {'-'*48}")
    for source in sorted(set(list(v1_by_source.keys()) + list(v2_by_source.keys()))):
        v1_s = v1_by_source.get(source, {}).get("ic", 0)
        v2_s = v2_by_source.get(source, {}).get("ic", 0)
        print(f"  {source:<12} {v1_s:+.4f}{'':<5} {v2_s:+.4f}{'':<5} {v2_s-v1_s:+.4f}")
    
    # ============================================================
    print(f"\n{'='*70}")
    print("IC DECAY CURVE (v2)")
    print(f"{'='*70}")
    
    print(f"\n  {'Horizon':<10} {'IC':<10} {'p-value':<10} {'n':<10}")
    print(f"  {'-'*40}")
    for h in [1, 2, 5, 10, 20]:
        ic, p, n = compute_ic(v2_preds, returns_df, horizon=h)
        if ic is not None:
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""
            print(f"  {h:<10} {ic:+.4f}{'':<4} {p:.4f}{'':<4} {n:<10} {sig}")
    
    # ============================================================
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    
    if v2_ic > v1_ic:
        improvement = (v2_ic - v1_ic) / abs(v1_ic) * 100 if v1_ic != 0 else float('inf')
        print(f"\n  ✓ IC IMPROVED: {v1_ic:+.4f} → {v2_ic:+.4f} ({improvement:+.0f}% relative improvement)")
    else:
        print(f"\n  ✗ IC decreased: {v1_ic:+.4f} → {v2_ic:+.4f}")
    
    if v2_acc and v1_acc and v2_acc > v1_acc:
        print(f"  ✓ Direction accuracy improved: {v1_acc:.1%} → {v2_acc:.1%}")
    elif v2_acc and v1_acc:
        print(f"  → Direction accuracy: {v1_acc:.1%} → {v2_acc:.1%}")


if __name__ == "__main__":
    main()
