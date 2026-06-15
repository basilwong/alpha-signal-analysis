"""
Deep comparison of OpenReasoning-Nemotron-7B vs Manus Teacher predictions.
Investigates whether the IC difference is driven by genuine reasoning or random variance.
"""
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict
from pathlib import Path

TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA", "QNT"]
ACTIVE_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"]

def load_predictions(path):
    preds = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                p = json.loads(line)
                if p.get("status") == "success" or p.get("success") == True:
                    idx = p.get("article_idx")
                    preds[idx] = p
    return preds

nemotron = load_predictions("data/eval/predictions_openreasoning7b_v4.jsonl")
manus = load_predictions("data/eval/predictions_manus_teacher.jsonl")

print("=" * 70)
print("ANALYSIS 1: Score Distribution Comparison")
print("=" * 70)

def get_scores(preds):
    scores_by_ticker = defaultdict(list)
    for p in preds.values():
        sv = p.get("signal", {}).get("signal_vector", {})
        for ticker in ACTIVE_TICKERS:
            if ticker in sv:
                score = sv[ticker].get("score", 0)
                if isinstance(score, (int, float)):
                    scores_by_ticker[ticker].append(score)
    return scores_by_ticker

nem_scores = get_scores(nemotron)
man_scores = get_scores(manus)

print(f"\n{'Ticker':<8} {'Nemotron Mean':>14} {'Nemotron Std':>13} {'Manus Mean':>12} {'Manus Std':>11} {'Nem NonZero%':>13} {'Man NonZero%':>13}")
print("-" * 90)
for ticker in ACTIVE_TICKERS:
    ns = nem_scores.get(ticker, [])
    ms = man_scores.get(ticker, [])
    if ns and ms:
        nem_nz = sum(1 for s in ns if abs(s) > 0.01) / len(ns) * 100
        man_nz = sum(1 for s in ms if abs(s) > 0.01) / len(ms) * 100
        print(f"  {ticker:<6} {np.mean(ns):>+12.4f} {np.std(ns):>12.4f} {np.mean(ms):>+12.4f} {np.std(ms):>10.4f} {nem_nz:>11.1f}% {man_nz:>11.1f}%")

print("\n" + "=" * 70)
print("ANALYSIS 2: Conviction Level (how often does each model assign 0.0?)")
print("=" * 70)

def count_zeros(preds):
    total = 0
    zeros = 0
    for p in preds.values():
        sv = p.get("signal", {}).get("signal_vector", {})
        for ticker in ACTIVE_TICKERS:
            if ticker in sv:
                score = sv[ticker].get("score", 0)
                total += 1
                if abs(score) < 0.01:
                    zeros += 1
    return zeros, total

nem_z, nem_t = count_zeros(nemotron)
man_z, man_t = count_zeros(manus)
print(f"\n  Nemotron: {nem_z}/{nem_t} zero scores ({nem_z/nem_t*100:.1f}%)")
print(f"  Manus:    {man_z}/{man_t} zero scores ({man_z/man_t*100:.1f}%)")
print(f"\n  Interpretation: {'Nemotron is MORE selective' if nem_z/nem_t > man_z/man_t else 'Manus is MORE selective'}")

print("\n" + "=" * 70)
print("ANALYSIS 3: Agreement Between Models (on overlapping articles)")
print("=" * 70)

common_indices = set(nemotron.keys()) & set(manus.keys())
print(f"\n  Common articles: {len(common_indices)}")

agreements = 0
disagreements = 0
correlations = []

for idx in common_indices:
    nem_sv = nemotron[idx].get("signal", {}).get("signal_vector", {})
    man_sv = manus[idx].get("signal", {}).get("signal_vector", {})
    
    nem_vec = []
    man_vec = []
    for ticker in ACTIVE_TICKERS:
        ns = nem_sv.get(ticker, {}).get("score", 0) if ticker in nem_sv else 0
        ms = man_sv.get(ticker, {}).get("score", 0) if ticker in man_sv else 0
        if isinstance(ns, (int, float)) and isinstance(ms, (int, float)):
            nem_vec.append(ns)
            man_vec.append(ms)
    
    if len(nem_vec) >= 5:
        # Check directional agreement on the strongest signal
        nem_max_idx = np.argmax(np.abs(nem_vec))
        man_max_idx = np.argmax(np.abs(man_vec))
        
        if nem_max_idx == man_max_idx and np.sign(nem_vec[nem_max_idx]) == np.sign(man_vec[man_max_idx]):
            agreements += 1
        else:
            disagreements += 1
        
        # Cross-correlation
        if np.std(nem_vec) > 0 and np.std(man_vec) > 0:
            corr = np.corrcoef(nem_vec, man_vec)[0, 1]
            correlations.append(corr)

print(f"  Strongest-signal agreement: {agreements}/{agreements+disagreements} ({agreements/(agreements+disagreements)*100:.1f}%)")
print(f"  Mean cross-correlation: {np.mean(correlations):.3f}")
print(f"  Median cross-correlation: {np.median(correlations):.3f}")

print("\n" + "=" * 70)
print("ANALYSIS 4: Score Magnitude Distribution")
print("=" * 70)

def magnitude_histogram(preds, name):
    all_scores = []
    for p in preds.values():
        sv = p.get("signal", {}).get("signal_vector", {})
        for ticker in ACTIVE_TICKERS:
            if ticker in sv:
                score = sv[ticker].get("score", 0)
                if isinstance(score, (int, float)):
                    all_scores.append(abs(score))
    
    bins = [0, 0.01, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 999]
    labels = ["0.0", "0.01-0.1", "0.1-0.3", "0.3-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", ">2.0"]
    hist = np.histogram(all_scores, bins=bins)[0]
    print(f"\n  {name}:")
    for label, count in zip(labels, hist):
        pct = count / len(all_scores) * 100
        bar = "#" * int(pct / 2)
        print(f"    {label:>10}: {count:>5} ({pct:>5.1f}%) {bar}")

magnitude_histogram(nemotron, "Nemotron-7B")
magnitude_histogram(manus, "Manus Teacher")

print("\n" + "=" * 70)
print("ANALYSIS 5: Does Nemotron produce more 'extreme' predictions?")
print("=" * 70)

nem_extreme = 0
man_extreme = 0
nem_total = 0
man_total = 0

for p in nemotron.values():
    sv = p.get("signal", {}).get("signal_vector", {})
    for ticker in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT"]:
        if ticker in sv:
            score = sv[ticker].get("score", 0)
            if isinstance(score, (int, float)):
                nem_total += 1
                if abs(score) >= 1.0:
                    nem_extreme += 1

for p in manus.values():
    sv = p.get("signal", {}).get("signal_vector", {})
    for ticker in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT"]:
        if ticker in sv:
            score = sv[ticker].get("score", 0)
            if isinstance(score, (int, float)):
                man_total += 1
                if abs(score) >= 1.0:
                    man_extreme += 1

print(f"\n  Pure-play scores >= 1.0 (high conviction):")
print(f"    Nemotron: {nem_extreme}/{nem_total} ({nem_extreme/nem_total*100:.1f}%)")
print(f"    Manus:    {man_extreme}/{man_total} ({man_extreme/man_total*100:.1f}%)")

print("\n" + "=" * 70)
print("ANALYSIS 6: MSFT/GOOGL/NVDA compliance (should always be 0.0)")
print("=" * 70)

def check_inactive_compliance(preds, name):
    violations = 0
    total = 0
    for p in preds.values():
        sv = p.get("signal", {}).get("signal_vector", {})
        for ticker in ["MSFT", "GOOGL", "NVDA"]:
            if ticker in sv:
                score = sv[ticker].get("score", 0)
                total += 1
                if isinstance(score, (int, float)) and abs(score) > 0.01:
                    violations += 1
    print(f"  {name}: {violations}/{total} violations ({violations/max(total,1)*100:.1f}%)")

check_inactive_compliance(nemotron, "Nemotron-7B")
check_inactive_compliance(manus, "Manus Teacher")

print("\n" + "=" * 70)
print("ANALYSIS 7: Sample predictions on the same article")
print("=" * 70)

# Find 3 articles where both models have predictions and they disagree
sample_count = 0
for idx in sorted(common_indices):
    if sample_count >= 3:
        break
    nem_sv = nemotron[idx].get("signal", {}).get("signal_vector", {})
    man_sv = manus[idx].get("signal", {}).get("signal_vector", {})
    
    # Check if they meaningfully disagree
    nem_ionq = nem_sv.get("IONQ", {}).get("score", 0) if "IONQ" in nem_sv else 0
    man_ionq = man_sv.get("IONQ", {}).get("score", 0) if "IONQ" in man_sv else 0
    
    if abs(nem_ionq) > 0.3 or abs(man_ionq) > 0.3:
        title = nemotron[idx].get("title", "") or manus[idx].get("title", "")
        print(f"\n  Article {idx}: {title[:60]}")
        print(f"  {'Ticker':<8} {'Nemotron':>10} {'Manus':>10}")
        for ticker in ACTIVE_TICKERS:
            ns = nem_sv.get(ticker, {}).get("score", 0) if ticker in nem_sv else 0
            ms = man_sv.get(ticker, {}).get("score", 0) if ticker in man_sv else 0
            if isinstance(ns, (int, float)) and isinstance(ms, (int, float)):
                flag = " <--" if abs(ns - ms) > 0.5 else ""
                print(f"    {ticker:<6} {ns:>+10.2f} {ms:>+10.2f}{flag}")
        sample_count += 1
