"""
Spot-check data quality across all training categories.
Examines formatting, content quality, score distributions, and reasoning quality.

Usage:
    python scripts/spot_check_quality.py
"""

import json
import sys
import random
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATA_TRAINING = PROJECT_ROOT / "data" / "training"

random.seed(42)


def load_all_data():
    """Load all training data files."""
    data = {}
    files = {
        "combined": "manus_teacher_combined.jsonl",
        "arxiv": "manus_arxiv_rebalance.jsonl",
        "qnt": "manus_qnt_examples.jsonl",
    }
    for key, fname in files.items():
        fpath = DATA_TRAINING / fname
        if fpath.exists():
            with open(fpath) as f:
                data[key] = [json.loads(l) for l in f if l.strip()]
        else:
            data[key] = []
    return data


def check_json_structure(record, label=""):
    """Check a single record for structural issues."""
    issues = []
    
    if not record.get("success"):
        return [f"{label}: not successful"]
    
    signal = record.get("signal")
    if not signal:
        return [f"{label}: no signal field"]
    
    # Check signal_vector
    sv = signal.get("signal_vector", {})
    expected = {"IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"}
    actual = set(sv.keys())
    if actual != expected:
        issues.append(f"{label}: tickers mismatch. Missing: {expected - actual}, Extra: {actual - expected}")
    
    # Check each ticker has score + reasoning
    for ticker, data in sv.items():
        if not isinstance(data, dict):
            issues.append(f"{label}: {ticker} is not a dict: {type(data)}")
            continue
        if "score" not in data:
            issues.append(f"{label}: {ticker} missing 'score'")
        if "reasoning" not in data:
            issues.append(f"{label}: {ticker} missing 'reasoning'")
        elif not data["reasoning"] or len(data["reasoning"]) < 10:
            issues.append(f"{label}: {ticker} reasoning too short: '{data.get('reasoning', '')[:30]}'")
    
    # Check required top-level fields
    required = ["event_type", "time_horizon", "information_novelty", 
                "technical_translation", "signal_rationale", "chain_of_thought"]
    for field in required:
        if field not in signal:
            issues.append(f"{label}: missing field '{field}'")
        elif not signal[field] or len(str(signal[field])) < 10:
            issues.append(f"{label}: field '{field}' too short: '{str(signal[field])[:30]}'")
    
    # Check no signal_decay
    if "signal_decay" in signal:
        issues.append(f"{label}: signal_decay still present")
    
    # Check time_horizon is valid enum
    valid_horizons = {"intraday", "2-5 days", "1-2 weeks", "1+ month"}
    if signal.get("time_horizon") and signal["time_horizon"] not in valid_horizons:
        issues.append(f"{label}: invalid time_horizon: '{signal['time_horizon']}'")
    
    # Check information_novelty is valid enum
    valid_novelty = {"high", "medium", "low"}
    if signal.get("information_novelty") and signal["information_novelty"] not in valid_novelty:
        issues.append(f"{label}: invalid information_novelty: '{signal['information_novelty']}'")
    
    return issues


def check_score_distributions(records, label=""):
    """Analyze score distributions for anomalies."""
    print(f"\n--- Score Distribution: {label} ({len(records)} records) ---")
    
    successful = [r for r in records if r.get("success") and r.get("signal")]
    if not successful:
        print("  No successful records")
        return
    
    # Per-ticker stats
    ticker_scores = {}
    for r in successful:
        sv = r["signal"].get("signal_vector", {})
        for ticker, data in sv.items():
            if ticker not in ticker_scores:
                ticker_scores[ticker] = []
            ticker_scores[ticker].append(data.get("score", 0))
    
    print(f"  {'Ticker':<8} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Zero%':>8} {'NonZero':>8}")
    for ticker in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"]:
        scores = ticker_scores.get(ticker, [])
        if not scores:
            continue
        mean = sum(scores) / len(scores)
        std = (sum((s - mean)**2 for s in scores) / len(scores)) ** 0.5
        zero_pct = sum(1 for s in scores if abs(s) < 0.001) / len(scores) * 100
        nonzero = sum(1 for s in scores if abs(s) >= 0.001)
        print(f"  {ticker:<8} {mean:>+8.3f} {std:>8.3f} {min(scores):>+8.2f} {max(scores):>+8.2f} {zero_pct:>7.0f}% {nonzero:>8}")


def spot_check_examples(records, label="", n=5):
    """Print n random examples for manual inspection."""
    print(f"\n--- Spot Check: {label} (random {n} examples) ---")
    
    successful = [r for r in records if r.get("success") and r.get("signal")]
    if not successful:
        print("  No successful records")
        return
    
    samples = random.sample(successful, min(n, len(successful)))
    
    for i, r in enumerate(samples):
        signal = r["signal"]
        sv = signal.get("signal_vector", {})
        
        # Get the highest-magnitude active ticker
        active_scores = [(t, sv[t]["score"]) for t in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"] if t in sv]
        active_scores.sort(key=lambda x: abs(x[1]), reverse=True)
        
        print(f"\n  [{i+1}] idx={r.get('article_idx', '?')} | category={r.get('category', '?')}")
        
        # Show context
        if r.get("title"):
            print(f"      Title: {r['title'][:70]}")
        elif r.get("scenario"):
            print(f"      Scenario: {r['scenario'][:70]}")
        
        # Show top scores
        print(f"      Top scores: {', '.join(f'{t}={s:+.2f}' for t, s in active_scores[:4])}")
        
        # Show event_type and time_horizon
        print(f"      Event: {signal.get('event_type', '?')} | Horizon: {signal.get('time_horizon', '?')} | Novelty: {signal.get('information_novelty', '?')}")
        
        # Show chain_of_thought quality
        cot = signal.get("chain_of_thought", "")
        print(f"      CoT ({len(cot)} chars): {cot[:120]}...")
        
        # Show technical_translation
        tt = signal.get("technical_translation", "")
        print(f"      Tech translation ({len(tt)} chars): {tt[:100]}...")
        
        # Check inactive tickers
        for t in ["MSFT", "GOOGL", "NVDA"]:
            if t in sv and sv[t]["score"] != 0.0:
                print(f"      ⚠️  {t} score = {sv[t]['score']} (should be 0.0!)")


def check_arxiv_quality(records):
    """Specific checks for arXiv examples."""
    print(f"\n--- ArXiv Quality Check ({len(records)} records) ---")
    
    successful = [r for r in records if r.get("success") and r.get("signal")]
    
    # Check by tier
    tiers = Counter(r.get("arxiv_tier", "unknown") for r in records)
    print(f"  Tiers: {dict(tiers)}")
    
    # Check score magnitudes by tier
    for tier in ["important", "incremental", "unrelated"]:
        tier_records = [r for r in successful if r.get("arxiv_tier") == tier]
        if not tier_records:
            continue
        
        max_scores = []
        for r in tier_records:
            sv = r["signal"]["signal_vector"]
            active_scores = [abs(sv[t]["score"]) for t in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"] if t in sv]
            max_scores.append(max(active_scores) if active_scores else 0)
        
        avg_max = sum(max_scores) / len(max_scores)
        actual_max = max(max_scores)
        all_zero = sum(1 for s in max_scores if s < 0.01)
        
        print(f"  {tier:>12}: avg_max_score={avg_max:.3f}, actual_max={actual_max:.3f}, all_zero={all_zero}/{len(tier_records)}")
        
        # Validate expectations
        if tier == "important" and avg_max < 0.2:
            print(f"    ⚠️  Important papers have low scores (avg_max={avg_max:.3f}). Expected 0.3-0.5")
        if tier == "unrelated" and avg_max > 0.05:
            print(f"    ⚠️  Unrelated papers have non-zero scores (avg_max={avg_max:.3f}). Expected ~0.0")
        if tier == "incremental" and avg_max > 0.2:
            print(f"    ⚠️  Incremental papers have high scores (avg_max={avg_max:.3f}). Expected 0.0-0.1")


def check_qnt_quality(records):
    """Specific checks for QNT examples."""
    print(f"\n--- QNT Quality Check ({len(records)} records) ---")
    
    successful = [r for r in records if r.get("success") and r.get("signal")]
    
    # Check by type
    types = Counter(r.get("qnt_type", "unknown") for r in records)
    print(f"  Types: {dict(types)}")
    
    # Check IONQ-QNT relationship
    same_direction = 0
    opposite_direction = 0
    
    for r in successful:
        sv = r["signal"]["signal_vector"]
        ionq_score = sv.get("IONQ", {}).get("score", 0)
        qnt_score = sv.get("QNT", {}).get("score", 0)
        
        if abs(ionq_score) < 0.05 or abs(qnt_score) < 0.05:
            continue  # Skip near-zero
        
        if (ionq_score > 0 and qnt_score > 0) or (ionq_score < 0 and qnt_score < 0):
            same_direction += 1
        else:
            opposite_direction += 1
    
    print(f"  IONQ-QNT relationship:")
    print(f"    Same direction: {same_direction}")
    print(f"    Opposite direction: {opposite_direction}")
    
    # For sector-wide, expect mostly same direction
    sector_records = [r for r in successful if r.get("qnt_type") == "sector_wide"]
    sector_same = 0
    for r in sector_records:
        sv = r["signal"]["signal_vector"]
        ionq = sv.get("IONQ", {}).get("score", 0)
        qnt = sv.get("QNT", {}).get("score", 0)
        if abs(ionq) > 0.05 and abs(qnt) > 0.05:
            if (ionq > 0 and qnt > 0) or (ionq < 0 and qnt < 0):
                sector_same += 1
    
    print(f"    Sector-wide same direction: {sector_same}/{len(sector_records)}")
    
    # For competitive, expect some opposite
    comp_records = [r for r in successful if r.get("qnt_type") == "competitive"]
    comp_opposite = 0
    for r in comp_records:
        sv = r["signal"]["signal_vector"]
        ionq = sv.get("IONQ", {}).get("score", 0)
        qnt = sv.get("QNT", {}).get("score", 0)
        if abs(ionq) > 0.05 and abs(qnt) > 0.05:
            if (ionq > 0 and qnt < 0) or (ionq < 0 and qnt > 0):
                comp_opposite += 1
    
    print(f"    Competitive opposite direction: {comp_opposite}/{len(comp_records)}")


def check_market_context_quality(records):
    """Check market context formatting."""
    print(f"\n--- Market Context Quality ---")
    
    with_context = [r for r in records if r.get("market_context")]
    print(f"  Records with market context: {len(with_context)}")
    
    if with_context:
        # Check formatting
        sample = random.choice(with_context)
        ctx = sample["market_context"]
        print(f"  Sample context ({len(ctx)} chars):")
        for line in ctx.split("\n")[:8]:
            print(f"    {line}")
        
        # Check all have proper table format
        malformed = 0
        for r in with_context:
            ctx = r["market_context"]
            if "| Ticker |" not in ctx:
                malformed += 1
        
        if malformed:
            print(f"  ⚠️  {malformed} records have malformed market context")
        else:
            print(f"  ✓ All {len(with_context)} market context blocks properly formatted")


def main():
    print("=" * 70)
    print("DATA QUALITY SPOT CHECK")
    print("=" * 70)
    
    data = load_all_data()
    
    # ============================================================
    # 1. Structural checks on ALL data
    # ============================================================
    print("\n" + "=" * 70)
    print("1. STRUCTURAL CHECKS")
    print("=" * 70)
    
    all_issues = []
    
    # Combined data
    for i, r in enumerate(data["combined"]):
        issues = check_json_structure(r, f"combined[{i}]")
        all_issues.extend(issues)
    
    # ArXiv data
    for i, r in enumerate(data["arxiv"]):
        issues = check_json_structure(r, f"arxiv[{i}]")
        all_issues.extend(issues)
    
    # QNT data
    for i, r in enumerate(data["qnt"]):
        issues = check_json_structure(r, f"qnt[{i}]")
        all_issues.extend(issues)
    
    if all_issues:
        print(f"\n  ⚠️  {len(all_issues)} structural issues found:")
        for issue in all_issues[:20]:
            print(f"    {issue}")
        if len(all_issues) > 20:
            print(f"    ... and {len(all_issues) - 20} more")
    else:
        print(f"\n  ✓ All records pass structural checks")
    
    # ============================================================
    # 2. Score distributions
    # ============================================================
    print("\n" + "=" * 70)
    print("2. SCORE DISTRIBUTIONS")
    print("=" * 70)
    
    check_score_distributions(data["combined"], "Combined (1000)")
    check_score_distributions(data["arxiv"], "ArXiv Rebalance (70)")
    check_score_distributions(data["qnt"], "QNT Examples (35)")
    
    # ============================================================
    # 3. Spot-check random examples
    # ============================================================
    print("\n" + "=" * 70)
    print("3. RANDOM SPOT CHECKS")
    print("=" * 70)
    
    # Sample from different categories in combined
    categories = {}
    for r in data["combined"]:
        cat = r.get("category", "unknown")
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(r)
    
    for cat, records in categories.items():
        spot_check_examples(records, f"Combined/{cat}", n=2)
    
    spot_check_examples(data["arxiv"], "ArXiv Rebalance", n=3)
    spot_check_examples(data["qnt"], "QNT Examples", n=3)
    
    # ============================================================
    # 4. Category-specific quality checks
    # ============================================================
    print("\n" + "=" * 70)
    print("4. CATEGORY-SPECIFIC QUALITY")
    print("=" * 70)
    
    check_arxiv_quality(data["arxiv"])
    check_qnt_quality(data["qnt"])
    check_market_context_quality(data["combined"])
    
    # ============================================================
    # 5. Summary
    # ============================================================
    print("\n" + "=" * 70)
    print("5. SUMMARY")
    print("=" * 70)
    
    total_records = len(data["combined"]) + len(data["arxiv"]) + len(data["qnt"])
    total_successful = (
        sum(1 for r in data["combined"] if r.get("success")) +
        sum(1 for r in data["arxiv"] if r.get("success")) +
        sum(1 for r in data["qnt"] if r.get("success"))
    )
    
    print(f"  Total records: {total_records}")
    print(f"  Total successful: {total_successful}")
    print(f"  Success rate: {total_successful/total_records*100:.1f}%")
    print(f"  Structural issues: {len(all_issues)}")
    print()
    
    if all_issues:
        print("  STATUS: ⚠️  ISSUES FOUND - review above")
        return 1
    else:
        print("  STATUS: ✓ ALL QUALITY CHECKS PASSED")
        return 0


if __name__ == "__main__":
    sys.exit(main())
