"""
Validation script: Verify all label quality fixes were applied correctly.
Run after all fixes are complete, before committing.

Usage:
    python scripts/validate_all_fixes.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
DATA_TRAINING = PROJECT_ROOT / "data" / "training"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"
DATA_MARKET = PROJECT_ROOT / "data" / "market"

EXPECTED_TICKERS = {"IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"}
INACTIVE_TICKERS = {"MSFT", "GOOGL", "NVDA"}
ACTIVE_TICKERS = {"IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"}


def check(condition: bool, msg: str, errors: list):
    """Assert a condition, log error if false."""
    if not condition:
        errors.append(msg)
        print(f"  ✗ {msg}")
    else:
        print(f"  ✓ {msg}")


def validate():
    errors = []
    
    # ============================================================
    print("\n=== 1. Combined Training File ===")
    # ============================================================
    combined_file = DATA_TRAINING / "manus_teacher_combined.jsonl"
    check(combined_file.exists(), "Combined JSONL exists", errors)
    
    if not combined_file.exists():
        print("FATAL: Combined file missing. Cannot continue.")
        return errors
    
    with open(combined_file) as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    check(len(records) == 1000, f"Line count = 1000 (got {len(records)})", errors)
    
    successful = [r for r in records if r.get("success") and r.get("signal")]
    print(f"  Successful examples: {len(successful)}")
    
    # ============================================================
    print("\n=== 2. Fix 1: Ticker Universe ===")
    # ============================================================
    for r in successful:
        sv = r["signal"].get("signal_vector", {})
        tickers = set(sv.keys())
        
        # All 10 tickers present
        if tickers != EXPECTED_TICKERS:
            missing = EXPECTED_TICKERS - tickers
            if missing:
                errors.append(f"idx={r.get('article_idx')}: missing tickers {missing}")
                break
        
        # Inactive tickers are 0.0
        for t in INACTIVE_TICKERS:
            if t in sv and sv[t].get("score", 0) != 0.0:
                errors.append(f"idx={r.get('article_idx')}: {t} score should be 0.0, got {sv[t]['score']}")
                break
    
    # Count non-zero inactive (should be 0)
    inactive_nonzero = sum(
        1 for r in successful
        for t in INACTIVE_TICKERS
        if r["signal"]["signal_vector"].get(t, {}).get("score", 0) != 0.0
    )
    check(inactive_nonzero == 0, f"All inactive tickers are 0.0 ({inactive_nonzero} violations)", errors)
    
    # QNT present in all
    qnt_present = sum(1 for r in successful if "QNT" in r["signal"]["signal_vector"])
    check(qnt_present == len(successful), f"QNT present in all {len(successful)} examples", errors)
    
    # ============================================================
    print("\n=== 3. Fix 5: No signal_decay ===")
    # ============================================================
    has_decay = sum(1 for r in successful if "signal_decay" in r["signal"])
    check(has_decay == 0, f"No signal_decay field ({has_decay} found)", errors)
    
    # ============================================================
    print("\n=== 4. Fix 16: No placeholder chain_of_thought ===")
    # ============================================================
    placeholder_count = 0
    for r in successful:
        cot = r["signal"].get("chain_of_thought", "")
        if len(cot) < 50:
            placeholder_count += 1
        elif len(cot) < 150 and ("redacted" in cot.lower() or "not disclosed" in cot.lower()):
            placeholder_count += 1
    
    check(placeholder_count == 0, f"No placeholder chain_of_thought ({placeholder_count} found)", errors)
    
    # ============================================================
    print("\n=== 5. Fix 3a: Market Context ===")
    # ============================================================
    has_context = sum(1 for r in records if r.get("market_context"))
    real_articles = sum(1 for r in records if r.get("category") == "real_articles" and r.get("date"))
    check(has_context >= 180, f"Market context added to {has_context} examples (expect ~190)", errors)
    
    # ============================================================
    print("\n=== 6. Fix 4: Teacher Accuracy Metadata ===")
    # ============================================================
    has_accuracy = sum(1 for r in records if r.get("teacher_market_accuracy") is not None)
    check(has_accuracy > 100, f"Teacher accuracy computed for {has_accuracy} examples", errors)
    
    # ============================================================
    print("\n=== 7. Fix 12: Market Regime ===")
    # ============================================================
    has_regime = sum(1 for r in records if r.get("market_regime"))
    check(has_regime >= 180, f"Market regime tagged for {has_regime} examples", errors)
    
    # ============================================================
    print("\n=== 8. Score Ranges ===")
    # ============================================================
    range_violations = 0
    for r in successful:
        sv = r["signal"]["signal_vector"]
        for ticker, data in sv.items():
            score = data.get("score", 0)
            if ticker in {"IONQ", "RGTI", "QBTS", "QUBT", "QNT"}:
                if abs(score) > 2.0:
                    range_violations += 1
            elif ticker == "IBM":
                if abs(score) > 0.15:
                    range_violations += 1
            elif ticker == "HON":
                if abs(score) > 0.3:
                    range_violations += 1
            elif ticker in INACTIVE_TICKERS:
                if score != 0.0:
                    range_violations += 1
    
    check(range_violations == 0, f"All scores within range ({range_violations} violations)", errors)
    
    # ============================================================
    print("\n=== 9. New Training Files ===")
    # ============================================================
    arxiv_file = DATA_TRAINING / "manus_arxiv_rebalance.jsonl"
    qnt_file = DATA_TRAINING / "manus_qnt_examples.jsonl"
    
    if arxiv_file.exists():
        with open(arxiv_file) as f:
            arxiv_records = [json.loads(l) for l in f if l.strip()]
        arxiv_success = sum(1 for r in arxiv_records if r.get("success"))
        print(f"  ArXiv rebalance: {len(arxiv_records)} total, {arxiv_success} successful")
        check(arxiv_success >= 50, f"ArXiv: at least 50 successful (got {arxiv_success})", errors)
    else:
        print(f"  ArXiv rebalance: FILE NOT FOUND (generation may still be running)")
    
    if qnt_file.exists():
        with open(qnt_file) as f:
            qnt_records = [json.loads(l) for l in f if l.strip()]
        qnt_success = sum(1 for r in qnt_records if r.get("success"))
        print(f"  QNT examples: {len(qnt_records)} total, {qnt_success} successful")
        check(qnt_success >= 25, f"QNT: at least 25 successful (got {qnt_success})", errors)
    else:
        print(f"  QNT examples: FILE NOT FOUND (generation may still be running)")
    
    # ============================================================
    print("\n=== 10. Market Data Files ===")
    # ============================================================
    qtum_file = DATA_MARKET / "QTUM.parquet"
    qnt_market = DATA_MARKET / "QNT.parquet"
    check(qtum_file.exists(), "QTUM.parquet exists (Fix 6)", errors)
    check(qnt_market.exists(), "QNT.parquet exists", errors)
    
    # ============================================================
    print("\n=== 11. Config File ===")
    # ============================================================
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.config import ACTIVE_TICKERS as cfg_active, INACTIVE_TICKERS as cfg_inactive, LIQUIDITY_TIERS
    
    check("QNT" in cfg_active, "QNT in ACTIVE_TICKERS", errors)
    check("MSFT" in cfg_inactive, "MSFT in INACTIVE_TICKERS", errors)
    check("GOOGL" in cfg_inactive, "GOOGL in INACTIVE_TICKERS", errors)
    check("NVDA" in cfg_inactive, "NVDA in INACTIVE_TICKERS", errors)
    check(len(LIQUIDITY_TIERS) >= 7, f"LIQUIDITY_TIERS has {len(LIQUIDITY_TIERS)} entries", errors)
    
    # ============================================================
    print(f"\n{'='*60}")
    if errors:
        print(f"VALIDATION FAILED: {len(errors)} errors")
        for e in errors[:20]:
            print(f"  ✗ {e}")
    else:
        print("✓ ALL CHECKS PASSED")
    print(f"{'='*60}")
    
    return errors


if __name__ == "__main__":
    errors = validate()
    sys.exit(1 if errors else 0)
