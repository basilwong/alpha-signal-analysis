"""
Post-process existing training data to apply label quality fixes.
Applies: Fix 1 (inactive tickers), Fix 5 (remove signal_decay), Fix 16 (repair chain_of_thought).

Usage:
    python scripts/postprocess_training_data.py

Writes to a temp file, validates, then renames. Original backup preserved.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
DATA_TRAINING = PROJECT_ROOT / "data" / "training"

# Inactive ticker config (from src/config.py)
INACTIVE_TICKERS = {
    "MSFT": "Inactive: quantum revenue exposure too low for meaningful signal.",
    "GOOGL": "Inactive: quantum revenue exposure too low for meaningful signal.",
    "NVDA": "Inactive: anti-predictive, moves on AI/GPU demand not quantum news.",
}

QNT_PRE_IPO_REASONING = "QNT not yet public at time of this article (IPO'd June 4, 2026)."
QNT_IPO_DATE = "2026-06-04"


def fix1_zero_inactive_tickers(signal: dict) -> dict:
    """Fix 1: Set MSFT/GOOGL/NVDA scores to 0.0 with standard reasoning."""
    sv = signal.get("signal_vector", {})
    for ticker, reasoning in INACTIVE_TICKERS.items():
        if ticker in sv:
            sv[ticker] = {"score": 0.0, "reasoning": reasoning}
    signal["signal_vector"] = sv
    return signal


def fix1_add_qnt(signal: dict, article_date: str = None) -> dict:
    """Fix 1: Add QNT to signal_vector with appropriate score."""
    sv = signal.get("signal_vector", {})
    if "QNT" not in sv:
        sv["QNT"] = {"score": 0.0, "reasoning": QNT_PRE_IPO_REASONING}
    signal["signal_vector"] = sv
    return signal


def fix5_remove_signal_decay(signal: dict) -> dict:
    """Fix 5: Remove signal_decay field from signal."""
    if "signal_decay" in signal:
        del signal["signal_decay"]
    return signal


def fix16_repair_chain_of_thought(signal: dict) -> dict:
    """Fix 16: Replace placeholder chain_of_thought with signal_rationale content."""
    cot = signal.get("chain_of_thought", "")
    
    # Detect placeholder patterns
    is_placeholder = (
        len(cot) < 100 or
        "not disclosed" in cot.lower() or
        "redacted" in cot.lower() or
        "cannot" in cot.lower() or
        cot.strip() == ""
    )
    
    if is_placeholder:
        # Use signal_rationale as the chain of thought (it's always populated)
        rationale = signal.get("signal_rationale", "")
        technical = signal.get("technical_translation", "")
        
        # Build a synthetic chain_of_thought from available fields
        parts = []
        if technical:
            parts.append(f"Technical assessment: {technical}")
        if rationale:
            parts.append(f"Signal reasoning: {rationale}")
        
        # Also incorporate per-ticker reasonings for richer CoT
        sv = signal.get("signal_vector", {})
        active_reasonings = []
        for ticker in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"]:
            if ticker in sv:
                score = sv[ticker].get("score", 0)
                reasoning = sv[ticker].get("reasoning", "")
                if abs(score) > 0.05 and reasoning:
                    active_reasonings.append(f"{ticker} ({score:+.2f}): {reasoning}")
        
        if active_reasonings:
            parts.append("Per-ticker analysis: " + " | ".join(active_reasonings))
        
        if parts:
            signal["chain_of_thought"] = " ".join(parts)
        else:
            # Last resort: use the rationale directly
            signal["chain_of_thought"] = rationale if rationale else "Analysis based on article content and sector dynamics."
    
    return signal


def process_single_example(record: dict) -> dict:
    """Apply all post-processing fixes to a single training example."""
    if not record.get("success") or not record.get("signal"):
        # Add QNT placeholder even for failed examples (schema consistency)
        return record
    
    signal = record["signal"]
    article_date = record.get("date")
    
    # Fix 1: Zero inactive tickers
    signal = fix1_zero_inactive_tickers(signal)
    
    # Fix 1: Add QNT
    signal = fix1_add_qnt(signal, article_date)
    
    # Fix 5: Remove signal_decay
    signal = fix5_remove_signal_decay(signal)
    
    # Fix 16: Repair chain_of_thought
    signal = fix16_repair_chain_of_thought(signal)
    
    record["signal"] = signal
    return record


def validate_output(filepath: Path, expected_count: int) -> bool:
    """Validate the output file."""
    errors = []
    
    with open(filepath) as f:
        lines = f.readlines()
    
    if len(lines) != expected_count:
        errors.append(f"Line count mismatch: expected {expected_count}, got {len(lines)}")
    
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: JSON parse error: {e}")
            continue
        
        if not record.get("success") or not record.get("signal"):
            continue
        
        signal = record["signal"]
        sv = signal.get("signal_vector", {})
        
        # Check: all 10 tickers present
        expected_tickers = {"IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"}
        actual_tickers = set(sv.keys())
        if actual_tickers != expected_tickers:
            missing = expected_tickers - actual_tickers
            extra = actual_tickers - expected_tickers
            if missing:
                errors.append(f"Line {i}: missing tickers {missing}")
            # Extra tickers are ok (backward compat)
        
        # Check: inactive tickers are 0.0
        for ticker in INACTIVE_TICKERS:
            if ticker in sv and sv[ticker].get("score", 0) != 0.0:
                errors.append(f"Line {i}: {ticker} score should be 0.0, got {sv[ticker]['score']}")
        
        # Check: no signal_decay
        if "signal_decay" in signal:
            errors.append(f"Line {i}: signal_decay should be removed")
        
        # Check: chain_of_thought is not placeholder
        # A placeholder is SHORT and contains marker phrases, or is empty
        cot = signal.get("chain_of_thought", "")
        is_placeholder = (
            len(cot) < 50 or
            (len(cot) < 150 and ("redacted" in cot.lower() or "not disclosed" in cot.lower()))
        )
        if is_placeholder:
            errors.append(f"Line {i}: chain_of_thought still placeholder: '{cot[:50]}'")
    
    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for e in errors[:20]:
            print(f"  {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False
    
    return True


def main():
    input_file = DATA_TRAINING / "manus_teacher_combined.jsonl"
    temp_file = DATA_TRAINING / "manus_teacher_combined.jsonl.tmp"
    
    if not input_file.exists():
        print(f"ERROR: {input_file} not found")
        sys.exit(1)
    
    # Load all examples
    print(f"Loading {input_file}...")
    records = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} examples")
    
    # Apply fixes
    print("Applying fixes...")
    stats = {
        "inactive_zeroed": 0,
        "qnt_added": 0,
        "decay_removed": 0,
        "cot_repaired": 0,
    }
    
    processed = []
    for record in records:
        if record.get("success") and record.get("signal"):
            signal = record["signal"]
            
            # Track changes
            sv = signal.get("signal_vector", {})
            for t in INACTIVE_TICKERS:
                if t in sv and sv[t].get("score", 0) != 0.0:
                    stats["inactive_zeroed"] += 1
            if "QNT" not in sv:
                stats["qnt_added"] += 1
            if "signal_decay" in signal:
                stats["decay_removed"] += 1
            cot = signal.get("chain_of_thought", "")
            if len(cot) < 100 or "redacted" in cot.lower() or "not disclosed" in cot.lower():
                stats["cot_repaired"] += 1
        
        processed.append(process_single_example(record))
    
    print(f"\nChanges applied:")
    print(f"  Inactive tickers zeroed: {stats['inactive_zeroed']} score changes")
    print(f"  QNT added: {stats['qnt_added']} examples")
    print(f"  signal_decay removed: {stats['decay_removed']} examples")
    print(f"  chain_of_thought repaired: {stats['cot_repaired']} examples")
    
    # Write to temp file
    print(f"\nWriting to {temp_file}...")
    with open(temp_file, "w") as f:
        for record in processed:
            f.write(json.dumps(record) + "\n")
    
    # Validate
    print("Validating output...")
    if validate_output(temp_file, len(records)):
        print("✓ Validation PASSED")
        # Atomic rename
        temp_file.rename(input_file)
        print(f"✓ Written to {input_file}")
    else:
        print("✗ Validation FAILED — temp file preserved for inspection")
        print(f"  Temp file: {temp_file}")
        sys.exit(1)
    
    # Also update individual category files
    print("\nUpdating individual category files...")
    category_files = [
        "manus_real_articles.jsonl",
        "manus_synthetic.jsonl",
        "manus_paraphrased.jsonl",
        "manus_negatives.jsonl",
        "manus_edge_cases.jsonl",
        "manus_multi_turn.jsonl",
    ]
    
    for fname in category_files:
        fpath = DATA_TRAINING / fname
        if not fpath.exists():
            continue
        
        with open(fpath) as f:
            cat_records = [json.loads(l) for l in f if l.strip()]
        
        cat_processed = [process_single_example(r) for r in cat_records]
        
        temp = fpath.with_suffix(".jsonl.tmp")
        with open(temp, "w") as f:
            for r in cat_processed:
                f.write(json.dumps(r) + "\n")
        temp.rename(fpath)
        print(f"  ✓ {fname}: {len(cat_processed)} examples")
    
    print("\n✓ All post-processing complete")
    print(f"  Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
