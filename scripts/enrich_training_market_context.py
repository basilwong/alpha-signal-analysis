"""
Fix 3a: Retroactively enrich existing training data with market context.
Fix 4: Compute teacher_market_accuracy metadata.
Fix 12: Add market_regime tag.

Reads each training example's date, computes market context from parquet files,
and adds it to the record. Also computes teacher accuracy vs actual 5d returns.

Usage:
    python scripts/enrich_training_market_context.py
"""

import json
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_context import get_market_context, get_5d_forward_returns, get_market_regime
from src.config import ACTIVE_TICKERS, INACTIVE_TICKERS

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
MARKET_DIR = PROJECT_ROOT / "data" / "market"


def compute_teacher_accuracy(signal: dict, forward_returns: dict) -> float:
    """
    Fix 4: Compute fraction of active tickers where teacher's direction matches actual.
    
    Ignores tickers where:
    - predicted score is 0.0 (no opinion)
    - actual return is very small (< 0.5% absolute)
    """
    if not forward_returns:
        return None
    
    correct = 0
    total = 0
    
    sv = signal.get("signal_vector", {})
    
    for ticker in ACTIVE_TICKERS:
        if ticker not in sv or ticker not in forward_returns:
            continue
        
        predicted = sv[ticker].get("score", 0)
        actual = forward_returns[ticker]
        
        # Skip if no opinion or trivial move
        if abs(predicted) < 0.01:
            continue
        if abs(actual) < 0.005:  # < 0.5% move is noise
            continue
        
        total += 1
        if (predicted > 0 and actual > 0) or (predicted < 0 and actual < 0):
            correct += 1
    
    if total == 0:
        return None
    
    return correct / total


def main():
    input_file = DATA_TRAINING / "manus_teacher_combined.jsonl"
    temp_file = DATA_TRAINING / "manus_teacher_combined.jsonl.tmp"
    
    print(f"Loading {input_file}...")
    records = []
    with open(input_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    print(f"Loaded {len(records)} examples")
    print(f"Market data directory: {MARKET_DIR}")
    
    # Stats
    enriched_count = 0
    accuracy_count = 0
    accuracy_values = []
    
    for i, record in enumerate(records):
        date = record.get("date")
        
        if not date:
            # No date (synthetic, edge cases, negatives without dates)
            record["market_context"] = ""
            record["market_regime"] = ""
            record["teacher_market_accuracy"] = None
            continue
        
        # Fix 3a: Add market context
        context = get_market_context(date, market_dir=MARKET_DIR)
        record["market_context"] = context
        if context:
            enriched_count += 1
        
        # Fix 12: Add market regime
        regime = get_market_regime(date, market_dir=MARKET_DIR)
        record["market_regime"] = regime
        
        # Fix 4: Compute teacher accuracy
        if record.get("success") and record.get("signal"):
            forward_returns = get_5d_forward_returns(date, market_dir=MARKET_DIR)
            accuracy = compute_teacher_accuracy(record["signal"], forward_returns)
            record["teacher_market_accuracy"] = accuracy
            if accuracy is not None:
                accuracy_count += 1
                accuracy_values.append(accuracy)
    
    # Print stats
    print(f"\nEnrichment results:")
    print(f"  Market context added: {enriched_count} examples")
    print(f"  Teacher accuracy computed: {accuracy_count} examples")
    if accuracy_values:
        import statistics
        print(f"  Teacher accuracy stats:")
        print(f"    Mean: {statistics.mean(accuracy_values):.3f}")
        print(f"    Median: {statistics.median(accuracy_values):.3f}")
        print(f"    Min: {min(accuracy_values):.3f}")
        print(f"    Max: {max(accuracy_values):.3f}")
    
    # Write to temp file
    print(f"\nWriting to {temp_file}...")
    with open(temp_file, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    
    # Validate line count
    with open(temp_file) as f:
        line_count = sum(1 for _ in f)
    
    if line_count != len(records):
        print(f"ERROR: Line count mismatch: {line_count} vs {len(records)}")
        sys.exit(1)
    
    # Atomic rename
    temp_file.rename(input_file)
    print(f"✓ Written to {input_file}")
    print(f"  Timestamp: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
