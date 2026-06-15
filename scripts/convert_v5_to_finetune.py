"""
Convert V5 raw output to final fine-tuning format.

Produces alpha_signal_train_v5.jsonl with:
- System message (V5 system prompt)
- User message (market context + article)
- Assistant message (<think>...</think> + JSON)

Usage:
    python scripts/convert_v5_to_finetune.py
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_v5_thinking import V5_SYSTEM_PROMPT, build_user_message

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
INPUT_FILE = DATA_TRAINING / "alpha_signal_train_v5_raw.jsonl"
OUTPUT_FILE = DATA_TRAINING / "alpha_signal_train_v5.jsonl"


def main():
    print("Converting V5 raw data to fine-tuning format...")
    
    with open(INPUT_FILE) as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    successes = [r for r in records if r.get("success") and r.get("signal") and r.get("thinking")]
    print(f"Total records: {len(records)}")
    print(f"With thinking + signal: {len(successes)}")
    
    converted = []
    issues = []
    
    for r in successes:
        thinking = r["thinking"]
        signal = r["signal"]
        
        # Build assistant message: <think>...</think>\nJSON
        signal_json = json.dumps(signal)
        assistant_msg = f"<think>\n{thinking}\n</think>\n{signal_json}"
        
        # Build user message
        user_msg = build_user_message(r)
        
        # Token estimate
        total_chars = len(V5_SYSTEM_PROMPT) + len(user_msg) + len(assistant_msg)
        est_tokens = total_chars // 4
        
        if est_tokens > 4096:
            # Truncate thinking to fit
            excess = total_chars - (4096 * 4)
            new_think_len = max(100, len(thinking) - excess - 50)
            thinking = thinking[:new_think_len] + "..."
            assistant_msg = f"<think>\n{thinking}\n</think>\n{signal_json}"
            issues.append(f"idx={r['idx']}: truncated thinking to fit 4096 tokens")
        
        chat_record = {
            "messages": [
                {"role": "system", "content": V5_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
        }
        converted.append(chat_record)
    
    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for record in converted:
            f.write(json.dumps(record) + "\n")
    
    print(f"\nConverted: {len(converted)} examples")
    print(f"Truncated: {len(issues)}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Validation
    print("\n--- Validation ---")
    
    errors = 0
    for i, record in enumerate(converted):
        assistant = record["messages"][2]["content"]
        
        # Check format
        if not assistant.startswith("<think>\n"):
            errors += 1
            print(f"  ERROR line {i}: doesn't start with <think>")
            continue
        
        if "</think>\n{" not in assistant:
            errors += 1
            print(f"  ERROR line {i}: missing </think> before JSON")
            continue
        
        # Extract and validate JSON
        think_end = assistant.find("</think>")
        json_str = assistant[think_end + len("</think>"):].strip()
        
        try:
            parsed = json.loads(json_str)
        except json.JSONDecodeError:
            errors += 1
            print(f"  ERROR line {i}: JSON parse error")
            continue
        
        # Check tickers
        sv = parsed.get("signal_vector", {})
        if len(sv) < 10:
            errors += 1
            print(f"  ERROR line {i}: only {len(sv)} tickers")
        
        # Check inactive
        for t in ["MSFT", "GOOGL", "NVDA"]:
            if sv.get(t, {}).get("score", 0) != 0.0:
                errors += 1
                print(f"  ERROR line {i}: {t} != 0.0")
    
    # Token stats
    lengths = [sum(len(m["content"]) for m in r["messages"]) // 4 for r in converted]
    print(f"\nToken stats: min={min(lengths)}, avg={sum(lengths)//len(lengths)}, max={max(lengths)}")
    print(f"Over 4096: {sum(1 for l in lengths if l > 4096)}")
    print(f"\nValidation errors: {errors}")
    
    if errors == 0:
        print("✓ ALL VALIDATION CHECKS PASSED")
    else:
        print(f"✗ {errors} errors found")


if __name__ == "__main__":
    main()
