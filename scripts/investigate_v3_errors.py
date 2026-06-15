"""
Investigate V3 training data quality and JSON error patterns.

1. Compare V2 vs V3 training data formatting
2. Look at the raw model output for failed predictions to understand the error
3. Check if V3 training examples have consistent JSON formatting
"""

import json
import re
from collections import Counter

# 1. Compare V2 and V3 training data
print("=" * 60)
print("TRAINING DATA COMPARISON: V2 vs V3")
print("=" * 60)

def analyze_training_file(path, label):
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    
    print(f"\n{label}: {len(examples)} examples")
    
    # Check the assistant responses (the JSON the model learns to produce)
    valid_json = 0
    invalid_json = 0
    has_signal_rationale = 0
    avg_response_length = 0
    
    for ex in examples:
        messages = ex.get("messages", [])
        assistant_msg = None
        for msg in messages:
            if msg.get("role") == "assistant":
                assistant_msg = msg.get("content", "")
                break
        
        if not assistant_msg:
            continue
        
        avg_response_length += len(assistant_msg)
        
        try:
            parsed = json.loads(assistant_msg)
            valid_json += 1
            if "signal_rationale" in parsed:
                has_signal_rationale += 1
                # Check if signal_rationale contains problematic characters
                rationale = parsed.get("signal_rationale", "")
                if '"' in rationale or '\n' in rationale:
                    pass  # This is fine if properly escaped in the JSON
        except json.JSONDecodeError as e:
            invalid_json += 1
            if invalid_json <= 3:
                print(f"  INVALID JSON in training example: {str(e)[:100]}")
                print(f"  Response preview: {assistant_msg[:200]}")
    
    avg_response_length /= max(1, len(examples))
    print(f"  Valid JSON responses: {valid_json}/{len(examples)}")
    print(f"  Invalid JSON responses: {invalid_json}/{len(examples)}")
    print(f"  Has signal_rationale field: {has_signal_rationale}/{valid_json}")
    print(f"  Avg response length: {avg_response_length:.0f} chars")
    
    return examples

v2_examples = analyze_training_file("data/training/alpha_signal_train_v2.jsonl", "V2 (May-Jun 2026)")
v3_examples = analyze_training_file("data/training/alpha_signal_train_v3.jsonl", "V3 (Aug 2024 - Dec 2025)")
combined_examples = analyze_training_file("data/training/alpha_signal_train_combined.jsonl", "Combined (V2+V3)")

# 2. Analyze the error pattern in predictions
print("\n" + "=" * 60)
print("PREDICTION ERROR ANALYSIS")
print("=" * 60)

errors = []
successes = []
with open("data/eval/predictions_v3_final.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "error":
                errors.append(r)
            elif r.get("status") == "success":
                successes.append(r)

print(f"\nTotal: {len(successes)} successes, {len(errors)} errors")

# Error pattern analysis
error_messages = Counter()
for e in errors:
    msg = e.get("error", "")
    # Normalize the error message (remove specific char positions)
    normalized = re.sub(r'line \d+ column \d+ \(char \d+\)', 'line X column Y (char Z)', msg)
    error_messages[normalized] += 1

print(f"\nError patterns:")
for msg, count in error_messages.most_common(5):
    print(f"  {count:4d}x: {msg[:100]}")

# 3. Look at raw output from a few errors to understand what's wrong
print("\n" + "=" * 60)
print("SAMPLE RAW OUTPUTS FROM ERRORS (if available)")
print("=" * 60)

# The predictions file might not have raw output, but let's check
for e in errors[:3]:
    print(f"\n  Article: {e.get('title', 'N/A')[:60]}")
    print(f"  Error: {e.get('error', 'N/A')}")
    if "raw" in e:
        print(f"  Raw output: {e['raw'][:300]}")

# 4. Compare V2 vs V3 training data structure differences
print("\n" + "=" * 60)
print("STRUCTURAL DIFFERENCES")
print("=" * 60)

def get_json_structure(examples):
    """Get the structure of assistant responses."""
    structures = Counter()
    for ex in examples:
        messages = ex.get("messages", [])
        for msg in messages:
            if msg.get("role") == "assistant":
                try:
                    parsed = json.loads(msg["content"])
                    keys = sorted(parsed.keys())
                    structures[str(keys)] += 1
                except:
                    structures["INVALID"] += 1
    return structures

v2_structures = get_json_structure(v2_examples)
v3_structures = get_json_structure(v3_examples)

print("\nV2 response structures:")
for struct, count in v2_structures.most_common(3):
    print(f"  {count}x: {struct[:120]}")

print("\nV3 response structures:")
for struct, count in v3_structures.most_common(3):
    print(f"  {count}x: {struct[:120]}")

# 5. Check for formatting inconsistencies in V3
print("\n" + "=" * 60)
print("V3 FORMATTING ANALYSIS")
print("=" * 60)

v3_indent_styles = Counter()
for ex in v3_examples:
    messages = ex.get("messages", [])
    for msg in messages:
        if msg.get("role") == "assistant":
            content = msg["content"]
            if content.startswith("{\n  "):
                v3_indent_styles["2-space indent"] += 1
            elif content.startswith("{\n    "):
                v3_indent_styles["4-space indent"] += 1
            elif content.startswith("{\""):
                v3_indent_styles["compact (no indent)"] += 1
            else:
                v3_indent_styles["other"] += 1

print(f"V3 JSON formatting styles:")
for style, count in v3_indent_styles.most_common():
    print(f"  {count}x: {style}")
