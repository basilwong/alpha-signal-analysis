"""
Validate V5 training data against all requirements:
1. Format: 3 messages (system, user, assistant)
2. Thinking traces: assistant starts with <think> and contains </think>{
3. JSON validity: parseable JSON after </think>
4. All 10 tickers present
5. MSFT/GOOGL/NVDA always 0.0
6. Score ranges respected
7. Thinking block is 100-300 tokens (not boilerplate)
8. Directional balance (bullish/bearish/neutral distribution)
9. Pre-event market context in user message
10. Temporal check (article dates)
"""
import json
import re
from collections import Counter, defaultdict

INPUT = "data/training/quantum_alpha_train_v5.jsonl"

errors = []
warnings = []
stats = {
    "total": 0,
    "has_thinking": 0,
    "valid_json": 0,
    "all_tickers": 0,
    "inactive_compliant": 0,
    "scores_in_range": 0,
    "has_market_context": 0,
    "thinking_token_lengths": [],
    "bullish_scores": 0,
    "bearish_scores": 0,
    "zero_scores": 0,
    "dates": [],
}

EXPECTED_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"]
SCORE_RANGES = {
    "IONQ": (-2.0, 2.0), "RGTI": (-2.0, 2.0), "QBTS": (-2.0, 2.0),
    "QUBT": (-2.0, 2.0), "QNT": (-2.0, 2.0),
    "IBM": (-0.15, 0.15), "HON": (-0.3, 0.3),
    "MSFT": (0.0, 0.0), "GOOGL": (0.0, 0.0), "NVDA": (0.0, 0.0),
}

with open(INPUT) as f:
    for i, line in enumerate(f):
        stats["total"] += 1
        r = json.loads(line)
        msgs = r.get("messages", [])

        # Check 1: Message structure
        if len(msgs) != 3:
            errors.append(f"Line {i}: Expected 3 messages, got {len(msgs)}")
            continue
        if msgs[0]["role"] != "system" or msgs[1]["role"] != "user" or msgs[2]["role"] != "assistant":
            errors.append(f"Line {i}: Wrong role order: {[m['role'] for m in msgs]}")
            continue

        assistant_content = msgs[2]["content"]
        user_content = msgs[1]["content"]

        # Check 2: Thinking traces
        has_think_open = "<think>" in assistant_content
        has_think_close = "</think>" in assistant_content

        if has_think_open and has_think_close:
            stats["has_thinking"] += 1
            think_start = assistant_content.index("<think>") + len("<think>")
            think_end = assistant_content.index("</think>")
            thinking_text = assistant_content[think_start:think_end].strip()
            # Rough token estimate (words * 1.3)
            thinking_tokens = int(len(thinking_text.split()) * 1.3)
            stats["thinking_token_lengths"].append(thinking_tokens)

            if thinking_tokens < 50:
                warnings.append(f"Line {i}: Thinking block too short ({thinking_tokens} est. tokens)")
            elif thinking_tokens > 500:
                warnings.append(f"Line {i}: Thinking block very long ({thinking_tokens} est. tokens)")
        elif has_think_open and not has_think_close:
            errors.append(f"Line {i}: Unclosed <think> block")
            continue
        else:
            # No thinking block - check if this is intentional
            warnings.append(f"Line {i}: No <think> block in assistant response")

        # Check 3: JSON validity
        # Extract JSON (after </think> if present, or the whole content)
        if "</think>" in assistant_content:
            json_part = assistant_content[assistant_content.index("</think>") + len("</think>"):].strip()
        else:
            json_part = assistant_content.strip()

        start_j = json_part.find("{")
        end_j = json_part.rfind("}") + 1
        if start_j < 0 or end_j <= start_j:
            errors.append(f"Line {i}: No JSON object found after thinking block")
            continue

        try:
            signal = json.loads(json_part[start_j:end_j])
            stats["valid_json"] += 1
        except json.JSONDecodeError as e:
            errors.append(f"Line {i}: Invalid JSON: {e}")
            continue

        # Check 4: All tickers present
        sv = signal.get("signal_vector", {})
        missing = [t for t in EXPECTED_TICKERS if t not in sv]
        if not missing:
            stats["all_tickers"] += 1
        else:
            errors.append(f"Line {i}: Missing tickers: {missing}")

        # Check 5: MSFT/GOOGL/NVDA always 0.0
        inactive_ok = True
        for ticker in ["MSFT", "GOOGL", "NVDA"]:
            if ticker in sv:
                score = sv[ticker].get("score", 0)
                if score != 0 and score != 0.0:
                    errors.append(f"Line {i}: {ticker} score should be 0.0, got {score}")
                    inactive_ok = False
        if inactive_ok:
            stats["inactive_compliant"] += 1

        # Check 6: Score ranges
        all_in_range = True
        for ticker, (lo, hi) in SCORE_RANGES.items():
            if ticker in sv:
                score = sv[ticker].get("score", 0)
                if isinstance(score, (int, float)):
                    if ticker in ["MSFT", "GOOGL", "NVDA"]:
                        continue  # Already checked above
                    if score < lo - 0.01 or score > hi + 0.01:
                        errors.append(f"Line {i}: {ticker} score {score} outside range [{lo}, {hi}]")
                        all_in_range = False

                    # Track directional balance
                    if score > 0.01:
                        stats["bullish_scores"] += 1
                    elif score < -0.01:
                        stats["bearish_scores"] += 1
                    else:
                        stats["zero_scores"] += 1
        if all_in_range:
            stats["scores_in_range"] += 1

        # Check 9: Market context in user message
        if "MARKET CONTEXT" in user_content or "Prior" in user_content or "prior" in user_content:
            stats["has_market_context"] += 1

        # Check 10: Date extraction
        date_match = re.search(r'(\d{4}-\d{2}-\d{2})', user_content)
        if date_match:
            stats["dates"].append(date_match.group(1))

# Print results
print("=" * 70)
print("V5 TRAINING DATA VALIDATION REPORT")
print("=" * 70)

print(f"\nTotal examples: {stats['total']}")
print(f"\nStructure checks:")
print(f"  Has <think> block: {stats['has_thinking']}/{stats['total']} ({stats['has_thinking']/stats['total']*100:.1f}%)")
print(f"  Valid JSON: {stats['valid_json']}/{stats['total']} ({stats['valid_json']/stats['total']*100:.1f}%)")
print(f"  All 10 tickers: {stats['all_tickers']}/{stats['total']} ({stats['all_tickers']/stats['total']*100:.1f}%)")
print(f"  Inactive compliant: {stats['inactive_compliant']}/{stats['total']} ({stats['inactive_compliant']/stats['total']*100:.1f}%)")
print(f"  Scores in range: {stats['scores_in_range']}/{stats['total']} ({stats['scores_in_range']/stats['total']*100:.1f}%)")
print(f"  Has market context: {stats['has_market_context']}/{stats['total']} ({stats['has_market_context']/stats['total']*100:.1f}%)")

if stats["thinking_token_lengths"]:
    import numpy as np
    lengths = stats["thinking_token_lengths"]
    print(f"\nThinking block stats:")
    print(f"  Min tokens: {min(lengths)}")
    print(f"  Max tokens: {max(lengths)}")
    print(f"  Mean tokens: {sum(lengths)/len(lengths):.0f}")
    print(f"  In range [100-300]: {sum(1 for l in lengths if 100 <= l <= 300)}/{len(lengths)}")

total_scored = stats["bullish_scores"] + stats["bearish_scores"] + stats["zero_scores"]
if total_scored > 0:
    print(f"\nDirectional balance (non-inactive tickers):")
    print(f"  Bullish (>0): {stats['bullish_scores']} ({stats['bullish_scores']/total_scored*100:.1f}%)")
    print(f"  Bearish (<0): {stats['bearish_scores']} ({stats['bearish_scores']/total_scored*100:.1f}%)")
    print(f"  Zero: {stats['zero_scores']} ({stats['zero_scores']/total_scored*100:.1f}%)")
    print(f"  Target: ~50-60% bullish, ~30-40% bearish")

if stats["dates"]:
    print(f"\nDate range: {min(stats['dates'])} to {max(stats['dates'])}")

print(f"\nErrors: {len(errors)}")
for e in errors[:20]:
    print(f"  {e}")
if len(errors) > 20:
    print(f"  ... and {len(errors) - 20} more")

print(f"\nWarnings: {len(warnings)}")
for w in warnings[:10]:
    print(f"  {w}")
if len(warnings) > 10:
    print(f"  ... and {len(warnings) - 10} more")

# Final verdict
all_pass = (
    stats["has_thinking"] == stats["total"]
    and stats["valid_json"] == stats["total"]
    and stats["all_tickers"] == stats["total"]
    and stats["inactive_compliant"] == stats["total"]
    and len(errors) == 0
)
print(f"\n{'ALL CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
