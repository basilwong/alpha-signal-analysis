"""
Validate all V5 training data files (original + bearish + robustness supplements).
"""
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FILES = [
    ("V5 Base", "data/training/quantum_alpha_train_v5.jsonl"),
    ("V5 Bearish", "data/training/quantum_alpha_train_v5_bearish.jsonl"),
    ("V5 Bearish B2", "data/training/quantum_alpha_train_v5_bearish_b2.jsonl"),
    ("V5 Robustness", "data/training/quantum_alpha_train_v5_robustness.jsonl"),
]

EXPECTED_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"]
SCORE_RANGES = {
    "IONQ": (-2.0, 2.0), "RGTI": (-2.0, 2.0), "QBTS": (-2.0, 2.0),
    "QUBT": (-2.0, 2.0), "QNT": (-2.0, 2.0),
    "IBM": (-0.15, 0.15), "HON": (-0.3, 0.3),
    "MSFT": (0.0, 0.0), "GOOGL": (0.0, 0.0), "NVDA": (0.0, 0.0),
}

# Aggregate stats across all files
total_all = 0
bullish_all = 0
bearish_all = 0
zero_all = 0
errors_all = []

for file_label, file_path in FILES:
    if not Path(file_path).exists():
        print(f"\n{file_label}: FILE NOT FOUND")
        continue

    errors = []
    stats = {"total": 0, "has_thinking": 0, "valid_json": 0, "all_tickers": 0,
             "inactive_ok": 0, "in_range": 0, "has_market_context": 0,
             "bullish": 0, "bearish": 0, "zero": 0, "thinking_lengths": []}

    with open(file_path) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            stats["total"] += 1
            total_all += 1

            try:
                r = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: Invalid JSON line: {e}")
                continue

            msgs = r.get("messages", [])
            if len(msgs) != 3:
                errors.append(f"Line {i}: Expected 3 messages, got {len(msgs)}")
                continue

            assistant = msgs[2]["content"]
            user = msgs[1]["content"]

            # Thinking block
            if "<think>" in assistant and "</think>" in assistant:
                stats["has_thinking"] += 1
                think_start = assistant.index("<think>") + 7
                think_end = assistant.index("</think>")
                think_text = assistant[think_start:think_end].strip()
                stats["thinking_lengths"].append(len(think_text.split()))

            # Market context
            if "Market Context" in user or "MARKET CONTEXT" in user or "market context" in user:
                stats["has_market_context"] += 1

            # JSON extraction
            if "</think>" in assistant:
                json_part = assistant[assistant.index("</think>") + 8:].strip()
            else:
                json_part = assistant.strip()

            start_j = json_part.find("{")
            end_j = json_part.rfind("}") + 1
            if start_j < 0 or end_j <= start_j:
                errors.append(f"Line {i}: No JSON found")
                continue

            try:
                signal = json.loads(json_part[start_j:end_j])
                stats["valid_json"] += 1
            except:
                # Try with trailing comma fix
                fixed = re.sub(r',\s*([}\]])', r'\1', json_part[start_j:end_j])
                try:
                    signal = json.loads(fixed)
                    stats["valid_json"] += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i}: Invalid JSON: {str(e)[:80]}")
                    continue

            sv = signal.get("signal_vector", {})

            # All tickers
            missing = [t for t in EXPECTED_TICKERS if t not in sv]
            if not missing:
                stats["all_tickers"] += 1
            else:
                errors.append(f"Line {i}: Missing tickers: {missing}")

            # Inactive compliance
            inactive_ok = True
            for t in ["MSFT", "GOOGL", "NVDA"]:
                if t in sv:
                    s = sv[t].get("score", 0)
                    if s != 0 and s != 0.0:
                        errors.append(f"Line {i}: {t}={s} (should be 0.0)")
                        inactive_ok = False
            if inactive_ok:
                stats["inactive_ok"] += 1

            # Score ranges and direction
            in_range = True
            for ticker in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"]:
                if ticker in sv:
                    score = sv[ticker].get("score", 0)
                    if isinstance(score, (int, float)):
                        lo, hi = SCORE_RANGES[ticker]
                        if score < lo - 0.01 or score > hi + 0.01:
                            errors.append(f"Line {i}: {ticker}={score} out of [{lo},{hi}]")
                            in_range = False
                        if score > 0.01:
                            stats["bullish"] += 1
                            bullish_all += 1
                        elif score < -0.01:
                            stats["bearish"] += 1
                            bearish_all += 1
                        else:
                            stats["zero"] += 1
                            zero_all += 1
            if in_range:
                stats["in_range"] += 1

    # Print per-file summary
    n = stats["total"]
    print(f"\n{'='*70}")
    print(f"{file_label} ({n} examples)")
    print(f"{'='*70}")
    print(f"  Thinking: {stats['has_thinking']}/{n} ({stats['has_thinking']/max(n,1)*100:.0f}%)")
    print(f"  Valid JSON: {stats['valid_json']}/{n} ({stats['valid_json']/max(n,1)*100:.0f}%)")
    print(f"  All tickers: {stats['all_tickers']}/{n} ({stats['all_tickers']/max(n,1)*100:.0f}%)")
    print(f"  Inactive OK: {stats['inactive_ok']}/{n} ({stats['inactive_ok']/max(n,1)*100:.0f}%)")
    print(f"  Scores in range: {stats['in_range']}/{n} ({stats['in_range']/max(n,1)*100:.0f}%)")
    print(f"  Market context: {stats['has_market_context']}/{n} ({stats['has_market_context']/max(n,1)*100:.0f}%)")

    scored = stats["bullish"] + stats["bearish"] + stats["zero"]
    if scored > 0:
        print(f"  Direction: bull={stats['bullish']} ({stats['bullish']/scored*100:.0f}%) "
              f"bear={stats['bearish']} ({stats['bearish']/scored*100:.0f}%) "
              f"zero={stats['zero']} ({stats['zero']/scored*100:.0f}%)")

    if stats["thinking_lengths"]:
        print(f"  Think words: min={min(stats['thinking_lengths'])}, "
              f"mean={sum(stats['thinking_lengths'])//len(stats['thinking_lengths'])}, "
              f"max={max(stats['thinking_lengths'])}")

    if errors:
        print(f"  ERRORS: {len(errors)}")
        for e in errors[:5]:
            print(f"    {e}")
        if len(errors) > 5:
            print(f"    ... and {len(errors)-5} more")
    errors_all.extend(errors)

# Combined summary
print(f"\n{'='*70}")
print(f"COMBINED SUMMARY (all files)")
print(f"{'='*70}")
print(f"  Total examples: {total_all}")
scored_all = bullish_all + bearish_all + zero_all
print(f"  Directional balance:")
print(f"    Bullish: {bullish_all} ({bullish_all/max(scored_all,1)*100:.1f}%)")
print(f"    Bearish: {bearish_all} ({bearish_all/max(scored_all,1)*100:.1f}%)")
print(f"    Zero:    {zero_all} ({zero_all/max(scored_all,1)*100:.1f}%)")
print(f"  Total errors: {len(errors_all)}")
print(f"\n  {'READY FOR TRAINING' if len(errors_all) == 0 else 'FIX ERRORS BEFORE TRAINING'}")
