import json
from collections import Counter

failures = []
successes = []

with open("data/eval/predictions_openreasoning7b_v4.jsonl") as f:
    for line in f:
        if line.strip():
            r = json.loads(line)
            if r.get("status") == "error":
                failures.append(r)
            elif r.get("status") == "success":
                successes.append(r)

print(f"Total: {len(successes) + len(failures) + len([1 for line in open('data/eval/predictions_openreasoning7b_v4.jsonl') if 'skipped' in line])}")
print(f"Success: {len(successes)}")
print(f"Errors: {len(failures)}")
print()

# Categorize errors
error_types = Counter()
for f_item in failures:
    err = f_item.get("error", "")
    if "No JSON found" in err:
        error_types["No JSON found"] += 1
    elif "Expecting" in err:
        error_types["JSON parse error"] += 1
    elif "Unterminated" in err:
        error_types["Unterminated string"] += 1
    else:
        error_types[err[:80]] += 1

print("Error categories:")
for err_type, count in error_types.most_common():
    print(f"  {count}x: {err_type}")

print("\n" + "=" * 60)
print("Sample failures (first 5):")
print("=" * 60)
for f_item in failures[:5]:
    print(f"\n  Article {f_item.get('article_idx')}: {f_item.get('title', '')[:60]}")
    print(f"  Source: {f_item.get('source')}")
    print(f"  Error: {f_item.get('error', '')[:200]}")

print("\n" + "=" * 60)
print("Source distribution of failures:")
print("=" * 60)
fail_sources = Counter(f_item.get("source") for f_item in failures)
success_sources = Counter(s.get("source") for s in successes)
for source in sorted(set(list(fail_sources.keys()) + list(success_sources.keys()))):
    total = fail_sources.get(source, 0) + success_sources.get(source, 0)
    fail_rate = fail_sources.get(source, 0) / max(total, 1) * 100
    print(f"  {source}: {fail_sources.get(source, 0)}/{total} failed ({fail_rate:.0f}%)")
