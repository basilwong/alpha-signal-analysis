"""
Analyze failed predictions to understand what's causing JSON parse errors.
"""

import json

# Load predictions
predictions = []
with open("data/eval/predictions_v2.jsonl") as f:
    for line in f:
        if line.strip():
            predictions.append(json.loads(line))

# Load original articles
articles = {}
with open("data/raw/articles.jsonl") as f:
    for i, line in enumerate(f):
        if line.strip():
            articles[i] = json.loads(line)

# Separate successes and failures
successes = [p for p in predictions if p["status"] == "success"]
failures = [p for p in predictions if p["status"] == "error"]

print(f"Total predictions: {len(predictions)}")
print(f"Successes: {len(successes)}")
print(f"Failures: {len(failures)}")
print(f"\n{'='*80}")
print("FAILED ARTICLES ANALYSIS")
print(f"{'='*80}")

for f_pred in failures:
    idx = f_pred["article_idx"]
    article = articles.get(idx, {})
    text = article.get("text", "")
    
    print(f"\n{'─'*80}")
    print(f"Article #{idx}: {f_pred.get('title', '')[:70]}")
    print(f"  Source: {f_pred.get('source', 'unknown')}")
    print(f"  Date: {f_pred.get('date', 'unknown')}")
    print(f"  Error: {f_pred.get('error', 'N/A')}")
    print(f"  Text length: {len(text)} chars")
    print(f"  Input tokens: {f_pred.get('input_tokens', 'N/A')}")
    
    # Analyze text characteristics
    import re
    url_count = len(re.findall(r'https?://\S+', text))
    html_count = len(re.findall(r'<[^>]+>', text))
    newline_count = text.count('\n')
    quote_count = text.count('"')
    
    print(f"  URLs in text: {url_count}")
    print(f"  HTML tags: {html_count}")
    print(f"  Newlines: {newline_count}")
    print(f"  Double quotes: {quote_count}")
    print(f"  First 200 chars: {text[:200]}")
    print(f"  Last 100 chars: {text[-100:]}")

# Summary statistics
print(f"\n{'='*80}")
print("COMPARISON: Success vs Failure article characteristics")
print(f"{'='*80}")

success_indices = [p["article_idx"] for p in successes]
failure_indices = [p["article_idx"] for p in failures]

success_lengths = [len(articles[idx].get("text", "")) for idx in success_indices if idx in articles]
failure_lengths = [len(articles[idx].get("text", "")) for idx in failure_indices if idx in articles]

success_tokens = [p.get("input_tokens", 0) for p in successes if p.get("input_tokens")]
failure_tokens = [p.get("input_tokens", 0) for p in failures if p.get("input_tokens")]

print(f"\nText length (chars):")
print(f"  Success avg: {sum(success_lengths)/len(success_lengths):.0f} | min: {min(success_lengths)} | max: {max(success_lengths)}")
print(f"  Failure avg: {sum(failure_lengths)/len(failure_lengths):.0f} | min: {min(failure_lengths)} | max: {max(failure_lengths)}")

if success_tokens:
    print(f"\nInput tokens:")
    print(f"  Success avg: {sum(success_tokens)/len(success_tokens):.0f} | min: {min(success_tokens)} | max: {max(success_tokens)}")
if failure_tokens:
    print(f"  Failure avg: {sum(failure_tokens)/len(failure_tokens):.0f} | min: {min(failure_tokens)} | max: {max(failure_tokens)}")

# Check for common patterns in failures
print(f"\nFailure error patterns:")
error_patterns = {}
for f_pred in failures:
    error = f_pred.get("error", "")
    # Extract the key part of the error
    if "Expecting value" in error:
        error_patterns["Expecting value (missing/invalid JSON value)"] = error_patterns.get("Expecting value (missing/invalid JSON value)", 0) + 1
    elif "Expecting ',' delimiter" in error:
        error_patterns["Missing comma delimiter"] = error_patterns.get("Missing comma delimiter", 0) + 1
    else:
        error_patterns[error[:50]] = error_patterns.get(error[:50], 0) + 1

for pattern, count in sorted(error_patterns.items(), key=lambda x: -x[1]):
    print(f"  {pattern}: {count}")
