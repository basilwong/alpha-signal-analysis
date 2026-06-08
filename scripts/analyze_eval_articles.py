"""
Analyze evaluation articles to identify ones likely to cause JSON parse errors.

Checks for:
- Very short text (just headlines with no body)
- HTML/URL-heavy content (RSS artifacts)
- Non-English content
- Unusual characters that could break JSON strings
- Articles with no clear quantum computing relevance
"""

import json
import re
from pathlib import Path
from collections import Counter

articles_path = "data/raw/articles.jsonl"

# Load evaluation articles (index 200+)
articles = []
with open(articles_path, "r") as f:
    for i, line in enumerate(f):
        if line.strip() and i >= 200:
            article = json.loads(line)
            article["idx"] = i
            articles.append(article)

print(f"Total evaluation articles: {len(articles)}")
print("=" * 60)

# Categorize issues
issues = {
    "too_short": [],        # Less than 100 chars of actual content
    "html_heavy": [],       # Contains HTML tags or raw URLs as primary content
    "url_only": [],         # Just a title + URL, no article body
    "special_chars": [],    # Contains characters that could break JSON (unescaped quotes, etc.)
    "non_quantum": [],      # Doesn't mention any quantum-related terms
    "very_long": [],        # Extremely long (might cause token limit issues)
    "clean": [],            # Should be fine
}

quantum_terms = ["quantum", "qubit", "ionq", "rigetti", "d-wave", "qbts", "rgti", 
                 "superconducting", "trapped ion", "error correction", "entanglement",
                 "quantinuum", "honeywell quantum", "quantum computing", "quantum advantage"]

for article in articles:
    text = article.get("text", "")
    title = article.get("title", "")
    idx = article["idx"]
    
    # Check for issues
    has_issue = False
    
    # Strip the title from the beginning of text if duplicated
    clean_text = text
    if text.startswith(title):
        clean_text = text[len(title):].strip()
    
    # Check: too short
    if len(clean_text) < 100:
        issues["too_short"].append({"idx": idx, "title": title, "text_len": len(clean_text)})
        has_issue = True
        continue
    
    # Check: URL/HTML heavy
    url_count = len(re.findall(r'https?://\S+', text))
    html_count = len(re.findall(r'<[^>]+>', text))
    text_without_urls = re.sub(r'https?://\S+', '', text)
    text_without_html = re.sub(r'<[^>]+>', '', text_without_urls)
    
    if len(text_without_html.strip()) < 100:
        issues["url_only"].append({"idx": idx, "title": title, "clean_len": len(text_without_html.strip())})
        has_issue = True
        continue
    
    if html_count > 5:
        issues["html_heavy"].append({"idx": idx, "title": title, "html_tags": html_count})
        has_issue = True
        continue
    
    # Check: special characters that could break JSON
    # Look for unescaped quotes within the text that might confuse the model
    problematic_chars = text.count('"') + text.count('\\') + text.count('\t')
    if problematic_chars > 50:
        issues["special_chars"].append({"idx": idx, "title": title, "problematic_count": problematic_chars})
        # Don't skip - just flag
    
    # Check: non-quantum content
    text_lower = (text + " " + title).lower()
    has_quantum_term = any(term in text_lower for term in quantum_terms)
    if not has_quantum_term:
        issues["non_quantum"].append({"idx": idx, "title": title, "source": article.get("source", "")})
        has_issue = True
        continue
    
    # Check: very long
    if len(text) > 5000:
        issues["very_long"].append({"idx": idx, "title": title, "text_len": len(text)})
        # Don't skip - just flag
    
    if not has_issue:
        issues["clean"].append({"idx": idx, "title": title})

# Report
print("\n📊 ARTICLE QUALITY ANALYSIS")
print("=" * 60)

for category, items in issues.items():
    print(f"\n{'─' * 60}")
    print(f"  {category.upper()}: {len(items)} articles")
    if items and category != "clean":
        for item in items[:5]:  # Show first 5
            print(f"    [{item['idx']}] {item['title'][:60]}")
            for k, v in item.items():
                if k not in ("idx", "title"):
                    print(f"         {k}: {v}")
        if len(items) > 5:
            print(f"    ... and {len(items) - 5} more")

print(f"\n{'=' * 60}")
print(f"SUMMARY:")
print(f"  Clean articles (ready for prediction): {len(issues['clean'])}")
print(f"  Problematic articles (should skip or clean): {len(articles) - len(issues['clean'])}")
print(f"    - Too short: {len(issues['too_short'])}")
print(f"    - URL only: {len(issues['url_only'])}")
print(f"    - HTML heavy: {len(issues['html_heavy'])}")
print(f"    - Non-quantum: {len(issues['non_quantum'])}")
print(f"    - Very long (flagged but included): {len(issues['very_long'])}")
print(f"    - Special chars (flagged but included): {len(issues['special_chars'])}")

# Save the problematic indices for filtering
problematic_indices = set()
for category in ["too_short", "url_only", "html_heavy", "non_quantum"]:
    for item in issues[category]:
        problematic_indices.add(item["idx"])

print(f"\n{'=' * 60}")
print(f"INDICES TO SKIP (save to file): {len(problematic_indices)}")

# Save problematic indices
with open("data/eval/problematic_articles.json", "w") as f:
    json.dump({
        "problematic_indices": sorted(list(problematic_indices)),
        "categories": {k: [item["idx"] for item in v] for k, v in issues.items() if k != "clean"},
        "clean_count": len(issues["clean"]),
        "total_eval": len(articles),
    }, f, indent=2)

print(f"Saved to: data/eval/problematic_articles.json")
