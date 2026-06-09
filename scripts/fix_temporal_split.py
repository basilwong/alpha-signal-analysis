"""
Fix Temporal Split: Sort articles chronologically and create proper walk-forward split.

Problem: Articles were sorted newest-first, so training data (indices 0-199) was from
May-June 2026 while evaluation data (indices 200+) was from Aug 2024 - May 2026.
This is inverted from proper walk-forward validation.

Fix: Sort all articles oldest-first, then split by date:
- Training: Articles before cutoff date (older data)
- Evaluation: Articles after cutoff date (newer data)

Also removes outcome contamination (sentences containing explicit price movements).

Usage:
    python scripts/fix_temporal_split.py
"""

import json
import re
from pathlib import Path
from collections import Counter

INPUT_PATH = "data/raw/articles.jsonl"
TRAIN_OUTPUT = "data/raw/articles_train.jsonl"
EVAL_OUTPUT = "data/raw/articles_eval.jsonl"

# Cutoff: train on everything before 2026, evaluate on 2026 data
# This gives us ~18 months of training data and ~6 months of evaluation
TRAIN_CUTOFF_DATE = "2025-12-31"


def clean_outcome_contamination(text: str) -> str:
    """
    Remove sentences that contain explicit stock price outcomes.
    These leak future information into the training data.
    """
    # Patterns that indicate outcome contamination
    outcome_patterns = [
        r'[^.]*stock\s+(rose|fell|dropped|surged|plunged|gained|lost|jumped|tumbled|soared|crashed)\s+\d+%[^.]*\.',
        r'[^.]*shares\s+(rose|fell|dropped|surged|plunged|gained|lost|jumped|tumbled|soared|crashed)\s+\d+%[^.]*\.',
        r'[^.]*price\s+target\s+(was\s+)?(raised|lowered|cut|increased)\s+(from|to)[^.]*\.',
        r'[^.]*implying\s+\d+%\s+(upside|downside)[^.]*\.',
        r'[^.]*in\s+(pre-market|after-hours|premarket)\s+trading[^.]*\.',
    ]

    cleaned = text
    for pattern in outcome_patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    # Clean up double spaces and leading/trailing whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def main():
    # Load all articles
    articles = []
    with open(INPUT_PATH) as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    print(f"Total articles loaded: {len(articles)}")

    # Sort chronologically (oldest first)
    articles.sort(key=lambda x: x.get("date", "9999-99-99"))

    # Show date distribution
    dates = [a.get("date", "N/A") for a in articles]
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # Split by cutoff date
    train_articles = []
    eval_articles = []

    for article in articles:
        date = article.get("date", "")
        if date <= TRAIN_CUTOFF_DATE:
            train_articles.append(article)
        else:
            eval_articles.append(article)

    print(f"\nSplit at cutoff: {TRAIN_CUTOFF_DATE}")
    print(f"  Training articles: {len(train_articles)} (before cutoff)")
    print(f"  Evaluation articles: {len(eval_articles)} (after cutoff)")

    if train_articles:
        train_dates = [a.get("date", "") for a in train_articles]
        print(f"  Training date range: {min(train_dates)} to {max(train_dates)}")
    if eval_articles:
        eval_dates = [a.get("date", "") for a in eval_articles]
        print(f"  Evaluation date range: {min(eval_dates)} to {max(eval_dates)}")

    # Clean outcome contamination from training articles
    print("\nCleaning outcome contamination from training articles...")
    contaminated_count = 0
    for article in train_articles:
        original_text = article.get("text", "")
        cleaned_text = clean_outcome_contamination(original_text)
        if cleaned_text != original_text:
            contaminated_count += 1
            article["text"] = cleaned_text
            article["outcome_cleaned"] = True

    print(f"  {contaminated_count} articles had outcome statements removed")

    # Also clean eval articles (model shouldn't rely on outcome statements at inference time either)
    for article in eval_articles:
        original_text = article.get("text", "")
        cleaned_text = clean_outcome_contamination(original_text)
        if cleaned_text != original_text:
            article["text"] = cleaned_text
            article["outcome_cleaned"] = True

    # Source distribution
    print("\nSource distribution:")
    for split_name, split_articles in [("Training", train_articles), ("Evaluation", eval_articles)]:
        sources = Counter(a.get("source", "unknown") for a in split_articles)
        print(f"  {split_name}: {dict(sources)}")

    # Save
    with open(TRAIN_OUTPUT, "w") as f:
        for a in train_articles:
            f.write(json.dumps(a) + "\n")

    with open(EVAL_OUTPUT, "w") as f:
        for a in eval_articles:
            f.write(json.dumps(a) + "\n")

    print(f"\nSaved:")
    print(f"  Training: {TRAIN_OUTPUT} ({len(train_articles)} articles)")
    print(f"  Evaluation: {EVAL_OUTPUT} ({len(eval_articles)} articles)")

    # Verify no temporal leakage
    if train_articles and eval_articles:
        train_max = max(a.get("date", "") for a in train_articles)
        eval_min = min(a.get("date", "") for a in eval_articles)
        if train_max < eval_min:
            print(f"\n✓ CLEAN TEMPORAL SPLIT: Training ends {train_max}, Evaluation starts {eval_min}")
        else:
            print(f"\n✗ WARNING: Overlap detected. Training max: {train_max}, Eval min: {eval_min}")


if __name__ == "__main__":
    main()
