"""
Fix 8: Semantic clustering for article staleness detection.
Fix 10: Event deduplication for evaluation.

Clusters articles within a sliding 3-day window by cosine similarity.
Assigns prior_coverage_count and event_id to each article.

Usage:
    python scripts/cluster_articles.py                    # Cluster and validate
    python scripts/cluster_articles.py --apply            # Apply to training data
    python scripts/cluster_articles.py --deduplicate-eval # Deduplicate eval predictions
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"
DATA_TRAINING = PROJECT_ROOT / "data" / "training"

COSINE_THRESHOLD = 0.75
WINDOW_DAYS = 3


def load_articles(filepath: Path) -> list:
    """Load articles from JSONL."""
    articles = []
    with open(filepath) as f:
        for i, line in enumerate(f):
            if line.strip():
                art = json.loads(line)
                art["_idx"] = i
                articles.append(art)
    return articles


def compute_embeddings(articles: list) -> np.ndarray:
    """Compute sentence embeddings for all articles."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("Installing sentence-transformers...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"], 
                      capture_output=True, check=True)
        from sentence_transformers import SentenceTransformer
    
    model = SentenceTransformer("all-MiniLM-L6-v2")
    
    # Use title + first 200 chars of text for embedding
    texts = []
    for art in articles:
        title = art.get("title", "")
        text = art.get("text", "")[:200]
        texts.append(f"{title}. {text}")
    
    print(f"Computing embeddings for {len(texts)} articles...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings


def extract_ticker_mentions(text: str) -> set:
    """Extract ticker mentions from text for secondary clustering check."""
    tickers = {"IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"}
    company_names = {
        "ionq": "IONQ", "rigetti": "RGTI", "d-wave": "QBTS", "dwave": "QBTS",
        "quantum computing inc": "QUBT", "quantinuum": "QNT", "honeywell": "HON",
        "microsoft": "MSFT", "google": "GOOGL", "nvidia": "NVDA", "ibm": "IBM"
    }
    
    text_upper = text.upper()
    text_lower = text.lower()
    
    found = set()
    for ticker in tickers:
        if ticker in text_upper:
            found.add(ticker)
    for name, ticker in company_names.items():
        if name in text_lower:
            found.add(ticker)
    
    return found


def cluster_articles(articles: list, embeddings: np.ndarray) -> list:
    """
    Cluster articles within sliding windows.
    Returns articles with prior_coverage_count and event_id added.
    """
    from numpy.linalg import norm
    
    # Sort by date
    dated_articles = [(i, art) for i, art in enumerate(articles) if art.get("date")]
    dated_articles.sort(key=lambda x: x[1]["date"])
    
    # Assign event_ids
    event_counter = 0
    event_assignments = {}  # article_idx -> event_id
    coverage_counts = {}    # article_idx -> prior_coverage_count
    
    for pos, (idx, art) in enumerate(dated_articles):
        if idx in event_assignments:
            continue
        
        # This article starts a new event
        event_counter += 1
        event_id = f"event_{event_counter:04d}"
        event_assignments[idx] = event_id
        coverage_counts[idx] = 0
        
        art_date = art["date"]
        art_tickers = extract_ticker_mentions(art.get("title", "") + " " + art.get("text", ""))
        art_embedding = embeddings[idx]
        art_norm = norm(art_embedding)
        
        if art_norm == 0:
            continue
        
        # Look forward within window
        coverage_order = 1
        for future_pos in range(pos + 1, len(dated_articles)):
            future_idx, future_art = dated_articles[future_pos]
            
            if future_idx in event_assignments:
                continue
            
            # Check date window
            try:
                date_diff = (datetime.strptime(future_art["date"], "%Y-%m-%d") - 
                           datetime.strptime(art_date, "%Y-%m-%d")).days
            except (ValueError, TypeError):
                continue
            
            if date_diff > WINDOW_DAYS:
                break  # Past the window
            
            # Check cosine similarity
            future_embedding = embeddings[future_idx]
            future_norm = norm(future_embedding)
            
            if future_norm == 0:
                continue
            
            cosine_sim = np.dot(art_embedding, future_embedding) / (art_norm * future_norm)
            
            if cosine_sim >= COSINE_THRESHOLD:
                # Secondary check: require overlapping ticker mentions
                future_tickers = extract_ticker_mentions(
                    future_art.get("title", "") + " " + future_art.get("text", ""))
                
                if art_tickers and future_tickers:
                    overlap = art_tickers.intersection(future_tickers)
                    if not overlap:
                        continue  # Different companies, don't cluster
                
                # Same event cluster
                event_assignments[future_idx] = event_id
                coverage_counts[future_idx] = coverage_order
                coverage_order += 1
    
    # Apply to articles
    for i, art in enumerate(articles):
        art["event_id"] = event_assignments.get(i, f"event_unique_{i:04d}")
        art["prior_coverage_count"] = coverage_counts.get(i, 0)
    
    return articles


def validate_clusters(articles: list, sample_size: int = 30):
    """Print sample clusters for manual validation."""
    # Find multi-article clusters
    clusters = defaultdict(list)
    for art in articles:
        eid = art.get("event_id", "")
        if not eid.startswith("event_unique"):
            clusters[eid].append(art)
    
    multi_clusters = {k: v for k, v in clusters.items() if len(v) > 1}
    
    print(f"\n{'='*60}")
    print(f"CLUSTERING VALIDATION")
    print(f"{'='*60}")
    print(f"Total articles: {len(articles)}")
    print(f"Unique events: {len(clusters)}")
    print(f"Multi-article clusters: {len(multi_clusters)}")
    print(f"Articles in multi-clusters: {sum(len(v) for v in multi_clusters.values())}")
    print()
    
    # Sample clusters for review
    import random
    sample_keys = random.sample(list(multi_clusters.keys()), min(sample_size, len(multi_clusters)))
    
    print(f"--- Sample {len(sample_keys)} clusters for manual review ---\n")
    for eid in sample_keys:
        arts = multi_clusters[eid]
        print(f"Cluster {eid} ({len(arts)} articles):")
        for art in arts:
            print(f"  [{art['date']}] {art.get('title', 'N/A')[:70]}")
            tickers = extract_ticker_mentions(art.get("title", "") + " " + art.get("text", ""))
            if tickers:
                print(f"           Tickers: {tickers}")
        print()
    
    return multi_clusters


def deduplicate_eval(articles_file: Path, predictions_file: Path, output_file: Path):
    """Fix 10: Deduplicate eval predictions by event_id."""
    # Load clustered articles
    articles = load_articles(articles_file)
    
    # Build event_id lookup by article index
    event_lookup = {}
    for art in articles:
        event_lookup[art["_idx"]] = art.get("event_id", f"unique_{art['_idx']}")
    
    # Load predictions
    predictions = []
    with open(predictions_file) as f:
        for line in f:
            if line.strip():
                predictions.append(json.loads(line))
    
    # Group by event_id, keep first (earliest)
    event_first = {}
    for pred in predictions:
        idx = pred.get("article_idx", -1)
        eid = event_lookup.get(idx, f"unique_{idx}")
        pred["event_id"] = eid
        
        if eid not in event_first:
            event_first[eid] = pred
        else:
            # Keep the one with earlier date
            existing_date = event_first[eid].get("date", "9999")
            new_date = pred.get("date", "9999")
            if new_date < existing_date:
                event_first[eid] = pred
    
    # Write deduplicated
    deduped = list(event_first.values())
    with open(output_file, "w") as f:
        for pred in deduped:
            f.write(json.dumps(pred) + "\n")
    
    print(f"Deduplication complete:")
    print(f"  Original: {len(predictions)} predictions")
    print(f"  Events: {len(event_first)}")
    print(f"  Deduplicated: {len(deduped)} predictions")
    print(f"  Saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Apply clustering to training data")
    parser.add_argument("--deduplicate-eval", action="store_true", help="Deduplicate eval predictions")
    args = parser.parse_args()
    
    articles_file = DATA_RAW / "articles.jsonl"
    if not articles_file.exists():
        print(f"ERROR: {articles_file} not found")
        sys.exit(1)
    
    # Load and cluster
    articles = load_articles(articles_file)
    embeddings = compute_embeddings(articles)
    articles = cluster_articles(articles, embeddings)
    
    # Save clustered articles
    clustered_file = DATA_RAW / "articles_clustered.jsonl"
    with open(clustered_file, "w") as f:
        for art in articles:
            # Remove internal _idx field
            art_copy = {k: v for k, v in art.items() if k != "_idx"}
            f.write(json.dumps(art_copy) + "\n")
    print(f"Saved clustered articles to: {clustered_file}")
    
    # Validate
    multi_clusters = validate_clusters(articles)
    
    if args.deduplicate_eval:
        eval_predictions = DATA_EVAL / "predictions_manus_teacher.jsonl"
        if eval_predictions.exists():
            deduplicate_eval(
                clustered_file,
                eval_predictions,
                DATA_EVAL / "predictions_deduplicated.jsonl"
            )
    
    if args.apply:
        print("\n[APPLY] Would apply prior_coverage_count to training data.")
        print("Skipping for now — run with --apply after manual validation of clusters above.")


if __name__ == "__main__":
    main()
