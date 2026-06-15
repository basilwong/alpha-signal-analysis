"""
Article Collection Script for Alpha Signal Analysis Platform

Collects quantum computing news articles from multiple sources
and saves them in the format expected by the training data generation pipeline.

Sources:
- arXiv API (quant-ph category)
- Yahoo Finance (quantum computing tickers)
- RSS feeds (financial news)

Output format (JSONL):
{"text": "...", "source": "arxiv|news|sec_filing", "title": "...", "date": "...", "url": "..."}

Usage:
    python scripts/collect_articles.py --output data/raw/articles.jsonl --limit 200
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta

# Quantum computing search terms
QUANTUM_KEYWORDS = [
    "quantum computing",
    "quantum error correction",
    "logical qubit",
    "IonQ",
    "Rigetti",
    "D-Wave",
    "quantum advantage",
    "superconducting qubit",
    "trapped ion quantum",
    "quantum processor",
    "fault tolerant quantum",
    "Quantinuum",
]

# Tickers to monitor
QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "INFQ", "IBM", "GOOGL", "MSFT", "HON"]


def collect_arxiv_papers(max_results: int = 50) -> list:
    """Collect recent quantum computing papers from arXiv."""
    try:
        import arxiv
    except ImportError:
        print("Installing arxiv package...")
        import subprocess
        subprocess.run(["pip", "install", "arxiv"], check=True)
        import arxiv

    print(f"Collecting up to {max_results} papers from arXiv (quant-ph)...")

    search = arxiv.Search(
        query="cat:quant-ph AND (quantum computing OR quantum error correction OR logical qubit OR fault tolerant)",
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    articles = []
    client = arxiv.Client()

    for result in client.results(search):
        article = {
            "text": f"Title: {result.title}\n\nAbstract: {result.summary}",
            "source": "arxiv",
            "title": result.title,
            "date": result.published.strftime("%Y-%m-%d"),
            "url": result.entry_id,
            "authors": [a.name for a in result.authors[:5]],
        }
        articles.append(article)

    print(f"  Collected {len(articles)} arXiv papers")
    return articles


def collect_rss_news(max_results: int = 50) -> list:
    """Collect quantum computing news from RSS feeds."""
    try:
        import feedparser
    except ImportError:
        print("Installing feedparser package...")
        import subprocess
        subprocess.run(["pip", "install", "feedparser"], check=True)
        import feedparser

    # Financial and tech news RSS feeds
    rss_feeds = [
        "https://news.google.com/rss/search?q=quantum+computing+stock&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=IonQ+OR+Rigetti+OR+D-Wave+quantum&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=quantum+computing+breakthrough&hl=en-US&gl=US&ceid=US:en",
    ]

    print(f"Collecting news from {len(rss_feeds)} RSS feeds...")
    articles = []

    for feed_url in rss_feeds:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:max_results // len(rss_feeds)]:
                # Combine title and summary for the text
                text = entry.get("title", "")
                if entry.get("summary"):
                    text += f"\n\n{entry['summary']}"

                article = {
                    "text": text,
                    "source": "news",
                    "title": entry.get("title", "Untitled"),
                    "date": entry.get("published", datetime.now().strftime("%Y-%m-%d")),
                    "url": entry.get("link", ""),
                }
                articles.append(article)
            time.sleep(1)  # Be respectful
        except Exception as e:
            print(f"  Warning: Failed to fetch feed: {e}")

    print(f"  Collected {len(articles)} news articles")
    return articles


def collect_sample_articles() -> list:
    """
    Provide a set of hand-crafted sample articles for initial testing.
    These represent the types of content the model will encounter.
    """
    samples = [
        {
            "text": "IonQ announced today that its latest trapped-ion quantum processor has achieved 35 algorithmic qubits, representing a significant improvement in the number of high-fidelity qubits available for running quantum algorithms. The company stated that this milestone brings them closer to achieving quantum advantage for commercially relevant problems in optimization and machine learning. IonQ's stock rose 8% in pre-market trading following the announcement.",
            "source": "news",
            "title": "IonQ Achieves 35 Algorithmic Qubits on Latest Processor",
            "date": "2026-05-15",
            "url": "https://example.com/ionq-35-qubits",
        },
        {
            "text": "Rigetti Computing reported Q1 2026 revenue of $4.2 million, missing analyst expectations of $5.1 million. The company cited delays in its 84-qubit Ankaa-3 system deployment as the primary factor. CEO Subodh Kulkarni noted that while hardware development is on track, customer adoption has been slower than anticipated. The company maintains its full-year guidance of $22-25 million in revenue.",
            "source": "news",
            "title": "Rigetti Q1 Revenue Misses Estimates, Cites Deployment Delays",
            "date": "2026-05-10",
            "url": "https://example.com/rigetti-q1-miss",
        },
        {
            "text": "Title: Demonstration of fault-tolerant quantum computation with 48 logical qubits\n\nAbstract: We demonstrate fault-tolerant quantum computation using 48 logical qubits encoded in a surface code architecture. Our system achieves a logical error rate of 10^-6 per round of error correction, representing a 100x improvement over the physical error rate. This result establishes a clear path toward scalable, fault-tolerant quantum computing and suggests that commercially relevant quantum advantage may be achievable within 3-5 years for specific optimization problems.",
            "source": "arxiv",
            "title": "Demonstration of fault-tolerant quantum computation with 48 logical qubits",
            "date": "2026-05-20",
            "url": "https://arxiv.org/abs/2026.xxxxx",
        },
        {
            "text": "The U.S. Department of Energy announced $500 million in new funding for quantum computing research, with grants distributed across five national laboratories and twelve university research groups. The funding specifically targets error correction and fault tolerance research, areas considered critical for achieving practical quantum computing. Companies including IonQ, Rigetti, and IBM are named as industry partners on several of the funded projects.",
            "source": "news",
            "title": "DOE Announces $500M in Quantum Computing Research Funding",
            "date": "2026-05-25",
            "url": "https://example.com/doe-quantum-funding",
        },
        {
            "text": "Google's Quantum AI team published results showing their Willow processor achieved quantum error correction below the surface code threshold for the first time at scale. The 105-qubit processor demonstrated that adding more qubits actually reduces the overall error rate, a key requirement for building large-scale fault-tolerant quantum computers. This result was previously considered a major unsolved challenge in the field.",
            "source": "news",
            "title": "Google Willow Processor Breaks Error Correction Threshold",
            "date": "2026-04-30",
            "url": "https://example.com/google-willow-threshold",
        },
        {
            "text": "D-Wave Quantum announced a strategic partnership with NVIDIA to integrate quantum annealing capabilities into NVIDIA's CUDA-Q platform. The partnership will allow developers to seamlessly combine classical GPU computing with D-Wave's quantum annealing processors for hybrid optimization workflows. D-Wave will receive $15 million in development funding from NVIDIA as part of the multi-year agreement.",
            "source": "news",
            "title": "D-Wave Partners with NVIDIA for Hybrid Quantum-Classical Computing",
            "date": "2026-05-05",
            "url": "https://example.com/dwave-nvidia-partnership",
        },
        {
            "text": "Microsoft announced that its topological qubit program has produced its first functional topological qubit, a milestone the company has been pursuing for over a decade. Unlike conventional qubits, topological qubits are theoretically more resistant to environmental noise, potentially requiring fewer physical qubits for error correction. However, the company acknowledged that scaling to multiple topological qubits remains a significant engineering challenge.",
            "source": "news",
            "title": "Microsoft Produces First Functional Topological Qubit",
            "date": "2026-06-01",
            "url": "https://example.com/microsoft-topological-qubit",
        },
        {
            "text": "Quantinuum, the quantum computing subsidiary of Honeywell, filed an S-1 registration statement with the SEC, signaling its intention to go public through an IPO. The filing reveals annual revenue of $43 million for fiscal year 2025, with a net loss of $180 million. The company's H-Series trapped-ion processors currently serve over 200 enterprise customers across finance, pharmaceuticals, and materials science.",
            "source": "sec_filing",
            "title": "Quantinuum Files S-1 for IPO, Reveals $43M Annual Revenue",
            "date": "2026-05-28",
            "url": "https://example.com/quantinuum-ipo-filing",
        },
    ]

    print(f"  Loaded {len(samples)} sample articles for testing")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Collect quantum computing articles")
    parser.add_argument(
        "--output", type=str, default="data/raw/articles.jsonl",
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--limit", type=int, default=200,
        help="Maximum number of articles to collect per source"
    )
    parser.add_argument(
        "--sources", type=str, nargs="+", default=["samples", "arxiv", "rss"],
        choices=["samples", "arxiv", "rss"],
        help="Which sources to collect from"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_articles = []

    if "samples" in args.sources:
        all_articles.extend(collect_sample_articles())

    if "arxiv" in args.sources:
        all_articles.extend(collect_arxiv_papers(max_results=args.limit))

    if "rss" in args.sources:
        all_articles.extend(collect_rss_news(max_results=args.limit))

    # Deduplicate by title
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        if article["title"] not in seen_titles:
            seen_titles.add(article["title"])
            unique_articles.append(article)

    # Save to JSONL
    with open(output_path, "w") as f:
        for article in unique_articles:
            f.write(json.dumps(article) + "\n")

    print(f"\nTotal: {len(unique_articles)} unique articles saved to {output_path}")


if __name__ == "__main__":
    main()
