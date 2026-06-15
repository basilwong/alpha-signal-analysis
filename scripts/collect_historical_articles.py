"""
Historical Article Collection Script for Alpha Signal Analysis Platform

Collects quantum computing news and papers from the last 2 years (mid-2024 to present)
to build a comprehensive training dataset that can be correlated with ticker price data.

Sources:
- arXiv API (quant-ph, cs.ET categories) — up to 500 papers
- Google News RSS (financial news about quantum tickers) — multiple date ranges
- Sample hand-crafted articles for edge cases

Usage:
    python scripts/collect_historical_articles.py --output data/raw/articles.jsonl
"""

import json
import argparse
import time
from pathlib import Path
from datetime import datetime, timedelta


# Quantum computing search queries for news
NEWS_QUERIES = [
    "IonQ quantum",
    "Rigetti Computing quantum",
    "D-Wave Quantum",
    "quantum computing stock",
    "quantum computing breakthrough",
    "quantum error correction",
    "logical qubit",
    "quantum advantage",
    "Quantinuum quantum",
    "IBM quantum processor",
    "Google quantum Willow",
    "Microsoft topological qubit",
    "quantum computing funding",
    "quantum computing IPO",
    "NVIDIA quantum simulation",
]

# Tickers for Yahoo Finance news
QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON"]


def collect_arxiv_papers(max_results: int = 300) -> list:
    """Collect quantum computing papers from arXiv over the last 2 years."""
    try:
        import arxiv
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "arxiv"], check=True, capture_output=True)
        import arxiv

    print(f"Collecting up to {max_results} papers from arXiv (last 2 years)...")

    # Search for quantum computing papers with commercial/applied relevance
    queries = [
        "cat:quant-ph AND (quantum computing OR quantum error correction OR logical qubit OR fault tolerant)",
        "cat:quant-ph AND (quantum advantage OR quantum supremacy OR quantum processor)",
        "cat:cs.ET AND (quantum computing OR quantum algorithm OR quantum machine learning)",
        "cat:quant-ph AND (IonQ OR Rigetti OR D-Wave OR Quantinuum OR IBM quantum)",
        "cat:quant-ph AND (superconducting qubit OR trapped ion OR neutral atom OR topological qubit)",
    ]

    all_papers = []
    seen_ids = set()
    client = arxiv.Client()

    for query in queries:
        print(f"  Querying: {query[:60]}...")
        search = arxiv.Search(
            query=query,
            max_results=max_results // len(queries),
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        try:
            for result in client.results(search):
                # Skip if older than 2 years
                if result.published < datetime(2024, 6, 1, tzinfo=result.published.tzinfo):
                    continue

                # Skip duplicates
                if result.entry_id in seen_ids:
                    continue
                seen_ids.add(result.entry_id)

                article = {
                    "text": f"Title: {result.title}\n\nAbstract: {result.summary}",
                    "source": "arxiv",
                    "title": result.title,
                    "date": result.published.strftime("%Y-%m-%d"),
                    "url": result.entry_id,
                }
                all_papers.append(article)
        except Exception as e:
            print(f"    Warning: Query failed: {e}")

        time.sleep(3)  # Be respectful to arXiv API

    print(f"  Collected {len(all_papers)} arXiv papers")
    return all_papers


def collect_google_news_rss(max_per_query: int = 30) -> list:
    """Collect quantum computing financial news from Google News RSS."""
    try:
        import feedparser
    except ImportError:
        import subprocess
        subprocess.run(["pip", "install", "feedparser"], check=True, capture_output=True)
        import feedparser

    print(f"Collecting news from Google News RSS ({len(NEWS_QUERIES)} queries)...")

    articles = []
    seen_titles = set()

    for query in NEWS_QUERIES:
        encoded_query = query.replace(" ", "+")
        feed_url = f"https://news.google.com/rss/search?q={encoded_query}+when:2y&hl=en-US&gl=US&ceid=US:en"

        try:
            feed = feedparser.parse(feed_url)
            count = 0
            for entry in feed.entries:
                if count >= max_per_query:
                    break

                title = entry.get("title", "")
                if title in seen_titles:
                    continue
                seen_titles.add(title)

                # Combine title and summary
                text = title
                if entry.get("summary"):
                    text += f"\n\n{entry['summary']}"

                # Parse date
                date_str = entry.get("published", "")
                try:
                    from email.utils import parsedate_to_datetime
                    pub_date = parsedate_to_datetime(date_str).strftime("%Y-%m-%d")
                except:
                    pub_date = datetime.now().strftime("%Y-%m-%d")

                article = {
                    "text": text,
                    "source": "news",
                    "title": title,
                    "date": pub_date,
                    "url": entry.get("link", ""),
                }
                articles.append(article)
                count += 1

            time.sleep(2)  # Rate limiting
        except Exception as e:
            print(f"    Warning: Failed for query '{query}': {e}")

    print(f"  Collected {len(articles)} news articles")
    return articles


def collect_sample_articles() -> list:
    """Hand-crafted sample articles representing key event types."""
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
            "date": "2024-12-09",
            "url": "https://example.com/google-willow-threshold",
        },
        {
            "text": "D-Wave Quantum announced a strategic partnership with NVIDIA to integrate quantum annealing capabilities into NVIDIA's CUDA-Q platform. The partnership will allow developers to seamlessly combine classical GPU computing with D-Wave's quantum annealing processors for hybrid optimization workflows. D-Wave will receive $15 million in development funding from NVIDIA as part of the multi-year agreement.",
            "source": "news",
            "title": "D-Wave Partners with NVIDIA for Hybrid Quantum-Classical Computing",
            "date": "2025-11-05",
            "url": "https://example.com/dwave-nvidia-partnership",
        },
        {
            "text": "Microsoft announced that its topological qubit program has produced its first functional topological qubit, a milestone the company has been pursuing for over a decade. Unlike conventional qubits, topological qubits are theoretically more resistant to environmental noise, potentially requiring fewer physical qubits for error correction. However, the company acknowledged that scaling to multiple topological qubits remains a significant engineering challenge.",
            "source": "news",
            "title": "Microsoft Produces First Functional Topological Qubit",
            "date": "2025-02-19",
            "url": "https://example.com/microsoft-topological-qubit",
        },
        {
            "text": "Quantinuum, the quantum computing subsidiary of Honeywell, filed an S-1 registration statement with the SEC, signaling its intention to go public through an IPO. The filing reveals annual revenue of $43 million for fiscal year 2025, with a net loss of $180 million. The company's H-Series trapped-ion processors currently serve over 200 enterprise customers across finance, pharmaceuticals, and materials science.",
            "source": "sec_filing",
            "title": "Quantinuum Files S-1 for IPO, Reveals $43M Annual Revenue",
            "date": "2026-05-28",
            "url": "https://example.com/quantinuum-ipo-filing",
        },
        {
            "text": "IonQ's Chief Technology Officer, Jungsang Kim, announced his departure from the company to pursue academic research. Kim was instrumental in developing IonQ's trapped-ion architecture and holds several key patents. The company named Dr. Dave Bacon, formerly of Google Quantum AI, as his replacement. Analysts noted that while the departure creates short-term uncertainty, Bacon's experience with error correction could accelerate IonQ's roadmap.",
            "source": "news",
            "title": "IonQ CTO Departs, Google Quantum AI Veteran Named Replacement",
            "date": "2025-08-12",
            "url": "https://example.com/ionq-cto-departure",
        },
        {
            "text": "Infleqtion, the neutral-atom quantum computing company formerly known as ColdQuanta, completed its IPO on the NASDAQ exchange under the ticker INFQ. The company raised $280 million at a valuation of $2.1 billion. Infleqtion's approach uses arrays of neutral atoms cooled to near absolute zero, which the company claims offers advantages in scalability compared to trapped-ion and superconducting approaches.",
            "source": "news",
            "title": "Infleqtion Completes IPO at $2.1B Valuation, Trades as INFQ",
            "date": "2026-03-15",
            "url": "https://example.com/infleqtion-ipo",
        },
        {
            "text": "JPMorgan Chase published a research note downgrading IonQ from Overweight to Neutral, citing concerns about the company's path to profitability. The analyst noted that while IonQ's technology is impressive, revenue growth has not kept pace with the stock's 300% appreciation over the past year. The price target was lowered from $45 to $32, implying 15% downside from current levels.",
            "source": "news",
            "title": "JPMorgan Downgrades IonQ, Cites Valuation Concerns",
            "date": "2025-06-20",
            "url": "https://example.com/jpmorgan-ionq-downgrade",
        },
        {
            "text": "Title: Quantum error correction below the surface code threshold\n\nAbstract: We report the operation of a quantum error-correcting code below the threshold for the surface code, the most promising architecture for fault-tolerant quantum computing. Using a 72-qubit superconducting processor, we demonstrate that increasing the code distance from 3 to 5 to 7 monotonically decreases the logical error rate, achieving a logical error rate of 0.14% per round. This is the first experimental demonstration of below-threshold performance, validating decades of theoretical work on the surface code.",
            "source": "arxiv",
            "title": "Quantum error correction below the surface code threshold",
            "date": "2024-12-09",
            "url": "https://arxiv.org/abs/2408.xxxxx",
        },
    ]

    print(f"  Loaded {len(samples)} sample articles")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Collect historical quantum computing articles (last 2 years)")
    parser.add_argument(
        "--output", type=str, default="data/raw/articles.jsonl",
        help="Path to output JSONL file"
    )
    parser.add_argument(
        "--arxiv-limit", type=int, default=300,
        help="Maximum number of arXiv papers to collect"
    )
    parser.add_argument(
        "--news-per-query", type=int, default=30,
        help="Maximum news articles per search query"
    )
    parser.add_argument(
        "--skip-arxiv", action="store_true",
        help="Skip arXiv collection (faster for testing)"
    )
    parser.add_argument(
        "--skip-news", action="store_true",
        help="Skip news RSS collection"
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_articles = []

    # Always include samples
    all_articles.extend(collect_sample_articles())

    # Collect from arXiv
    if not args.skip_arxiv:
        all_articles.extend(collect_arxiv_papers(max_results=args.arxiv_limit))

    # Collect from news RSS
    if not args.skip_news:
        all_articles.extend(collect_google_news_rss(max_per_query=args.news_per_query))

    # Deduplicate by title
    seen_titles = set()
    unique_articles = []
    for article in all_articles:
        title_key = article["title"].lower().strip()
        if title_key not in seen_titles:
            seen_titles.add(title_key)
            unique_articles.append(article)

    # Sort by date (newest first)
    unique_articles.sort(key=lambda x: x.get("date", ""), reverse=True)

    # Save to JSONL
    with open(output_path, "w") as f:
        for article in unique_articles:
            f.write(json.dumps(article) + "\n")

    print(f"\n{'=' * 60}")
    print(f"Collection complete!")
    print(f"  Total unique articles: {len(unique_articles)}")
    print(f"  Date range: {unique_articles[-1].get('date', 'N/A')} to {unique_articles[0].get('date', 'N/A')}")
    print(f"  Sources: arXiv ({sum(1 for a in unique_articles if a['source'] == 'arxiv')}), "
          f"news ({sum(1 for a in unique_articles if a['source'] == 'news')}), "
          f"sec_filing ({sum(1 for a in unique_articles if a['source'] == 'sec_filing')})")
    print(f"  Saved to: {output_path}")


if __name__ == "__main__":
    main()
