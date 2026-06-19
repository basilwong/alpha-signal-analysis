"""
Data source connectors for ingesting news and market data.

Supports:
1. RSS feeds (quantum computing news)
2. Manual article submission via API
3. Scheduled ingestion via cron
"""
import httpx
import json
from datetime import datetime
from .config import QUANTUM_TICKERS


# RSS feed sources for quantum computing news
RSS_SOURCES = [
    {"name": "Google News - Quantum Computing", "url": "https://news.google.com/rss/search?q=quantum+computing+stocks"},
    {"name": "Google News - IonQ", "url": "https://news.google.com/rss/search?q=IonQ"},
    {"name": "Google News - Rigetti", "url": "https://news.google.com/rss/search?q=Rigetti+Computing"},
]


async def fetch_rss_articles(source_url: str, limit: int = 5) -> list:
    """Fetch articles from an RSS feed."""
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.get(source_url)
            if resp.status_code != 200:
                return []
            # Simple XML parsing for RSS (no external dependency)
            text = resp.text
            articles = []
            items = text.split("<item>")[1:]  # Skip header
            for item in items[:limit]:
                title = _extract_tag(item, "title")
                link = _extract_tag(item, "link")
                pub_date = _extract_tag(item, "pubDate")
                description = _extract_tag(item, "description")
                if title:
                    articles.append({
                        "title": title,
                        "link": link,
                        "date": pub_date,
                        "description": description,
                        "source": source_url
                    })
            return articles
    except Exception as e:
        return []


def _extract_tag(xml_text: str, tag: str) -> str:
    """Extract content from an XML tag."""
    start = xml_text.find(f"<{tag}>")
    end = xml_text.find(f"</{tag}>")
    if start == -1 or end == -1:
        # Try CDATA
        start = xml_text.find(f"<{tag}><![CDATA[")
        if start != -1:
            start += len(f"<{tag}><![CDATA[")
            end = xml_text.find(f"]]></{tag}>")
            return xml_text[start:end].strip() if end != -1 else ""
        return ""
    start += len(f"<{tag}>")
    return xml_text[start:end].strip()


async def ingest_all_sources(memory_store, retriever, generate_signal_fn) -> dict:
    """Run a full ingestion cycle across all RSS sources."""
    total_articles = 0
    total_signals = 0

    for source in RSS_SOURCES:
        articles = await fetch_rss_articles(source["url"])
        total_articles += len(articles)

        for article in articles:
            # Combine title and description for analysis
            text = f"{article['title']}. {article.get('description', '')}"

            # Check if we already processed this article (dedup by title)
            existing = memory_store.conn.execute(
                "SELECT id FROM signal_history WHERE article_title = ?",
                (article['title'][:100],)
            ).fetchone()
            if existing:
                continue

            # Generate signal with memory context
            memory_context = retriever.retrieve_context(text)
            try:
                result = generate_signal_fn(text, "news", memory_context, enable_thinking=False)
                content = result["content"]
                s = content.find("{")
                e = content.rfind("}") + 1
                if s != -1:
                    signal = json.loads(content[s:e])
                    signal_vector = signal.get("signal_vector", {})
                    if signal_vector:
                        memory_store.store_signal(
                            article_date=article.get("date", datetime.utcnow().strftime("%Y-%m-%d")),
                            article_title=article["title"][:100],
                            article_source=source["name"],
                            signal_vector=signal_vector,
                            reasoning=signal.get("chain_of_thought", ""),
                            model_used="qwen3-max"
                        )
                        total_signals += 1
            except Exception as e:
                continue

    return {"articles_fetched": total_articles, "signals_generated": total_signals}
