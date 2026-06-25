"""
Data source connectors for the Memory Agent.
Implements 7 data streams for quantum computing sector intelligence:
1. Patent filings (USPTO PatentsView API)
2. Insider transactions (SEC EDGAR / Yahoo Finance)
3. Job postings (company career pages)
4. GitHub/code activity (GitHub REST API)
5. Conference presentations (arXiv proxy)
6. arXiv papers (arXiv API)
7. News articles (RSS feeds)

Each connector returns a list of standardized MemoryEvent objects.
"""
import httpx
import json
import time
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import List, Optional

# Quantum computing companies and their identifiers
COMPANY_MAP = {
    "IONQ": {"name": "IonQ", "sec_cik": "0001812364", "github_orgs": ["ionq-samples"], "patent_assignee": "IonQ"},
    "RGTI": {"name": "Rigetti", "sec_cik": "0001838359", "github_orgs": ["rigetti"], "patent_assignee": "Rigetti"},
    "QBTS": {"name": "D-Wave", "sec_cik": "0001907982", "github_orgs": ["dwavesystems"], "patent_assignee": "D-Wave"},
    "QUBT": {"name": "Quantum Computing Inc", "sec_cik": "0001758009", "github_orgs": [], "patent_assignee": "Quantum Computing"},
    "QNT": {"name": "Quantinuum", "sec_cik": "", "github_orgs": ["CQCL"], "patent_assignee": "Quantinuum"},
    "IBM": {"name": "IBM", "sec_cik": "0000051143", "github_orgs": ["Qiskit"], "patent_assignee": "International Business Machines"},
    "GOOGL": {"name": "Google", "sec_cik": "0001652044", "github_orgs": ["quantumlib"], "patent_assignee": "Google"},
    "MSFT": {"name": "Microsoft", "sec_cik": "0000789019", "github_orgs": ["microsoft"], "patent_assignee": "Microsoft"},
    "HON": {"name": "Honeywell", "sec_cik": "0000773840", "github_orgs": [], "patent_assignee": "Honeywell"},
    "NVDA": {"name": "NVIDIA", "sec_cik": "0001045810", "github_orgs": ["NVIDIA"], "patent_assignee": "NVIDIA"},
}

QUANTUM_GITHUB_REPOS = [
    "Qiskit/qiskit",          # IBM
    "quantumlib/Cirq",        # Google
    "PennyLaneAI/pennylane",  # Xanadu
    "CQCL/tket",              # Quantinuum
    "rigetti/pyquil",         # Rigetti
    "dwavesystems/dwave-ocean-sdk",  # D-Wave
]


@dataclass
class MemoryEvent:
    """Standardized event from any data source."""
    source_type: str       # patent, insider, job, github, conference, arxiv, news
    ticker: str            # Primary affected ticker
    date: str              # ISO date string
    title: str             # Short description
    content: str           # Full content/details
    significance: str      # high, medium, low
    url: Optional[str] = None
    metadata: Optional[dict] = None


class PatentConnector:
    """USPTO PatentsView API connector for quantum computing patents."""
    BASE_URL = "https://api.patentsview.org/patents/query"
    
    def fetch(self, days_back: int = 90) -> List[MemoryEvent]:
        events = []
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        for ticker, info in COMPANY_MAP.items():
            if not info["patent_assignee"]:
                continue
            try:
                query = {
                    "q": {"_and": [
                        {"_gte": {"patent_date": since_date}},
                        {"_contains": {"assignee_organization": info["patent_assignee"]}},
                        {"_or": [
                            {"_text_any": {"patent_title": "quantum"}},
                            {"_text_any": {"patent_abstract": "quantum qubit"}}
                        ]}
                    ]},
                    "f": ["patent_number", "patent_title", "patent_date", "patent_abstract"],
                    "o": {"per_page": 25}
                }
                resp = httpx.post(self.BASE_URL, json=query, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    for patent in data.get("patents", []):
                        events.append(MemoryEvent(
                            source_type="patent",
                            ticker=ticker,
                            date=patent.get("patent_date", ""),
                            title=f"Patent: {patent.get('patent_title', 'Unknown')}",
                            content=patent.get("patent_abstract", "")[:500],
                            significance="medium",
                            url=f"https://patents.google.com/patent/US{patent.get('patent_number', '')}",
                            metadata={"patent_number": patent.get("patent_number")}
                        ))
                time.sleep(1)
            except Exception as e:
                print(f"  Patent fetch error for {ticker}: {e}")
        return events


class InsiderTransactionConnector:
    """SEC EDGAR insider transaction connector."""
    BASE_URL = "https://efts.sec.gov/LATEST/search-index"
    
    def fetch(self, days_back: int = 90) -> List[MemoryEvent]:
        events = []
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        for ticker, info in COMPANY_MAP.items():
            cik = info.get("sec_cik", "")
            if not cik:
                continue
            try:
                # Use SEC full-text search for Form 4 filings
                url = f"https://efts.sec.gov/LATEST/search-index?q=%22{info['name']}%22&forms=4&dateRange=custom&startdt={since_date}"
                headers = {"User-Agent": "AlphaSignalAnalysis research@example.com"}
                resp = httpx.get(url, headers=headers, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    for hit in data.get("hits", {}).get("hits", [])[:10]:
                        source = hit.get("_source", {})
                        filing_date = source.get("file_date", "")
                        display_name = source.get("display_names", ["Unknown"])[0] if source.get("display_names") else "Unknown"
                        events.append(MemoryEvent(
                            source_type="insider",
                            ticker=ticker,
                            date=filing_date,
                            title=f"Insider transaction: {display_name} at {info['name']}",
                            content=f"Form 4 filing by {display_name}. Filed {filing_date}.",
                            significance="medium",
                            url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik}&type=4",
                            metadata={"filer": display_name, "form_type": "4"}
                        ))
                time.sleep(1)
            except Exception as e:
                print(f"  Insider fetch error for {ticker}: {e}")
        return events


class JobPostingConnector:
    """Job posting tracker for quantum computing companies.
    Since no free API exists, we track via news mentions and known career pages."""
    
    # Known career page patterns
    CAREER_URLS = {
        "IONQ": "https://ionq.com/careers",
        "RGTI": "https://www.rigetti.com/careers",
        "QBTS": "https://www.dwavesys.com/careers",
        "QNT": "https://www.quantinuum.com/careers",
    }
    
    def fetch(self, days_back: int = 90) -> List[MemoryEvent]:
        """Fetch job-related signals from news and career page checks."""
        events = []
        # Use Google News RSS to find hiring/layoff news
        try:
            import feedparser
            for ticker, info in COMPANY_MAP.items():
                if ticker not in self.CAREER_URLS:
                    continue
                query = f"{info['name']} quantum hiring OR jobs OR layoffs OR workforce"
                feed_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:5]:
                    pub_date = entry.get("published", "")
                    # Parse date
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(pub_date)
                        date_str = dt.strftime("%Y-%m-%d")
                    except:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                    
                    title = entry.get("title", "")
                    if any(kw in title.lower() for kw in ["hiring", "job", "layoff", "workforce", "engineer", "recruit"]):
                        significance = "high" if "layoff" in title.lower() else "medium"
                        events.append(MemoryEvent(
                            source_type="job",
                            ticker=ticker,
                            date=date_str,
                            title=f"Hiring signal: {title[:80]}",
                            content=title,
                            significance=significance,
                            url=entry.get("link", ""),
                        ))
                time.sleep(0.5)
        except ImportError:
            pass
        return events


class GitHubActivityConnector:
    """GitHub repository activity tracker for quantum computing projects."""
    BASE_URL = "https://api.github.com"
    
    def fetch(self, days_back: int = 30) -> List[MemoryEvent]:
        events = []
        since_date = (datetime.now() - timedelta(days=days_back)).isoformat() + "Z"
        
        for repo in QUANTUM_GITHUB_REPOS:
            try:
                # Get recent commits count
                url = f"{self.BASE_URL}/repos/{repo}/commits?since={since_date}&per_page=1"
                headers = {"Accept": "application/vnd.github.v3+json"}
                resp = httpx.get(url, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    # Get total from Link header
                    link_header = resp.headers.get("Link", "")
                    commits_count = 1
                    if "last" in link_header:
                        match = re.search(r'page=(\d+)>; rel="last"', link_header)
                        if match:
                            commits_count = int(match.group(1))
                    
                    # Get latest release
                    release_url = f"{self.BASE_URL}/repos/{repo}/releases/latest"
                    release_resp = httpx.get(release_url, headers=headers, timeout=10)
                    latest_release = ""
                    if release_resp.status_code == 200:
                        release_data = release_resp.json()
                        latest_release = f"Latest release: {release_data.get('tag_name', 'N/A')} ({release_data.get('published_at', '')[:10]})"
                    
                    # Map repo to ticker
                    ticker = self._repo_to_ticker(repo)
                    
                    significance = "high" if commits_count > 50 else "medium" if commits_count > 10 else "low"
                    events.append(MemoryEvent(
                        source_type="github",
                        ticker=ticker,
                        date=datetime.now().strftime("%Y-%m-%d"),
                        title=f"GitHub: {repo} - {commits_count} commits in {days_back}d",
                        content=f"Repository {repo} had {commits_count} commits in the last {days_back} days. {latest_release}",
                        significance=significance,
                        url=f"https://github.com/{repo}",
                        metadata={"repo": repo, "commits": commits_count}
                    ))
                time.sleep(1)
            except Exception as e:
                print(f"  GitHub fetch error for {repo}: {e}")
        return events
    
    def _repo_to_ticker(self, repo: str) -> str:
        mapping = {
            "Qiskit": "IBM", "quantumlib": "GOOGL", "PennyLaneAI": "QNT",
            "CQCL": "QNT", "rigetti": "RGTI", "dwavesystems": "QBTS",
            "microsoft": "MSFT", "NVIDIA": "NVDA"
        }
        org = repo.split("/")[0]
        return mapping.get(org, "")


class ConferenceConnector:
    """Conference presentation tracker. Uses arXiv as a proxy for conference papers."""
    
    CONFERENCES = ["QIP", "APS March Meeting", "IEEE Quantum Week", "Q2B"]
    
    def fetch(self, days_back: int = 90) -> List[MemoryEvent]:
        """Search arXiv for papers mentioning major quantum conferences."""
        events = []
        try:
            since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y%m%d")
            for conf in self.CONFERENCES:
                query = f"all:{conf.replace(' ', '+')}+AND+cat:quant-ph"
                url = f"http://export.arxiv.org/api/query?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results=5"
                resp = httpx.get(url, timeout=15)
                if resp.status_code == 200:
                    root = ET.fromstring(resp.content)
                    ns = {"atom": "http://www.w3.org/2005/Atom"}
                    for entry in root.findall("atom:entry", ns):
                        title = entry.find("atom:title", ns).text.strip() if entry.find("atom:title", ns) is not None else ""
                        summary = entry.find("atom:summary", ns).text.strip()[:300] if entry.find("atom:summary", ns) is not None else ""
                        published = entry.find("atom:published", ns).text[:10] if entry.find("atom:published", ns) is not None else ""
                        link = entry.find("atom:id", ns).text if entry.find("atom:id", ns) is not None else ""
                        
                        # Determine affected ticker from authors/affiliations
                        ticker = self._infer_ticker(title + " " + summary)
                        
                        events.append(MemoryEvent(
                            source_type="conference",
                            ticker=ticker,
                            date=published,
                            title=f"Conference paper ({conf}): {title[:80]}",
                            content=summary,
                            significance="medium",
                            url=link,
                            metadata={"conference": conf}
                        ))
                time.sleep(3)  # arXiv rate limit
        except Exception as e:
            print(f"  Conference fetch error: {e}")
        return events
    
    def _infer_ticker(self, text: str) -> str:
        text_lower = text.lower()
        if "ionq" in text_lower or "trapped ion" in text_lower:
            return "IONQ"
        if "rigetti" in text_lower or "superconducting" in text_lower:
            return "RGTI"
        if "d-wave" in text_lower or "annealing" in text_lower:
            return "QBTS"
        if "quantinuum" in text_lower or "honeywell" in text_lower:
            return "QNT"
        if "ibm" in text_lower or "qiskit" in text_lower:
            return "IBM"
        if "google" in text_lower or "cirq" in text_lower:
            return "GOOGL"
        return "IONQ"  # Default to most-traded quantum stock


class ArxivConnector:
    """arXiv paper tracker for quantum computing research."""
    BASE_URL = "http://export.arxiv.org/api/query"
    
    def fetch(self, days_back: int = 30, max_results: int = 20) -> List[MemoryEvent]:
        events = []
        try:
            query = "cat:quant-ph+AND+(all:qubit+OR+all:quantum+computing+OR+all:error+correction)"
            url = f"{self.BASE_URL}?search_query={query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
            resp = httpx.get(url, timeout=15)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                ns = {"atom": "http://www.w3.org/2005/Atom"}
                for entry in root.findall("atom:entry", ns):
                    title = entry.find("atom:title", ns).text.strip() if entry.find("atom:title", ns) is not None else ""
                    summary = entry.find("atom:summary", ns).text.strip()[:400] if entry.find("atom:summary", ns) is not None else ""
                    published = entry.find("atom:published", ns).text[:10] if entry.find("atom:published", ns) is not None else ""
                    link = entry.find("atom:id", ns).text if entry.find("atom:id", ns) is not None else ""
                    
                    ticker = ConferenceConnector()._infer_ticker(title + " " + summary)
                    
                    # Determine significance based on keywords
                    sig = "low"
                    if any(kw in title.lower() for kw in ["logical qubit", "fault-tolerant", "error correction", "breakthrough"]):
                        sig = "high"
                    elif any(kw in title.lower() for kw in ["scalable", "milestone", "record", "novel"]):
                        sig = "medium"
                    
                    events.append(MemoryEvent(
                        source_type="arxiv",
                        ticker=ticker,
                        date=published,
                        title=f"arXiv: {title[:80]}",
                        content=summary,
                        significance=sig,
                        url=link,
                    ))
        except Exception as e:
            print(f"  arXiv fetch error: {e}")
        return events


class NewsConnector:
    """Financial news connector via RSS feeds."""
    
    def fetch(self, days_back: int = 7) -> List[MemoryEvent]:
        events = []
        try:
            import feedparser
            tickers_to_search = ["IONQ", "RGTI", "QBTS", "QUBT"]
            for ticker in tickers_to_search:
                name = COMPANY_MAP[ticker]["name"]
                query = f"{name} quantum computing stock"
                feed_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    pub_date = entry.get("published", "")
                    try:
                        from email.utils import parsedate_to_datetime
                        dt = parsedate_to_datetime(pub_date)
                        date_str = dt.strftime("%Y-%m-%d")
                    except:
                        date_str = datetime.now().strftime("%Y-%m-%d")
                    
                    title = entry.get("title", "")
                    # Strip HTML from title
                    title = re.sub(r'<[^>]+>', '', title)
                    
                    significance = "high" if any(kw in title.lower() for kw in ["earnings", "revenue", "contract", "partnership", "breakthrough"]) else "medium"
                    
                    events.append(MemoryEvent(
                        source_type="news",
                        ticker=ticker,
                        date=date_str,
                        title=title[:100],
                        content=title,
                        significance=significance,
                        url=entry.get("link", ""),
                    ))
                time.sleep(0.5)
        except ImportError:
            pass
        return events


class DataSourceOrchestrator:
    """Orchestrates all data source connectors and feeds events into memory."""
    
    def __init__(self):
        self.connectors = {
            "patent": PatentConnector(),
            "insider": InsiderTransactionConnector(),
            "job": JobPostingConnector(),
            "github": GitHubActivityConnector(),
            "conference": ConferenceConnector(),
            "arxiv": ArxivConnector(),
            "news": NewsConnector(),
        }
    
    def fetch_all(self, days_back: int = 30) -> List[MemoryEvent]:
        """Fetch events from all data sources."""
        all_events = []
        for name, connector in self.connectors.items():
            print(f"  Fetching from {name}...", end=" ")
            try:
                events = connector.fetch(days_back=days_back)
                all_events.extend(events)
                print(f"{len(events)} events")
            except Exception as e:
                print(f"ERROR: {e}")
        
        # Sort by date (newest first)
        all_events.sort(key=lambda e: e.date, reverse=True)
        return all_events
    
    def fetch_source(self, source_type: str, days_back: int = 30) -> List[MemoryEvent]:
        """Fetch events from a specific data source."""
        connector = self.connectors.get(source_type)
        if connector:
            return connector.fetch(days_back=days_back)
        return []
    
    def events_to_memory(self, events: List[MemoryEvent], memory_store) -> int:
        """Store events as knowledge facts in the memory store."""
        stored = 0
        for event in events:
            memory_store.store_knowledge(
                ticker=event.ticker,
                fact_type=event.source_type,
                content=f"[{event.date}] {event.title}: {event.content[:200]}",
                source=event.source_type,
                confidence=1.0 if event.significance == "high" else 0.7 if event.significance == "medium" else 0.4,
                ttl_days=90 if event.significance == "high" else 60 if event.significance == "medium" else 30
            )
            stored += 1
        return stored
