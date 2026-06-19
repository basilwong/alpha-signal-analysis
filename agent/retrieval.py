"""
Memory retrieval engine.

Given a query (article text or user question), retrieves the most relevant
memories to inject into the LLM context window.

Retrieval strategy:
1. Extract tickers mentioned in the query
2. Retrieve sector knowledge for those tickers (recency-weighted)
3. Retrieve recent signal history for those tickers
4. Retrieve accuracy stats (so the model knows its own strengths/weaknesses)
5. Truncate to fit within MAX_MEMORY_CONTEXT_TOKENS
"""
import re
from .config import QUANTUM_TICKERS, MAX_MEMORY_CONTEXT_TOKENS


class MemoryRetriever:
    def __init__(self, memory_store):
        self.memory = memory_store

    def retrieve_context(self, query: str, max_tokens: int = None) -> str:
        """Build a memory context string to inject into the LLM prompt."""
        max_tokens = max_tokens or MAX_MEMORY_CONTEXT_TOKENS
        mentioned_tickers = self._extract_tickers(query)

        sections = []

        # 1. Accuracy stats (always include, helps calibration)
        stats = self.memory.get_accuracy_stats()
        if stats["correct"] + stats["incorrect"] > 0:
            sections.append(f"[YOUR TRACK RECORD] Accuracy: {stats['accuracy']:.1%} ({stats['correct']}/{stats['correct']+stats['incorrect']} correct predictions)")

        # 2. Sector knowledge for mentioned tickers
        for ticker in mentioned_tickers[:3]:  # Limit to top 3 tickers
            knowledge = self.memory.retrieve_knowledge(ticker=ticker, limit=5)
            if knowledge:
                facts = [row[3] for row in knowledge]  # content field
                sections.append(f"[MEMORY: {ticker}] " + " | ".join(facts[:3]))

        # 3. Recent signal history for mentioned tickers
        for ticker in mentioned_tickers[:2]:
            signals = self.memory.retrieve_signal_history(ticker=ticker, limit=3)
            if signals:
                for sig in signals[:2]:
                    date = sig[1]  # article_date
                    title = sig[2]  # article_title
                    outcome = sig[7]  # actual_outcome
                    if outcome:
                        sections.append(f"[PAST SIGNAL: {ticker} on {date}] '{title}' → Outcome: {outcome}")

        # 4. Truncate to fit context window
        context = "\n".join(sections)
        # Rough token estimate: 1 token ≈ 4 chars
        while len(context) > max_tokens * 4 and sections:
            sections.pop()
            context = "\n".join(sections)

        return context

    def _extract_tickers(self, text: str) -> list:
        """Extract mentioned tickers from text."""
        mentioned = []
        text_upper = text.upper()
        for ticker in QUANTUM_TICKERS:
            if ticker in text_upper:
                mentioned.append(ticker)
        # Also check company names
        name_map = {"IONQ": "IonQ", "RIGETTI": "RGTI", "D-WAVE": "QBTS", "QUANTINUUM": "QNT", "HONEYWELL": "HON"}
        for name, ticker in name_map.items():
            if name.lower() in text.lower() and ticker not in mentioned:
                mentioned.append(ticker)
        return mentioned if mentioned else QUANTUM_TICKERS[:5]  # Default to pure-play tickers
