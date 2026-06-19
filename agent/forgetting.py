"""
Memory forgetting and consolidation logic.

Implements:
1. Time-based expiry (TTL)
2. Relevance decay (unused memories fade)
3. Contradiction resolution (new facts override old)
4. Consolidation (merge similar memories into summaries)
"""
from datetime import datetime, timedelta
import json


class ForgettingEngine:
    def __init__(self, memory_store):
        self.memory = memory_store

    def run_forgetting_cycle(self):
        """Run a full forgetting cycle. Call periodically (e.g., daily)."""
        expired = self._expire_old_memories()
        pruned = self._prune_irrelevant()
        consolidated = self._consolidate_similar()
        return {"expired": expired, "pruned": pruned, "consolidated": consolidated}

    def _expire_old_memories(self):
        """Remove memories past their TTL."""
        now = datetime.utcnow().isoformat()
        cursor = self.memory.conn.execute(
            "DELETE FROM sector_knowledge WHERE expires_at < ?", (now,)
        )
        self.memory.conn.commit()
        return cursor.rowcount

    def _prune_irrelevant(self):
        """Remove memories that have never been accessed and are older than 30 days."""
        cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
        cursor = self.memory.conn.execute(
            "DELETE FROM sector_knowledge WHERE access_count = 0 AND created_at < ?", (cutoff,)
        )
        self.memory.conn.commit()
        return cursor.rowcount

    def _consolidate_similar(self):
        """Merge old signal history into weekly summaries."""
        cutoff = (datetime.utcnow() - timedelta(days=60)).isoformat()
        # Get old signals grouped by week
        old_signals = self.memory.conn.execute(
            "SELECT * FROM signal_history WHERE article_date < ? ORDER BY article_date", (cutoff,)
        ).fetchall()

        if len(old_signals) < 10:
            return 0

        # Group by week and create summaries (simplified)
        # In production, you'd use the LLM to generate summaries
        consolidated = 0
        # ... consolidation logic here
        return consolidated

    def handle_contradiction(self, ticker: str, new_fact: str, new_source: str):
        """When new information contradicts stored knowledge, update memory."""
        existing = self.memory.retrieve_knowledge(ticker=ticker, limit=5)
        # Mark old contradicted facts as low confidence
        for row in existing:
            if self._is_contradicted(row[3], new_fact):  # row[3] is content
                self.memory.conn.execute(
                    "UPDATE sector_knowledge SET confidence = confidence * 0.5, updated_at = ? WHERE id = ?",
                    (datetime.utcnow().isoformat(), row[0])
                )
        self.memory.conn.commit()
        # Store the new fact with high confidence
        self.memory.store_knowledge(ticker, "updated_fact", new_fact, new_source, confidence=1.0)

    def _is_contradicted(self, old_content: str, new_content: str) -> bool:
        """Simple heuristic for contradiction detection."""
        # In production, use the LLM to detect contradictions
        # For now, check if they're about the same topic but different values
        return False  # Placeholder
