"""
Persistent memory store using SQLite.

Three memory types:
1. Sector Knowledge: Facts about companies, technologies, milestones
2. Signal History: Previous predictions and their outcomes
3. User Preferences: What the user cares about, risk tolerance
"""
import sqlite3
import json
import time
from datetime import datetime, timedelta
from pathlib import Path


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._init_tables()

    def _init_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS sector_knowledge (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT,
                fact_type TEXT,
                content TEXT,
                source TEXT,
                confidence REAL DEFAULT 1.0,
                created_at TEXT,
                updated_at TEXT,
                expires_at TEXT,
                access_count INTEGER DEFAULT 0,
                last_accessed TEXT
            );

            CREATE TABLE IF NOT EXISTS signal_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                article_date TEXT,
                article_title TEXT,
                article_source TEXT,
                signal_vector TEXT,  -- JSON
                reasoning TEXT,
                predicted_direction TEXT,  -- JSON {ticker: direction}
                actual_outcome TEXT,  -- JSON {ticker: return} (filled later)
                was_correct INTEGER,  -- NULL until outcome known
                model_used TEXT,
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS user_preferences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE,
                value TEXT,
                updated_at TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_consolidations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_ids TEXT,  -- JSON list of merged memory IDs
                summary TEXT,
                created_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_knowledge_ticker ON sector_knowledge(ticker);
            CREATE INDEX IF NOT EXISTS idx_knowledge_type ON sector_knowledge(fact_type);
            CREATE INDEX IF NOT EXISTS idx_signals_date ON signal_history(article_date);
        """)
        self.conn.commit()

    def store_knowledge(self, ticker: str, fact_type: str, content: str, source: str, confidence: float = 1.0, ttl_days: int = 90):
        now = datetime.utcnow().isoformat()
        expires = (datetime.utcnow() + timedelta(days=ttl_days)).isoformat()
        self.conn.execute(
            "INSERT INTO sector_knowledge (ticker, fact_type, content, source, confidence, created_at, updated_at, expires_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (ticker, fact_type, content, source, confidence, now, now, expires)
        )
        self.conn.commit()

    def store_signal(self, article_date: str, article_title: str, article_source: str, signal_vector: dict, reasoning: str, model_used: str):
        now = datetime.utcnow().isoformat()
        # Handle both formats: {ticker: score} and {ticker: {"score": x, ...}}
        predicted = {}
        for t, s in signal_vector.items():
            if isinstance(s, dict):
                score = s.get("score", 0)
            else:
                score = float(s) if s is not None else 0
            predicted[t] = "bullish" if score > 0.1 else "bearish" if score < -0.1 else "neutral"
        self.conn.execute(
            "INSERT INTO signal_history (article_date, article_title, article_source, signal_vector, reasoning, predicted_direction, model_used, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (article_date, article_title, article_source, json.dumps(signal_vector), reasoning, json.dumps(predicted), model_used, now)
        )
        self.conn.commit()

    def retrieve_knowledge(self, ticker: str = None, fact_type: str = None, limit: int = 10):
        """Retrieve relevant knowledge, sorted by recency and access frequency."""
        query = "SELECT * FROM sector_knowledge WHERE expires_at > ? "
        params = [datetime.utcnow().isoformat()]
        if ticker:
            query += "AND ticker = ? "
            params.append(ticker)
        if fact_type:
            query += "AND fact_type = ? "
            params.append(fact_type)
        query += "ORDER BY updated_at DESC, access_count DESC LIMIT ?"
        params.append(limit)

        rows = self.conn.execute(query, params).fetchall()
        # Update access counts
        for row in rows:
            self.conn.execute("UPDATE sector_knowledge SET access_count = access_count + 1, last_accessed = ? WHERE id = ?", (datetime.utcnow().isoformat(), row[0]))
        self.conn.commit()
        return rows

    def retrieve_signal_history(self, ticker: str = None, limit: int = 20):
        """Retrieve past signals, optionally filtered by ticker."""
        if ticker:
            rows = self.conn.execute(
                "SELECT * FROM signal_history WHERE signal_vector LIKE ? ORDER BY article_date DESC LIMIT ?",
                (f'%"{ticker}"%', limit)
            ).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM signal_history ORDER BY article_date DESC LIMIT ?", (limit,)).fetchall()
        return rows

    def get_accuracy_stats(self):
        """Get prediction accuracy statistics."""
        correct = self.conn.execute("SELECT COUNT(*) FROM signal_history WHERE was_correct = 1").fetchone()[0]
        incorrect = self.conn.execute("SELECT COUNT(*) FROM signal_history WHERE was_correct = 0").fetchone()[0]
        pending = self.conn.execute("SELECT COUNT(*) FROM signal_history WHERE was_correct IS NULL").fetchone()[0]
        total = correct + incorrect
        accuracy = correct / total if total > 0 else 0
        return {"correct": correct, "incorrect": incorrect, "pending": pending, "accuracy": accuracy}

    def get_memory_stats(self):
        """Get overall memory statistics."""
        knowledge_count = self.conn.execute("SELECT COUNT(*) FROM sector_knowledge").fetchone()[0]
        signal_count = self.conn.execute("SELECT COUNT(*) FROM signal_history").fetchone()[0]
        return {"knowledge_facts": knowledge_count, "signals_stored": signal_count, **self.get_accuracy_stats()}
