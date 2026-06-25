"""
Memory Improvement Loop for the Alpha Signal Analysis Agent.

Implements three memory types from cognitive science:
1. Semantic Memory: Facts and knowledge (already in memory.py)
2. Episodic Memory: Past experiences and outcomes
3. Procedural Memory: Behavioral rules learned from experience

Plus the feedback loop:
  Capture traces → Analyze outcomes → Generate rules → Update memory
"""
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.memory import MemoryStore
from agent.config import QUANTUM_TICKERS


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Episode:
    """A single prediction episode with its outcome."""
    date: str
    ticker: str
    predicted_score: float
    predicted_direction: str  # bullish, bearish, neutral
    actual_return_5d: Optional[float]
    actual_direction: Optional[str]
    was_correct: Optional[bool]
    article_title: str
    source_type: str
    reasoning_summary: str
    lesson: Optional[str] = None  # What can be learned from this


@dataclass
class ProceduralRule:
    """A behavioral rule learned from experience."""
    rule_id: str
    rule_text: str
    confidence: float  # 0-1, based on evidence strength
    evidence_count: int  # How many episodes support this rule
    created_at: str
    last_validated: str
    category: str  # source_bias, ticker_bias, pattern, calibration, multi_source


# ============================================================
# EPISODIC MEMORY
# ============================================================

class EpisodicMemory:
    """Stores and retrieves past prediction episodes with outcomes."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store
        self._ensure_tables()
    
    def _ensure_tables(self):
        self.memory.conn.executescript("""
            CREATE TABLE IF NOT EXISTS episodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                ticker TEXT,
                predicted_score REAL,
                predicted_direction TEXT,
                actual_return_5d REAL,
                actual_direction TEXT,
                was_correct INTEGER,
                article_title TEXT,
                source_type TEXT,
                reasoning_summary TEXT,
                lesson TEXT,
                created_at TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_ticker ON episodes(ticker);
            CREATE INDEX IF NOT EXISTS idx_episodes_correct ON episodes(was_correct);
            CREATE INDEX IF NOT EXISTS idx_episodes_source ON episodes(source_type);
        """)
        self.memory.conn.commit()
    
    def store_episode(self, episode: Episode):
        self.memory.conn.execute(
            """INSERT INTO episodes (date, ticker, predicted_score, predicted_direction,
               actual_return_5d, actual_direction, was_correct, article_title, source_type,
               reasoning_summary, lesson, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (episode.date, episode.ticker, episode.predicted_score,
             episode.predicted_direction, episode.actual_return_5d,
             episode.actual_direction, episode.was_correct,
             episode.article_title, episode.source_type,
             episode.reasoning_summary, episode.lesson,
             datetime.utcnow().isoformat())
        )
        self.memory.conn.commit()
    
    def get_similar_episodes(self, ticker: str = None, source_type: str = None, limit: int = 5) -> List[Episode]:
        """Retrieve past episodes similar to the current situation."""
        query = "SELECT * FROM episodes WHERE was_correct IS NOT NULL "
        params = []
        if ticker:
            query += "AND ticker = ? "
            params.append(ticker)
        if source_type:
            query += "AND source_type = ? "
            params.append(source_type)
        query += "ORDER BY date DESC LIMIT ?"
        params.append(limit)
        
        rows = self.memory.conn.execute(query, params).fetchall()
        episodes = []
        for row in rows:
            episodes.append(Episode(
                date=row[1], ticker=row[2], predicted_score=row[3],
                predicted_direction=row[4], actual_return_5d=row[5],
                actual_direction=row[6], was_correct=bool(row[7]),
                article_title=row[8], source_type=row[9],
                reasoning_summary=row[10], lesson=row[11]
            ))
        return episodes
    
    def get_accuracy_by_category(self) -> Dict:
        """Get accuracy breakdown by ticker, source, and direction."""
        stats = {}
        
        # By ticker
        rows = self.memory.conn.execute(
            "SELECT ticker, SUM(was_correct), COUNT(*) FROM episodes WHERE was_correct IS NOT NULL GROUP BY ticker"
        ).fetchall()
        stats["by_ticker"] = {row[0]: {"correct": row[1], "total": row[2], "accuracy": row[1]/row[2] if row[2] > 0 else 0} for row in rows}
        
        # By source type
        rows = self.memory.conn.execute(
            "SELECT source_type, SUM(was_correct), COUNT(*) FROM episodes WHERE was_correct IS NOT NULL GROUP BY source_type"
        ).fetchall()
        stats["by_source"] = {row[0]: {"correct": row[1], "total": row[2], "accuracy": row[1]/row[2] if row[2] > 0 else 0} for row in rows}
        
        # By predicted direction
        rows = self.memory.conn.execute(
            "SELECT predicted_direction, SUM(was_correct), COUNT(*) FROM episodes WHERE was_correct IS NOT NULL GROUP BY predicted_direction"
        ).fetchall()
        stats["by_direction"] = {row[0]: {"correct": row[1], "total": row[2], "accuracy": row[1]/row[2] if row[2] > 0 else 0} for row in rows}
        
        # Overall
        row = self.memory.conn.execute(
            "SELECT SUM(was_correct), COUNT(*) FROM episodes WHERE was_correct IS NOT NULL"
        ).fetchone()
        stats["overall"] = {"correct": row[0] or 0, "total": row[1] or 0, "accuracy": (row[0] or 0)/(row[1] or 1)}
        
        return stats


# ============================================================
# PROCEDURAL MEMORY
# ============================================================

class ProceduralMemory:
    """Stores and retrieves behavioral rules learned from experience."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store
        self._ensure_tables()
    
    def _ensure_tables(self):
        self.memory.conn.executescript("""
            CREATE TABLE IF NOT EXISTS procedural_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_id TEXT UNIQUE,
                rule_text TEXT,
                confidence REAL,
                evidence_count INTEGER,
                created_at TEXT,
                last_validated TEXT,
                category TEXT
            );
        """)
        self.memory.conn.commit()
    
    def store_rule(self, rule: ProceduralRule):
        """Store or update a procedural rule."""
        existing = self.memory.conn.execute(
            "SELECT id FROM procedural_rules WHERE rule_id = ?", (rule.rule_id,)
        ).fetchone()
        
        if existing:
            self.memory.conn.execute(
                """UPDATE procedural_rules SET rule_text=?, confidence=?, evidence_count=?,
                   last_validated=? WHERE rule_id=?""",
                (rule.rule_text, rule.confidence, rule.evidence_count,
                 rule.last_validated, rule.rule_id)
            )
        else:
            self.memory.conn.execute(
                """INSERT INTO procedural_rules (rule_id, rule_text, confidence, evidence_count,
                   created_at, last_validated, category)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (rule.rule_id, rule.rule_text, rule.confidence, rule.evidence_count,
                 rule.created_at, rule.last_validated, rule.category)
            )
        self.memory.conn.commit()
    
    def get_active_rules(self, min_confidence: float = 0.5) -> List[ProceduralRule]:
        """Get all active rules above confidence threshold."""
        rows = self.memory.conn.execute(
            "SELECT * FROM procedural_rules WHERE confidence >= ? ORDER BY confidence DESC",
            (min_confidence,)
        ).fetchall()
        return [ProceduralRule(
            rule_id=row[1], rule_text=row[2], confidence=row[3],
            evidence_count=row[4], created_at=row[5],
            last_validated=row[6], category=row[7]
        ) for row in rows]
    
    def get_rules_as_context(self, min_confidence: float = 0.5) -> str:
        """Format active rules as context for the LLM prompt."""
        rules = self.get_active_rules(min_confidence)
        if not rules:
            return ""
        
        lines = ["[BEHAVIORAL RULES LEARNED FROM EXPERIENCE]"]
        for rule in rules:
            conf_label = "HIGH" if rule.confidence >= 0.8 else "MEDIUM" if rule.confidence >= 0.6 else "LOW"
            lines.append(f"  [{conf_label} confidence, {rule.evidence_count} observations] {rule.rule_text}")
        return "\n".join(lines)


# ============================================================
# FEEDBACK LOOP
# ============================================================

class FeedbackLoop:
    """Analyzes prediction outcomes and generates procedural rules."""
    
    def __init__(self, memory_store: MemoryStore, llm_client=None):
        self.memory = memory_store
        self.episodic = EpisodicMemory(memory_store)
        self.procedural = ProceduralMemory(memory_store)
        self.llm_client = llm_client  # OpenAI-compatible client for rule generation
    
    def record_outcome(self, prediction: dict, actual_return_5d: float):
        """Record the outcome of a prediction and store as an episode."""
        signal = prediction.get("signal", {})
        sv = signal.get("signal_vector", signal)
        
        for ticker in QUANTUM_TICKERS:
            score = 0
            if isinstance(sv, dict):
                val = sv.get(ticker, 0)
                score = val if isinstance(val, (int, float)) else (val.get("score", 0) if isinstance(val, dict) else 0)
            
            if abs(score) < 0.3:
                continue  # Skip trivial predictions
            
            predicted_dir = "bullish" if score > 0 else "bearish"
            actual_dir = "bullish" if actual_return_5d > 0 else "bearish" if actual_return_5d < 0 else "neutral"
            was_correct = (predicted_dir == actual_dir)
            
            episode = Episode(
                date=prediction.get("date", ""),
                ticker=ticker,
                predicted_score=score,
                predicted_direction=predicted_dir,
                actual_return_5d=actual_return_5d,
                actual_direction=actual_dir,
                was_correct=was_correct,
                article_title=prediction.get("title", "")[:100],
                source_type=prediction.get("source", "news"),
                reasoning_summary=signal.get("chain_of_thought", "")[:200],
            )
            self.episodic.store_episode(episode)
    
    def analyze_and_generate_rules(self) -> List[ProceduralRule]:
        """Analyze all episodes and generate/update procedural rules."""
        stats = self.episodic.get_accuracy_by_category()
        rules = []
        now = datetime.utcnow().isoformat()
        
        # Rule 1: Source-specific confidence calibration
        for source, data in stats.get("by_source", {}).items():
            if data["total"] >= 10:
                acc = data["accuracy"]
                if acc >= 0.65:
                    rule = ProceduralRule(
                        rule_id=f"source_strong_{source}",
                        rule_text=f"Predictions from {source} articles have been {acc*100:.0f}% accurate ({data['total']} observations). Assign HIGH conviction when analyzing {source} content.",
                        confidence=min(acc, 0.95),
                        evidence_count=data["total"],
                        created_at=now, last_validated=now,
                        category="source_bias"
                    )
                    rules.append(rule)
                elif acc <= 0.45:
                    rule = ProceduralRule(
                        rule_id=f"source_weak_{source}",
                        rule_text=f"Predictions from {source} articles have been only {acc*100:.0f}% accurate ({data['total']} observations). Be CONSERVATIVE and cap conviction at 0.5 when analyzing {source} content.",
                        confidence=min(1 - acc, 0.95),
                        evidence_count=data["total"],
                        created_at=now, last_validated=now,
                        category="source_bias"
                    )
                    rules.append(rule)
        
        # Rule 2: Ticker-specific calibration
        for ticker, data in stats.get("by_ticker", {}).items():
            if data["total"] >= 10:
                acc = data["accuracy"]
                if acc >= 0.65:
                    rule = ProceduralRule(
                        rule_id=f"ticker_strong_{ticker}",
                        rule_text=f"Predictions for {ticker} have been {acc*100:.0f}% accurate. The model has good signal for this ticker.",
                        confidence=min(acc, 0.95),
                        evidence_count=data["total"],
                        created_at=now, last_validated=now,
                        category="ticker_bias"
                    )
                    rules.append(rule)
                elif acc <= 0.40:
                    rule = ProceduralRule(
                        rule_id=f"ticker_weak_{ticker}",
                        rule_text=f"Predictions for {ticker} have been only {acc*100:.0f}% accurate. Reduce conviction for this ticker or consider that the stock may be driven by factors not captured in news.",
                        confidence=min(1 - acc, 0.95),
                        evidence_count=data["total"],
                        created_at=now, last_validated=now,
                        category="ticker_bias"
                    )
                    rules.append(rule)
        
        # Rule 3: Direction-specific calibration
        for direction, data in stats.get("by_direction", {}).items():
            if data["total"] >= 15:
                acc = data["accuracy"]
                if acc >= 0.60 and direction == "bullish":
                    rule = ProceduralRule(
                        rule_id="direction_bullish_reliable",
                        rule_text=f"Bullish predictions have been {acc*100:.0f}% accurate. The model is better at identifying positive catalysts than negative ones.",
                        confidence=acc,
                        evidence_count=data["total"],
                        created_at=now, last_validated=now,
                        category="calibration"
                    )
                    rules.append(rule)
                elif acc <= 0.40 and direction == "bearish":
                    rule = ProceduralRule(
                        rule_id="direction_bearish_unreliable",
                        rule_text=f"Bearish predictions have been only {acc*100:.0f}% accurate. Be cautious with negative signals. Consider that bad news may already be priced in.",
                        confidence=1 - acc,
                        evidence_count=data["total"],
                        created_at=now, last_validated=now,
                        category="calibration"
                    )
                    rules.append(rule)
        
        # Rule 4: Overall calibration
        overall = stats.get("overall", {})
        if overall.get("total", 0) >= 20:
            acc = overall["accuracy"]
            rule = ProceduralRule(
                rule_id="overall_calibration",
                rule_text=f"Overall prediction accuracy is {acc*100:.0f}% across {overall['total']} predictions. {'Model is well-calibrated.' if 0.55 <= acc <= 0.70 else 'Model tends to be overconfident.' if acc < 0.50 else 'Model has strong predictive power.'}",
                confidence=0.9,
                evidence_count=overall["total"],
                created_at=now, last_validated=now,
                category="calibration"
            )
            rules.append(rule)
        
        # Store all generated rules
        for rule in rules:
            self.procedural.store_rule(rule)
        
        return rules
    
    def generate_advanced_rules_with_llm(self, episodes: List[Episode]) -> List[ProceduralRule]:
        """Use an LLM to analyze episodes and generate more nuanced rules."""
        if not self.llm_client or not episodes:
            return []
        
        # Format episodes for the LLM
        episode_text = "\n".join([
            f"- {e.date} | {e.ticker} | Predicted: {e.predicted_direction} ({e.predicted_score:+.1f}) | Actual: {e.actual_direction} ({e.actual_return_5d:+.2%}) | {'CORRECT' if e.was_correct else 'WRONG'} | Source: {e.source_type} | Article: {e.article_title[:50]}"
            for e in episodes[:30]
        ])
        
        prompt = f"""Analyze these prediction episodes and generate 3-5 behavioral rules that would improve future predictions.

EPISODES:
{episode_text}

Generate rules in this JSON format:
[
  {{"rule_id": "pattern_name", "rule_text": "When X happens, do Y because Z", "confidence": 0.7, "category": "pattern"}}
]

Focus on:
1. Patterns in what types of articles lead to correct vs incorrect predictions
2. Whether certain tickers are more predictable than others
3. Whether the model is systematically overconfident or underconfident
4. Any temporal patterns (e.g., predictions are better/worse on certain days)

Output ONLY valid JSON array."""

        try:
            response = self.llm_client.chat.completions.create(
                model="gpt-5-nano",
                messages=[
                    {"role": "system", "content": "You are an expert quantitative analyst reviewing prediction performance data. Generate concise, actionable behavioral rules."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=1000
            )
            # Handle both regular and reasoning model responses
            content = response.choices[0].message.content or ""
            if not content and hasattr(response.choices[0].message, 'reasoning_content'):
                content = response.choices[0].message.reasoning_content or ""
            if not content:
                return []
            # Parse JSON
            s = content.find('[')
            e = content.rfind(']') + 1
            if s != -1:
                rules_data = json.loads(content[s:e])
                now = datetime.utcnow().isoformat()
                rules = []
                for rd in rules_data:
                    rule = ProceduralRule(
                        rule_id=rd.get("rule_id", f"llm_rule_{len(rules)}"),
                        rule_text=rd.get("rule_text", ""),
                        confidence=rd.get("confidence", 0.6),
                        evidence_count=len(episodes),
                        created_at=now, last_validated=now,
                        category=rd.get("category", "pattern")
                    )
                    rules.append(rule)
                    self.procedural.store_rule(rule)
                return rules
        except Exception as ex:
            print(f"  LLM rule generation failed: {ex}")
        return []
    
    def run_full_loop(self, predictions: List[dict], market_data: dict) -> dict:
        """Run the complete feedback loop: record outcomes, analyze, generate rules."""
        from agent.config import QUANTUM_TICKERS
        
        # Step 1: Record outcomes for all predictions
        outcomes_recorded = 0
        for pred in predictions:
            if pred.get("status") != "success":
                continue
            date = pred.get("date", "")
            signal = pred.get("signal", {})
            sv = signal.get("signal_vector", signal)
            
            if not date or not isinstance(sv, dict):
                continue
            
            for ticker in QUANTUM_TICKERS[:5]:  # Pure-play only
                score = sv.get(ticker, 0)
                if isinstance(score, dict):
                    score = score.get("score", 0)
                if abs(score) < 0.3:
                    continue
                
                # Get actual 5-day return
                actual_ret = self._get_forward_return(market_data, ticker, date, horizon=5)
                if actual_ret is not None:
                    self.record_outcome(pred, actual_ret)
                    outcomes_recorded += 1
        
        # Step 2: Analyze and generate rules
        rules = self.analyze_and_generate_rules()
        
        # Step 3: Generate advanced rules with LLM (if available)
        episodes = self.episodic.get_similar_episodes(limit=30)
        llm_rules = self.generate_advanced_rules_with_llm(episodes)
        
        # Step 4: Get updated stats
        stats = self.episodic.get_accuracy_by_category()
        
        return {
            "outcomes_recorded": outcomes_recorded,
            "rules_generated": len(rules),
            "llm_rules_generated": len(llm_rules),
            "accuracy_stats": stats,
            "active_rules": [asdict(r) for r in self.procedural.get_active_rules()],
        }
    
    def _get_forward_return(self, market_data: dict, ticker: str, event_date: str, horizon: int = 5) -> Optional[float]:
        """Get the forward return for a ticker from event_date."""
        if ticker not in market_data:
            return None
        dates = market_data[ticker]["dates"]
        values = market_data[ticker]["values"]
        try:
            start_idx = next(i for i, d in enumerate(dates) if d >= event_date)
        except StopIteration:
            return None
        end_idx = min(start_idx + horizon, len(values) - 1)
        if end_idx <= start_idx or values[start_idx] == 0:
            return None
        return (values[end_idx] - values[start_idx]) / values[start_idx]


# ============================================================
# ENHANCED RETRIEVER (uses all 3 memory types)
# ============================================================

class EnhancedRetriever:
    """Retrieves context from all three memory types for the LLM prompt."""
    
    def __init__(self, memory_store: MemoryStore):
        self.memory = memory_store
        self.episodic = EpisodicMemory(memory_store)
        self.procedural = ProceduralMemory(memory_store)
    
    def build_full_context(self, query: str, source_type: str = "news", max_tokens: int = 4000) -> str:
        """Build a comprehensive memory context using all three memory types."""
        sections = []
        
        # 1. Procedural rules (always include, highest priority)
        rules_context = self.procedural.get_rules_as_context(min_confidence=0.5)
        if rules_context:
            sections.append(rules_context)
        
        # 2. Episodic memory (relevant past experiences)
        tickers = self._extract_tickers(query)
        episodes = []
        for ticker in tickers[:3]:
            eps = self.episodic.get_similar_episodes(ticker=ticker, source_type=source_type, limit=3)
            episodes.extend(eps)
        
        if episodes:
            ep_lines = ["[PAST EXPERIENCES]"]
            for ep in episodes[:5]:
                outcome = "CORRECT" if ep.was_correct else "WRONG"
                ep_lines.append(f"  [{outcome}] {ep.date} {ep.ticker}: Predicted {ep.predicted_direction} ({ep.predicted_score:+.1f}), actual {ep.actual_return_5d:+.2%}. {ep.lesson or ''}")
            sections.append("\n".join(ep_lines))
        
        # 3. Semantic memory (facts and knowledge)
        from agent.retrieval import MemoryRetriever
        base_retriever = MemoryRetriever(self.memory)
        semantic_context = base_retriever.retrieve_context(query)
        if semantic_context:
            sections.append(semantic_context)
        
        # 4. Accuracy stats (self-awareness)
        stats = self.episodic.get_accuracy_by_category()
        if stats.get("overall", {}).get("total", 0) > 0:
            overall = stats["overall"]
            sections.append(f"[YOUR TRACK RECORD] Overall accuracy: {overall['accuracy']*100:.0f}% ({overall['correct']}/{overall['total']} correct)")
        
        # Combine and truncate
        context = "\n\n".join(sections)
        max_chars = max_tokens * 4
        if len(context) > max_chars:
            context = context[:max_chars] + "\n[... memory truncated to fit context window]"
        
        return context
    
    def _extract_tickers(self, text: str) -> List[str]:
        mentioned = []
        text_upper = text.upper()
        for ticker in QUANTUM_TICKERS:
            if ticker in text_upper:
                mentioned.append(ticker)
        name_map = {"IONQ": "IonQ", "RIGETTI": "RGTI", "D-WAVE": "QBTS", "QUANTINUUM": "QNT", "HONEYWELL": "HON"}
        for name, ticker in name_map.items():
            if name.lower() in text.lower() and ticker not in mentioned:
                mentioned.append(ticker)
        return mentioned if mentioned else QUANTUM_TICKERS[:5]
