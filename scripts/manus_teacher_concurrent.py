"""
Manus Teacher Pipeline (Concurrent Version)
Uses asyncio + aiohttp to run multiple Manus API tasks concurrently.
Always uses agent_profile: "max" for highest quality.

Usage:
    python scripts/manus_teacher_concurrent.py --phase 1
    python scripts/manus_teacher_concurrent.py --phase 3
    python scripts/manus_teacher_concurrent.py --phase all
    python scripts/manus_teacher_concurrent.py --aggregate
"""

import asyncio
import aiohttp
import json
import time
import argparse
import random
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# ============================================================
# Configuration
# ============================================================

API_KEY = "sk-3oFNwU2aOnhvss4PtIMzRIEX5XvFMyLjkV8BTw_kAMkZKVAHu9HKMD3NIIe2Wuwt3XnWF_Qh3ppmJ_qZ_z7CsuFoqTYq"
BASE_URL = "https://api.manus.ai/v2"
HEADERS = {"x-manus-api-key": API_KEY, "Content-Type": "application/json"}

# Concurrency settings
MAX_CONCURRENT = 10  # Start conservative, scale up
CREATION_DELAY = 2.0  # Seconds between task creations to avoid bursts
POLL_INTERVAL = 30  # Seconds between polls
MAX_POLL_TIME = 900  # 15 minutes max per task
RETRY_DELAY = 60
MAX_RETRIES = 3

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_TRAINING = PROJECT_ROOT / "data" / "training"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"
LOGS_DIR = PROJECT_ROOT / "logs"

DATA_TRAINING.mkdir(parents=True, exist_ok=True)
DATA_EVAL.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Shared Context
# ============================================================

SHARED_CONTEXT = """**The quantum computing universe (9 tickers):**
- IONQ: IonQ (trapped-ion, 100% quantum revenue, pure-play)
- RGTI: Rigetti Computing (superconducting, 100% quantum revenue, pure-play)
- QBTS: D-Wave Quantum (quantum annealing, 100% quantum revenue, pure-play)
- QUBT: Quantum Computing Inc. (neutral atom, 100% quantum revenue, pure-play)
- IBM: International Business Machines (superconducting, ~2% quantum revenue)
- GOOGL: Alphabet/Google (superconducting, <0.1% quantum revenue)
- MSFT: Microsoft (topological, <0.1% quantum revenue)
- HON: Honeywell/Quantinuum (trapped-ion, ~5% quantum revenue via Quantinuum subsidiary)
- NVDA: NVIDIA (adjacent/enabler, sells simulation hardware, ~1% quantum revenue)

**Scoring guidelines:**
- Scores range from -2.0 (strongly bearish) to +2.0 (strongly bullish)
- Pure-play companies (IONQ, RGTI, QBTS, QUBT): full range [-2.0, +2.0]
- HON: max +/-0.3 | IBM: max +/-0.15 | NVDA: max +/-0.10 | GOOGL, MSFT: max +/-0.05
- If NOT about quantum computing: all scores = 0.0

**Technology competitive dynamics:**
- Trapped-ion breakthroughs → bullish IONQ/HON, bearish RGTI/IBM/GOOGL
- Superconducting breakthroughs → bullish RGTI/IBM/GOOGL, bearish IONQ/HON
- Error correction advances → benefit ALL gate-based approaches
- Government funding → broadly bullish for entire sector
- Negative news about one company → slightly bullish for direct competitors"""

# ============================================================
# Schemas
# ============================================================

SIGNAL_SCHEMA = {
    "type": "object",
    "properties": {
        "signal_vector": {
            "type": "object",
            "properties": {
                "IONQ": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "RGTI": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "QBTS": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "QUBT": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "IBM": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "GOOGL": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "MSFT": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "HON": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "NVDA": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
            },
            "required": ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"],
            "additionalProperties": False
        },
        "event_type": {"type": "string"},
        "time_horizon": {"type": "string", "enum": ["intraday", "2-5 days", "1-2 weeks", "1+ month"]},
        "signal_decay": {"type": "string", "enum": ["fast", "medium", "slow"]},
        "information_novelty": {"type": "string", "enum": ["high", "medium", "low"]},
        "technical_translation": {"type": "string"},
        "signal_rationale": {"type": "string"},
        "chain_of_thought": {"type": "string"}
    },
    "required": ["signal_vector", "event_type", "time_horizon", "signal_decay", "information_novelty", "technical_translation", "signal_rationale", "chain_of_thought"],
    "additionalProperties": False
}

FOLLOWUP_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "updated_signal_vector": {
            "type": "object",
            "properties": {
                "IONQ": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "RGTI": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "QBTS": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "QUBT": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "IBM": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "GOOGL": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "MSFT": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "HON": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "NVDA": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
            },
            "required": ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"],
            "additionalProperties": False
        },
        "scores_changed": {"type": "boolean"},
        "reasoning_for_change": {"type": ["string", "null"]}
    },
    "required": ["answer", "updated_signal_vector", "scores_changed", "reasoning_for_change"],
    "additionalProperties": False
}

# ============================================================
# Async API Helpers
# ============================================================

class RateLimiter:
    """Adaptive rate limiter that backs off on 429 errors."""
    def __init__(self, max_concurrent: int = MAX_CONCURRENT):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.current_concurrent = max_concurrent
        self.tasks_completed = 0
        self.rate_limit_hits = 0
        self.creation_delay = CREATION_DELAY
    
    def report_success(self):
        self.tasks_completed += 1
        # Scale up after 20 successful tasks with no rate limits
        if self.tasks_completed % 20 == 0 and self.rate_limit_hits == 0:
            new_limit = min(self.current_concurrent + 10, 50)
            if new_limit > self.current_concurrent:
                print(f"  [RATE] Scaling up concurrency: {self.current_concurrent} -> {new_limit}")
                self.current_concurrent = new_limit
                self.semaphore = asyncio.Semaphore(new_limit)
    
    def report_rate_limit(self):
        self.rate_limit_hits += 1
        new_limit = max(self.current_concurrent // 2, 5)
        print(f"  [RATE] Rate limited! Reducing concurrency: {self.current_concurrent} -> {new_limit}")
        self.current_concurrent = new_limit
        self.semaphore = asyncio.Semaphore(new_limit)
        self.creation_delay = min(self.creation_delay * 2, 10.0)


async def create_task_async(session: aiohttp.ClientSession, prompt: str, schema: dict, 
                            rate_limiter: RateLimiter, retries: int = MAX_RETRIES) -> Optional[str]:
    """Create a Manus task asynchronously."""
    for attempt in range(retries):
        try:
            payload = {
                "message": {"content": prompt},
                "structured_output_schema": schema,
                "agent_profile": "manus-1.6-max"
            }
            async with session.post(f"{BASE_URL}/task.create", headers=HEADERS, json=payload) as resp:
                if resp.status == 429:
                    rate_limiter.report_rate_limit()
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    print(f"  [429] Rate limited. Waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue
                
                data = await resp.json()
                if data.get("ok"):
                    await asyncio.sleep(rate_limiter.creation_delay)  # Stagger creations
                    return data["task_id"]
                else:
                    error_msg = data.get("error", {}).get("message", "unknown")
                    print(f"  [WARN] Task creation failed (attempt {attempt+1}): {error_msg}")
                    if attempt < retries - 1:
                        await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  [ERROR] Task creation exception (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                await asyncio.sleep(RETRY_DELAY)
    return None


async def send_message_async(session: aiohttp.ClientSession, task_id: str, message: str, 
                             schema: dict, retries: int = MAX_RETRIES) -> bool:
    """Send a follow-up message asynchronously."""
    for attempt in range(retries):
        try:
            payload = {
                "task_id": task_id,
                "message": {"content": message},
                "structured_output_schema": schema,
                "agent_profile": "manus-1.6-max"
            }
            async with session.post(f"{BASE_URL}/task.sendMessage", headers=HEADERS, json=payload) as resp:
                if resp.status == 429:
                    await asyncio.sleep(60)
                    continue
                data = await resp.json()
                if data.get("ok"):
                    return True
                else:
                    print(f"  [WARN] sendMessage failed: {data.get('error', {}).get('message', 'unknown')}")
                    if attempt < retries - 1:
                        await asyncio.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  [ERROR] sendMessage exception: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(RETRY_DELAY)
    return False


async def poll_for_result_async(session: aiohttp.ClientSession, task_id: str, 
                                max_time: int = MAX_POLL_TIME) -> dict:
    """Poll a task until completion."""
    start = time.time()
    while time.time() - start < max_time:
        try:
            params = {"task_id": task_id, "order": "desc", "limit": 20}
            async with session.get(f"{BASE_URL}/task.listMessages", headers=HEADERS, params=params) as resp:
                if resp.status == 429:
                    await asyncio.sleep(60)
                    continue
                data = await resp.json()
                if not data.get("ok"):
                    await asyncio.sleep(POLL_INTERVAL)
                    continue

                messages = data.get("messages", [])
                
                # Check for structured output result
                for msg in messages:
                    if msg.get("type") == "structured_output_result":
                        return msg["structured_output_result"]
                
                # Check for status updates
                for msg in messages:
                    if msg.get("type") == "status_update":
                        status = msg.get("status_update", {}).get("agent_status")
                        if status == "stopped":
                            # Check all messages for structured output
                            params_all = {"task_id": task_id, "order": "asc"}
                            async with session.get(f"{BASE_URL}/task.listMessages", headers=HEADERS, params=params_all) as resp2:
                                all_data = await resp2.json()
                                for m in all_data.get("messages", []):
                                    if m.get("type") == "structured_output_result":
                                        return m["structured_output_result"]
                            return {"success": False, "error": "Task stopped without structured output"}
                        elif status == "error":
                            return {"success": False, "error": "Task errored"}

        except Exception as e:
            print(f"  [WARN] Poll exception for {task_id}: {e}")
        
        await asyncio.sleep(POLL_INTERVAL)
    
    return {"success": False, "error": "Polling timeout"}


# ============================================================
# File I/O Helpers (thread-safe with lock)
# ============================================================

_file_lock = asyncio.Lock()

async def append_result_async(filepath: Path, result: dict):
    """Thread-safe append to JSONL file."""
    async with _file_lock:
        with open(filepath, "a") as f:
            f.write(json.dumps(result) + "\n")


def get_completed_indices(filepath: Path) -> set:
    """Get set of already-completed article indices."""
    indices = set()
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    try:
                        data = json.loads(line)
                        if "article_idx" in data:
                            indices.add(data["article_idx"])
                    except json.JSONDecodeError:
                        continue
    return indices


def load_existing_results(filepath: Path) -> list:
    """Load existing results."""
    results = []
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return results


# ============================================================
# Validation (local, no API calls needed)
# ============================================================

def validate_signal_format(signal: dict) -> tuple:
    """Local format validation. Returns (is_valid, issues, severity)."""
    issues = []
    sv = signal.get("signal_vector", {})
    required_tickers = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]
    
    for ticker in required_tickers:
        if ticker not in sv:
            issues.append(f"Missing ticker: {ticker}")
        else:
            score = sv[ticker].get("score", 0)
            if ticker in ["IONQ", "RGTI", "QBTS", "QUBT"]:
                if abs(score) > 2.0:
                    issues.append(f"{ticker} score {score} exceeds [-2.0, 2.0]")
            elif ticker == "HON":
                if abs(score) > 0.3:
                    issues.append(f"HON score {score} exceeds [-0.3, 0.3]")
            elif ticker == "IBM":
                if abs(score) > 0.15:
                    issues.append(f"IBM score {score} exceeds [-0.15, 0.15]")
            elif ticker == "NVDA":
                if abs(score) > 0.10:
                    issues.append(f"NVDA score {score} exceeds [-0.10, 0.10]")
            elif ticker in ["GOOGL", "MSFT"]:
                if abs(score) > 0.05:
                    issues.append(f"{ticker} score {score} exceeds [-0.05, 0.05]")
    
    if not issues:
        severity = "pass"
    elif any("Missing" in i for i in issues):
        severity = "critical"
    elif any("exceeds" in i for i in issues):
        severity = "major"
    else:
        severity = "minor"
    
    return (len(issues) == 0, issues, severity)


# ============================================================
# Generic concurrent task runner
# ============================================================

async def process_single_task(session: aiohttp.ClientSession, idx: int, prompt: str, 
                              schema: dict, metadata: dict, output_file: Path,
                              rate_limiter: RateLimiter, category: str):
    """Process a single task: create, poll, validate, save."""
    async with rate_limiter.semaphore:
        start_time = time.time()
        print(f"  [{category}][{idx}] Starting...")
        
        task_id = await create_task_async(session, prompt, schema, rate_limiter)
        if not task_id:
            record = {**metadata, "article_idx": idx, "category": category,
                      "task_id": None, "success": False, "signal": None,
                      "error": "Failed to create task", "timestamp": datetime.now().isoformat(),
                      "time_seconds": time.time() - start_time}
            await append_result_async(output_file, record)
            return record
        
        print(f"  [{category}][{idx}] Task created: {task_id}")
        result = await poll_for_result_async(session, task_id)
        elapsed = time.time() - start_time
        
        success = result.get("success", False)
        signal = result.get("value") if success else None
        
        # Validate format
        validation = None
        if success and signal:
            is_valid, issues, severity = validate_signal_format(signal)
            validation = {"is_valid": is_valid, "issues": issues, "severity": severity}
            if severity == "critical":
                print(f"  [{category}][{idx}] CRITICAL validation failure: {issues}")
        
        record = {
            **metadata,
            "article_idx": idx,
            "category": category,
            "task_id": task_id,
            "success": success,
            "signal": signal,
            "error": result.get("error"),
            "validation": validation,
            "time_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }
        
        await append_result_async(output_file, record)
        rate_limiter.report_success()
        
        status = "SUCCESS" if success else "FAILED"
        print(f"  [{category}][{idx}] {status} ({elapsed:.0f}s)")
        return record


async def run_batch_concurrent(tasks_data: list, schema: dict, output_file: Path, 
                               category: str, max_concurrent: int = MAX_CONCURRENT):
    """Run a batch of tasks concurrently.
    
    tasks_data: list of dicts with keys 'idx', 'prompt', 'metadata'
    """
    rate_limiter = RateLimiter(max_concurrent)
    completed = get_completed_indices(output_file)
    
    # Filter out already completed
    pending = [t for t in tasks_data if t["idx"] not in completed]
    print(f"\n{'='*60}")
    print(f"CATEGORY: {category}")
    print(f"Total: {len(tasks_data)} | Already done: {len(completed)} | Pending: {len(pending)}")
    print(f"Max concurrent: {max_concurrent}")
    print(f"{'='*60}\n")
    
    if not pending:
        print("All tasks already completed!")
        return
    
    connector = aiohttp.TCPConnector(limit=max_concurrent + 10)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            process_single_task(
                session, t["idx"], t["prompt"], schema, t["metadata"],
                output_file, rate_limiter, category
            )
            for t in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Summary
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    failures = len(results) - successes
    print(f"\n{category} COMPLETE: {successes} successes, {failures} failures")
    return results


# ============================================================
# Phase 1: Real Articles (190)
# ============================================================

def prepare_phase1_tasks() -> list:
    """Prepare task data for Phase 1."""
    articles = []
    with open(DATA_RAW / "articles_train.jsonl") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    articles = articles[:190]
    tasks = []
    
    for idx, article in enumerate(articles):
        prompt = f"""You are a senior quantitative analyst specializing in the quantum computing sector. Analyze this article and produce a cross-sectional trading signal vector.

**IMPORTANT: Research the context thoroughly.** Look up the companies mentioned, check their stock performance around the article date ({article['date']}), verify claims, and understand competitive dynamics. Use web browsing.

{SHARED_CONTEXT}

**Your chain of thought MUST include:**
1. What is this article actually saying? (separate fact from hype)
2. Which technology approach does this relate to?
3. How significant is this relative to the company's roadmap?
4. How quickly will the market price this in?
5. What are the second-order effects on competitors?

---

**Article:**
Title: {article['title']}
Source: {article['source']}
Date: {article['date']}

{article['text']}"""
        
        tasks.append({
            "idx": idx,
            "prompt": prompt,
            "metadata": {
                "title": article["title"],
                "date": article["date"],
                "source": article["source"],
                "text": article["text"]
            }
        })
    
    return tasks


# ============================================================
# Phase 3: Synthetic Articles (200)
# ============================================================

# Import scenarios from the original pipeline
from manus_teacher_pipeline import SYNTHETIC_SCENARIOS

def prepare_phase3_tasks() -> list:
    """Prepare task data for Phase 3 (Synthetic)."""
    scenarios = SYNTHETIC_SCENARIOS[:200]
    tasks = []
    
    for idx, scenario in enumerate(scenarios):
        prompt = f"""You are a financial journalist AND a quantitative analyst. Your task has two parts:

**Part 1: Generate a realistic news article** about the following quantum computing scenario. The article should read like a real financial news piece (Reuters/Bloomberg style) with specific details, quotes, and context. Make it 150-300 words.

**Scenario:** {scenario}

**Part 2: Analyze the article you just wrote** and produce a cross-sectional trading signal vector for all 9 quantum computing tickers.

{SHARED_CONTEXT}

**Your chain of thought should explain your reasoning for each ticker's score.**"""
        
        tasks.append({
            "idx": idx,
            "prompt": prompt,
            "metadata": {"scenario": scenario}
        })
    
    return tasks


# ============================================================
# Phase 4: Paraphrased Articles (190)
# ============================================================

PARAPHRASE_STYLES = [
    "Formal SEC filing language",
    "Casual tech blog post",
    "Twitter/X thread (series of short posts)",
    "Analyst research note",
    "Reddit r/investing discussion post",
]

def prepare_phase4_tasks() -> list:
    """Prepare task data for Phase 4 (Paraphrased)."""
    articles = []
    with open(DATA_RAW / "articles_train.jsonl") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    articles = articles[:190]
    tasks = []
    
    for idx, article in enumerate(articles):
        style = PARAPHRASE_STYLES[idx % len(PARAPHRASE_STYLES)]
        
        prompt = f"""You are a financial content editor AND a quantitative analyst. Your task has two parts:

**Part 1: Rewrite the following article** in a completely different style. Choose this style:
- {style}

The rewritten version must convey the SAME factual information but in a completely different tone, structure, and vocabulary. Do NOT add new information that wasn't in the original.

**Part 2: Analyze the rewritten article** and produce a cross-sectional trading signal vector. The signal should be IDENTICAL or very close to what you'd produce for the original article (since the underlying facts are the same).

{SHARED_CONTEXT}

**Your chain of thought should note that the style is different but the underlying signal is the same, and explain why.**

---

**Original article to rewrite:**
Title: {article['title']}
Source: {article['source']}
Date: {article['date']}

{article['text']}"""
        
        tasks.append({
            "idx": idx,
            "prompt": prompt,
            "metadata": {
                "original_title": article["title"],
                "original_date": article["date"],
                "style": style
            }
        })
    
    return tasks


# ============================================================
# Phase 5: Negative Examples (150)
# ============================================================

NEGATIVE_TOPICS = [
    "classical computing chip release", "cloud services announcement", "AI language model release",
    "semiconductor earnings report", "general stock market news", "renewable energy breakthrough",
    "biotech drug approval", "cryptocurrency regulation", "electric vehicle news",
    "social media platform update", "cybersecurity breach", "5G network deployment",
    "autonomous driving milestone", "space exploration news", "fintech IPO",
    "streaming service launch", "e-commerce earnings", "robotics advancement",
    "climate change policy", "supply chain disruption", "merger and acquisition",
    "central bank interest rate decision", "inflation data release", "employment report",
    "oil price movement", "real estate market update", "banking sector news",
    "healthcare technology", "agricultural technology", "gaming industry news",
]

def prepare_phase5_tasks() -> list:
    """Prepare task data for Phase 5 (Negatives)."""
    tasks = []
    
    for idx in range(150):
        topic = NEGATIVE_TOPICS[idx % len(NEGATIVE_TOPICS)]
        start_year = 2024 + (idx % 3)
        start_month = 1 + (idx % 12)
        start_date = f"{start_year}-{start_month:02d}-01"
        end_date = f"{start_year}-{start_month:02d}-28"
        
        prompt = f"""You are a quantitative analyst specializing in the quantum computing sector. Your task has two parts:

**Part 1: Find a real news article** from the web that is about technology or finance but is NOT related to quantum computing. Find an article about: {topic}

Browse the web and find a real article published between {start_date} and {end_date}. Copy its title and a 100-200 word summary.

**Part 2: Analyze this article** as if it were submitted to your quantum computing signal system. Since it is NOT about quantum computing, ALL ticker scores should be 0.0 (or very close to zero). Explain in your reasoning why each ticker is unaffected.

{SHARED_CONTEXT}

**IMPORTANT: All scores MUST be 0.0 for this article. The chain of thought should explain why this article has no relevance to the quantum computing sector.**"""
        
        tasks.append({
            "idx": idx,
            "prompt": prompt,
            "metadata": {"topic": topic, "date_range": f"{start_date} to {end_date}"}
        })
    
    return tasks


# ============================================================
# Phase 6: Edge Cases (100)
# ============================================================

from manus_teacher_pipeline import EDGE_CASES

def prepare_phase6_tasks() -> list:
    """Prepare task data for Phase 6 (Edge Cases)."""
    scenarios = EDGE_CASES[:100]
    tasks = []
    
    for idx, edge_case in enumerate(scenarios):
        prompt = f"""You are a senior quantitative analyst facing an ambiguous situation. Analyze the following scenario that has CONFLICTING or UNCLEAR implications for quantum computing stocks.

**Scenario:** {edge_case}

{SHARED_CONTEXT}

**IMPORTANT: This is intentionally ambiguous. Your chain of thought MUST:**
1. Acknowledge the ambiguity explicitly
2. Present arguments for both bullish and bearish interpretations
3. Explain which interpretation you weight more heavily and why
4. Assign scores that reflect your uncertainty (moderate scores, not extreme)
5. Note what additional information would resolve the ambiguity

**The scores should reflect genuine uncertainty — avoid defaulting to 0.0 just because it's ambiguous. Take a position, but a measured one.**"""
        
        tasks.append({
            "idx": idx,
            "prompt": prompt,
            "metadata": {"scenario": edge_case}
        })
    
    return tasks


# ============================================================
# Phase 7: Evaluation Predictions (421)
# ============================================================

def prepare_phase7_tasks() -> list:
    """Prepare task data for Phase 7 (Eval)."""
    articles = []
    with open(DATA_RAW / "articles_eval.jsonl") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    articles = articles[:421]
    tasks = []
    
    for idx, article in enumerate(articles):
        prompt = f"""You are a senior quantitative analyst. Analyze this article and produce a cross-sectional trading signal vector.

**CRITICAL CONSTRAINT: You are analyzing this article as if today's date is {article['date']} (the article's publication date). You MUST NOT:**
- Look up any information published after {article['date']}
- Check current stock prices or any price movements after {article['date']}
- Reference any events that occurred after {article['date']}
- Use web browsing to look up what happened to these stocks after this date

**Base your analysis SOLELY on:**
- The article text provided below
- Your pre-existing knowledge about these companies and quantum computing technology
- General market dynamics and sector relationships

{SHARED_CONTEXT}

**Your chain of thought should reason about the article's implications WITHOUT referencing any future events.**

---

**Article (analyze as if today is {article['date']}):**
Title: {article['title']}
Source: {article['source']}
Date: {article['date']}

{article['text']}"""
        
        tasks.append({
            "idx": idx,
            "prompt": prompt,
            "metadata": {
                "title": article["title"],
                "date": article["date"],
                "source": article["source"],
                "text": article["text"]
            }
        })
    
    return tasks


# ============================================================
# Phase 2: Multi-Turn Follow-ups (170)
# ============================================================

FOLLOWUP_QUESTIONS = [
    "Why did you score {ticker} as {score}? What specific evidence from the article supports this? What would make you change this score?",
    "If this article had been published 6 months earlier, would your scores be different? Why or why not?",
    "What is the single most important piece of information in this article that drives the largest score? If that fact turned out to be wrong, how would all scores change?",
    "Imagine a follow-up article published 2 weeks later says the claims in this article were exaggerated. How would you update your signal vector?",
    "A trader asks: 'Should I act on this signal today or wait?' What would you advise and why? Consider the time_horizon and signal_decay you assigned.",
    "Rank the 9 tickers from most affected to least affected by this news. For the top 3, explain the causal chain from article to stock impact.",
    "What information is MISSING from this article that would significantly change your analysis if you had it?",
]


async def run_phase2_multi_turn(max_concurrent: int = MAX_CONCURRENT):
    """Run multi-turn follow-ups on Phase 1 results."""
    output_file = DATA_TRAINING / "manus_multi_turn.jsonl"
    completed = get_completed_indices(output_file)
    
    # Load Phase 1 results
    phase1_file = DATA_TRAINING / "manus_real_articles.jsonl"
    phase1_results = load_existing_results(phase1_file)
    successful = [r for r in phase1_results if r.get("success") and r.get("task_id")]
    
    if not successful:
        print("ERROR: No successful Phase 1 results with task_ids. Run Phase 1 first.")
        return
    
    to_process = successful[:170]
    pending = [(i, r) for i, r in enumerate(to_process) if i not in completed]
    
    print(f"\n{'='*60}")
    print(f"PHASE 2: Multi-Turn Follow-ups")
    print(f"Total: {len(to_process)} | Already done: {len(completed)} | Pending: {len(pending)}")
    print(f"{'='*60}\n")
    
    if not pending:
        print("All follow-ups already completed!")
        return
    
    rate_limiter = RateLimiter(max_concurrent)
    connector = aiohttp.TCPConnector(limit=max_concurrent + 10)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = []
        for idx, phase1_result in pending:
            tasks.append(_process_followup(session, idx, phase1_result, output_file, rate_limiter))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    print(f"\nPhase 2 COMPLETE: {successes} successes, {len(results) - successes} failures")


async def _process_followup(session, idx, phase1_result, output_file, rate_limiter):
    """Process a single follow-up task."""
    async with rate_limiter.semaphore:
        start_time = time.time()
        
        task_id = phase1_result.get("task_id")
        signal = phase1_result.get("signal", {})
        
        # Pick follow-up question
        q_template = FOLLOWUP_QUESTIONS[idx % len(FOLLOWUP_QUESTIONS)]
        if "{ticker}" in q_template and signal:
            sv = signal.get("signal_vector", {})
            if sv:
                max_ticker = max(sv.keys(), key=lambda t: abs(sv[t].get("score", 0)))
                max_score = sv[max_ticker]["score"]
                question = q_template.format(ticker=max_ticker, score=max_score)
            else:
                question = q_template.replace("{ticker}", "IONQ").replace("{score}", "0.0")
        else:
            question = q_template
        
        print(f"  [followup][{idx}] Sending follow-up to task {task_id}...")
        
        success = await send_message_async(session, task_id, question, FOLLOWUP_SCHEMA)
        if not success:
            record = {
                "article_idx": idx, "category": "multi_turn",
                "original_task_id": task_id, "follow_up_question": question,
                "success": False, "error": "Failed to send follow-up",
                "timestamp": datetime.now().isoformat(), "time_seconds": time.time() - start_time
            }
            await append_result_async(output_file, record)
            return record
        
        result = await poll_for_result_async(session, task_id)
        elapsed = time.time() - start_time
        
        record = {
            "article_idx": idx,
            "category": "multi_turn",
            "original_article_idx": phase1_result.get("article_idx"),
            "original_title": phase1_result.get("title", phase1_result.get("metadata", {}).get("title", "")),
            "original_signal": signal,
            "follow_up_question": question,
            "task_id": task_id,
            "success": result.get("success", False),
            "follow_up_response": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "time_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }
        
        await append_result_async(output_file, record)
        rate_limiter.report_success()
        
        status = "SUCCESS" if record["success"] else "FAILED"
        print(f"  [followup][{idx}] {status} ({elapsed:.0f}s)")
        return record


# ============================================================
# Aggregation
# ============================================================

def run_aggregation():
    """Combine all training files and report statistics."""
    print("\n" + "="*60)
    print("FINAL AGGREGATION")
    print("="*60)
    
    combined_file = DATA_TRAINING / "manus_teacher_combined.jsonl"
    total_count = 0
    success_count = 0
    
    training_files = [
        ("Real Articles", DATA_TRAINING / "manus_real_articles.jsonl"),
        ("Multi-Turn", DATA_TRAINING / "manus_multi_turn.jsonl"),
        ("Synthetic", DATA_TRAINING / "manus_synthetic.jsonl"),
        ("Paraphrased", DATA_TRAINING / "manus_paraphrased.jsonl"),
        ("Negatives", DATA_TRAINING / "manus_negatives.jsonl"),
        ("Edge Cases", DATA_TRAINING / "manus_edge_cases.jsonl"),
    ]
    
    with open(combined_file, "w") as out:
        for name, tf in training_files:
            if tf.exists():
                with open(tf) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            total_count += 1
                            if data.get("success"):
                                success_count += 1
                            out.write(line)
    
    print(f"\nTraining Data Statistics:")
    print(f"  Total examples: {total_count}")
    print(f"  Successful: {success_count}")
    print(f"  Success rate: {success_count/max(total_count,1)*100:.1f}%")
    print()
    
    for name, tf in training_files:
        if tf.exists():
            with open(tf) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            successes = sum(1 for l in lines if l.get("success"))
            avg_time = 0
            times = [l.get("time_seconds", 0) for l in lines if l.get("time_seconds")]
            if times:
                avg_time = sum(times) / len(times)
            print(f"  {name}: {len(lines)} total, {successes} successful, avg {avg_time:.0f}s/task")
    
    # Eval predictions
    eval_file = DATA_EVAL / "predictions_manus_teacher.jsonl"
    if eval_file.exists():
        with open(eval_file) as f:
            eval_lines = [json.loads(l) for l in f if l.strip()]
        eval_successes = sum(1 for l in eval_lines if l.get("success"))
        print(f"\nEvaluation Predictions:")
        print(f"  Total: {len(eval_lines)}")
        print(f"  Successful: {eval_successes}")
        print(f"  Success rate: {eval_successes/max(len(eval_lines),1)*100:.1f}%")
    
    print(f"\nCombined file: {combined_file}")


# ============================================================
# Main
# ============================================================

async def async_main(phase: str, max_concurrent: int):
    """Main async entry point."""
    
    if phase == "1":
        tasks = prepare_phase1_tasks()
        await run_batch_concurrent(tasks, SIGNAL_SCHEMA, 
                                   DATA_TRAINING / "manus_real_articles.jsonl",
                                   "real_articles", max_concurrent)
    
    elif phase == "2":
        await run_phase2_multi_turn(max_concurrent)
    
    elif phase == "3":
        tasks = prepare_phase3_tasks()
        await run_batch_concurrent(tasks, SIGNAL_SCHEMA,
                                   DATA_TRAINING / "manus_synthetic.jsonl",
                                   "synthetic", max_concurrent)
    
    elif phase == "4":
        tasks = prepare_phase4_tasks()
        await run_batch_concurrent(tasks, SIGNAL_SCHEMA,
                                   DATA_TRAINING / "manus_paraphrased.jsonl",
                                   "paraphrased", max_concurrent)
    
    elif phase == "5":
        tasks = prepare_phase5_tasks()
        await run_batch_concurrent(tasks, SIGNAL_SCHEMA,
                                   DATA_TRAINING / "manus_negatives.jsonl",
                                   "negatives", max_concurrent)
    
    elif phase == "6":
        tasks = prepare_phase6_tasks()
        await run_batch_concurrent(tasks, SIGNAL_SCHEMA,
                                   DATA_TRAINING / "manus_edge_cases.jsonl",
                                   "edge_cases", max_concurrent)
    
    elif phase == "7":
        tasks = prepare_phase7_tasks()
        await run_batch_concurrent(tasks, SIGNAL_SCHEMA,
                                   DATA_EVAL / "predictions_manus_teacher.jsonl",
                                   "eval_predictions", max_concurrent)
    
    elif phase == "training":
        # Run phases 1, 3, 4, 5, 6 concurrently (all independent)
        print("Running all training phases (1, 3, 4, 5, 6) concurrently...")
        
        all_tasks = []
        phase_configs = [
            (prepare_phase1_tasks(), DATA_TRAINING / "manus_real_articles.jsonl", "real_articles"),
            (prepare_phase3_tasks(), DATA_TRAINING / "manus_synthetic.jsonl", "synthetic"),
            (prepare_phase4_tasks(), DATA_TRAINING / "manus_paraphrased.jsonl", "paraphrased"),
            (prepare_phase5_tasks(), DATA_TRAINING / "manus_negatives.jsonl", "negatives"),
            (prepare_phase6_tasks(), DATA_TRAINING / "manus_edge_cases.jsonl", "edge_cases"),
        ]
        
        # Run each phase sequentially but tasks within each phase concurrently
        for tasks, output_file, category in phase_configs:
            await run_batch_concurrent(tasks, SIGNAL_SCHEMA, output_file, category, max_concurrent)
    
    elif phase == "all":
        # Run training phases first
        phase_configs = [
            (prepare_phase1_tasks(), DATA_TRAINING / "manus_real_articles.jsonl", "real_articles"),
            (prepare_phase3_tasks(), DATA_TRAINING / "manus_synthetic.jsonl", "synthetic"),
            (prepare_phase4_tasks(), DATA_TRAINING / "manus_paraphrased.jsonl", "paraphrased"),
            (prepare_phase5_tasks(), DATA_TRAINING / "manus_negatives.jsonl", "negatives"),
            (prepare_phase6_tasks(), DATA_TRAINING / "manus_edge_cases.jsonl", "edge_cases"),
        ]
        
        for tasks, output_file, category in phase_configs:
            await run_batch_concurrent(tasks, SIGNAL_SCHEMA, output_file, category, max_concurrent)
        
        # Phase 2 depends on Phase 1
        await run_phase2_multi_turn(max_concurrent)
        
        # Phase 7 (eval)
        tasks = prepare_phase7_tasks()
        await run_batch_concurrent(tasks, SIGNAL_SCHEMA,
                                   DATA_EVAL / "predictions_manus_teacher.jsonl",
                                   "eval_predictions", max_concurrent)
        
        # Aggregation
        run_aggregation()


def main():
    parser = argparse.ArgumentParser(description="Manus Teacher Pipeline (Concurrent)")
    parser.add_argument("--phase", type=str, default="all",
                       help="Phase: 1-7, 'training' (1,3,4,5,6), or 'all'")
    parser.add_argument("--max-concurrent", type=int, default=MAX_CONCURRENT,
                       help=f"Max concurrent tasks (default: {MAX_CONCURRENT})")
    parser.add_argument("--aggregate", action="store_true",
                       help="Run aggregation only")
    args = parser.parse_args()
    
    if args.aggregate:
        run_aggregation()
        return
    
    print(f"Manus Teacher Pipeline (Concurrent)")
    print(f"Max concurrent tasks: {args.max_concurrent}")
    print(f"Agent profile: max")
    print(f"Phase: {args.phase}")
    print()
    
    asyncio.run(async_main(args.phase, args.max_concurrent))


if __name__ == "__main__":
    main()
