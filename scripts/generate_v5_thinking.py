"""
Generate V5 training data with <think> reasoning traces.

V5 spec:
- Thinking block (100-300 tokens) drives the scores
- Scoring philosophy: probability of reaching quantum supremacy milestones
- Simplified market context (one-line 5d returns)
- ArXiv papers must be actually read
- All tasks go to "Training Tasks" project

Usage:
    python scripts/generate_v5_thinking.py
    python scripts/generate_v5_thinking.py --start 0 --end 50
"""

import asyncio
import aiohttp
import json
import time
import argparse
import sys
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.market_context import get_market_context

# ============================================================
# Configuration
# ============================================================

API_KEY = "sk-3oFNwU2aOnhvss4PtIMzRIEX5XvFMyLjkV8BTw_kAMkZKVAHu9HKMD3NIIe2Wuwt3XnWF_Qh3ppmJ_qZ_z7CsuFoqTYq"
BASE_URL = "https://api.manus.ai/v2"
HEADERS = {"x-manus-api-key": API_KEY, "Content-Type": "application/json"}
PROJECT_ID = "3uqozQ7JNK7mUnFKnnXNj4"  # "Training Tasks" project

MAX_CONCURRENT = 10
CREATION_DELAY = 2.0
POLL_INTERVAL = 30
MAX_POLL_TIME = 1200

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
MARKET_DIR = PROJECT_ROOT / "data" / "market"
OUTPUT_FILE = DATA_TRAINING / "alpha_signal_train_v5_raw.jsonl"

# ============================================================
# V5 System Prompt
# ============================================================

V5_SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must THINK step-by-step and then produce a signal vector scoring all companies in the quantum computing universe.

**OUTPUT FORMAT (MANDATORY):**
You MUST output a <think>...</think> block followed immediately by a JSON object. No other text.

<think>
[Your genuine step-by-step reasoning, 100-300 tokens. This must drive your scores.]
</think>
{"signal_vector": {...}, "event_type": "...", ...}

**THINKING STRUCTURE (guide, not rigid):**
1. What is the core event or finding?
2. What technology does this relate to? (trapped-ion, superconducting, annealing, neutral-atom, photonic, topological)
3. How does this advance (or hinder) the path toward quantum supremacy for the relevant companies?
4. Which companies benefit? Which face competitive pressure?
5. What are the second-order effects across the sector?
6. How novel is this? Is it incremental or a genuine milestone?
7. Calibration: Am I within score ranges? Does my reasoning justify the magnitude?

**YOUR THINKING MUST DRIVE YOUR SCORES.** The JSON scores must be logically consistent with your reasoning.

---

**QUANTUM COMPUTING UNIVERSE (10 tickers):**

Active (assign scores based on reasoning):
- IONQ: IonQ (trapped-ion, pure-play, 100% quantum revenue)
- RGTI: Rigetti Computing (superconducting, pure-play, 100% quantum revenue)
- QBTS: D-Wave Quantum (quantum annealing, pure-play, 100% quantum revenue)
- QUBT: Quantum Computing Inc. (neutral atom, pure-play, 100% quantum revenue)
- QNT: Quantinuum (trapped-ion, pure-play, 100% quantum revenue, IPO'd June 2026)
- IBM: International Business Machines (superconducting, ~2% quantum revenue)
- HON: Honeywell (trapped-ion, ~1% quantum revenue post-Quantinuum spinoff)

Inactive (always 0.0, but reason about their impact on active tickers):
- MSFT: Microsoft (topological approach) — score 0.0 but their progress affects others
- GOOGL: Alphabet/Google (superconducting) — score 0.0 but their breakthroughs validate/threaten others
- NVDA: NVIDIA (quantum enabler) — score 0.0

**SCORE RANGES (STRICT — scores are decimal numbers, NOT percentages):**
- Pure-play (IONQ, RGTI, QBTS, QUBT, QNT): minimum -2.0, maximum +2.0. Example scores: +0.5, -1.2, +2.0, -0.3
- HON: minimum -0.3, maximum +0.3. Example scores: +0.2, -0.15, +0.3
- IBM: minimum -0.15, maximum +0.15. Example scores: +0.1, -0.05, +0.15
- MSFT: ALWAYS exactly 0.0. No exceptions.
- GOOGL: ALWAYS exactly 0.0. No exceptions.
- NVDA: ALWAYS exactly 0.0. No exceptions.

Scores are NOT percentages. A score of +1.5 means "strongly bullish". A score of +50 or +78 is WRONG.
The scale is: 0.0 = no opinion, +/-0.5 = mild, +/-1.0 = moderate, +/-1.5 = strong, +/-2.0 = maximum conviction.

**SCORING PHILOSOPHY:**
Scores reflect how the news is expected to move the stock price over the next 5 trading days:
- Does this news change investor expectations about the company's revenue trajectory, competitive position, or technology validation?
- Does this validate or invalidate the company's technology approach in a way the market will reprice?
- Does this shift the competitive timeline between companies in a way that affects relative valuations?
- Does this affect the company's ability to attract talent, funding, or partnerships that the market cares about?
- Consider: milestones on the path to fault-tolerant quantum computing are what drive these stocks. Progress toward that goal is what investors are pricing.
The score should reflect expected stock movement, not abstract technological assessment.

**TECHNOLOGY VALIDATION vs COMPETITIVE THREAT:**
When a large company (Google, IBM, Microsoft) achieves a technology breakthrough:
- This VALIDATES the approach → BULLISH for smaller same-technology pure-plays
- Empirical evidence: Google Willow (superconducting) → RGTI surged +89% in 5 days
- Only score same-tech competitors BEARISH when it's a zero-sum business win (contract, exclusive deal)

**IONQ-QNT COMPETITIVE DYNAMIC:**
- Both are trapped-ion pure-plays competing for the same customers
- Sector-wide events: both move together
- Company-specific wins: one gains at the other's expense

**MINIMUM CONVICTION (source-aware):**
- News about quantum companies: at least one pure-play should almost always get non-zero
- ArXiv papers: default to 0.0 unless genuine commercial breakthrough with measured hardware results
- Non-quantum content: all scores 0.0

**ArXiv RULES:**
- Maximum score: 1.0 (up to 2.0 only for company-authored major breakthrough)
- You MUST read the paper content to assess significance, not just the title
- Most papers are incremental → all scores 0.0
- Only genuine hardware milestones with measured metrics warrant non-zero

**JSON STRUCTURE:**
{
    "signal_vector": {
        "IONQ": {"score": float, "reasoning": "1-2 sentences"},
        "RGTI": {"score": float, "reasoning": "1-2 sentences"},
        "QBTS": {"score": float, "reasoning": "1-2 sentences"},
        "QUBT": {"score": float, "reasoning": "1-2 sentences"},
        "QNT": {"score": float, "reasoning": "1-2 sentences"},
        "IBM": {"score": float, "reasoning": "1-2 sentences"},
        "HON": {"score": float, "reasoning": "1-2 sentences"},
        "MSFT": {"score": 0.0, "reasoning": "Inactive"},
        "GOOGL": {"score": 0.0, "reasoning": "Inactive"},
        "NVDA": {"score": 0.0, "reasoning": "Inactive"}
    },
    "event_type": "descriptive category",
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "information_novelty": "high" | "medium" | "low",
    "technical_translation": "2-3 sentences explaining commercial significance for a portfolio manager",
    "signal_rationale": "Why these specific scores? What reasoning drove this distribution?",
    "chain_of_thought": "Brief summary of your reasoning process"
}"""


# ============================================================
# Market Context (full table format)
# ============================================================

def get_full_market_context(date: str) -> str:
    """Get full market context table for the user message."""
    if not date:
        return ""
    return get_market_context(date, market_dir=MARKET_DIR)


# ============================================================
# User message builder
# ============================================================

def build_user_message(record: dict) -> str:
    """Build user message with full market context table."""
    parts = []
    
    # Market context (full table)
    date = record.get("date", "")
    if date:
        ctx = get_full_market_context(date)
        if ctx:
            parts.append(ctx)
            parts.append("")
    
    # Source-specific instruction
    source = record.get("source", "news")
    arxiv_tier = record.get("arxiv_tier", "")
    if source == "arxiv" or arxiv_tier:
        parts.append("[ARXIV PAPER - Read the content carefully. Most papers warrant all-zero scores.]")
    else:
        parts.append("[ARTICLE]")
    
    # Content
    if record.get("title"):
        parts.append(f"Title: {record['title']}")
    if record.get("date"):
        parts.append(f"Date: {record['date']}")
    if record.get("source"):
        parts.append(f"Source: {record['source']}")
    parts.append("")
    if record.get("text"):
        parts.append(record["text"])
    elif record.get("scenario"):
        parts.append(f"Scenario: {record['scenario']}")
    
    return "\n".join(parts)


# ============================================================
# Async API
# ============================================================

_file_lock = asyncio.Lock()


async def append_result(filepath, record):
    async with _file_lock:
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")


# V5 Structured Output Schema (includes thinking field)
V5_SCHEMA = {
    "type": "object",
    "properties": {
        "thinking": {"type": "string", "description": "Your step-by-step reasoning (100-300 tokens). Must address: 1) core event, 2) technology approach, 3) which companies benefit/suffer, 4) second-order effects, 5) novelty assessment, 6) conviction level, 7) score calibration check. Your scores MUST be consistent with this reasoning."},
        "signal_vector": {
            "type": "object",
            "properties": {
                "IONQ": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "RGTI": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "QBTS": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "QUBT": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "QNT": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "IBM": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "HON": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "MSFT": {"type": "object", "properties": {"score": {"type": "number", "description": "MUST be exactly 0.0"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "GOOGL": {"type": "object", "properties": {"score": {"type": "number", "description": "MUST be exactly 0.0"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "NVDA": {"type": "object", "properties": {"score": {"type": "number", "description": "MUST be exactly 0.0"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
            },
            "required": ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"],
            "additionalProperties": False
        },
        "event_type": {"type": "string"},
        "time_horizon": {"type": "string", "enum": ["intraday", "2-5 days", "1-2 weeks", "1+ month"]},
        "information_novelty": {"type": "string", "enum": ["high", "medium", "low"]},
        "technical_translation": {"type": "string"},
        "signal_rationale": {"type": "string"},
    },
    "required": ["thinking", "signal_vector", "event_type", "time_horizon", "information_novelty", "technical_translation", "signal_rationale"],
    "additionalProperties": False
}


async def create_task(session, user_msg, retries=3):
    for attempt in range(retries):
        try:
            payload = {
                "message": {"content": user_msg},
                "structured_output_schema": V5_SCHEMA,
                "project_id": PROJECT_ID,
                "agent_profile": "manus-1.6-max"
            }
            async with session.post(f"{BASE_URL}/task.create", headers=HEADERS, json=payload) as resp:
                if resp.status == 429:
                    retry_after = int(resp.headers.get("Retry-After", 60))
                    print(f"  [429] Rate limited. Waiting {retry_after}s...")
                    await asyncio.sleep(retry_after)
                    continue
                data = await resp.json()
                if data.get("ok"):
                    await asyncio.sleep(CREATION_DELAY)
                    return data["task_id"]
                else:
                    msg = data.get("error", {}).get("message", "?")
                    print(f"  [WARN] Create failed: {msg}")
                    if attempt < retries - 1:
                        await asyncio.sleep(60)
        except Exception as e:
            print(f"  [ERROR] {e}")
            if attempt < retries - 1:
                await asyncio.sleep(60)
    return None


async def poll_result(session, task_id, max_time=MAX_POLL_TIME):
    """Poll for structured output result."""
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
                
                # Look for structured output result
                for msg in data.get("messages", []):
                    if msg.get("type") == "structured_output_result":
                        result = msg["structured_output_result"]
                        return result
                
                # Check for completion/error
                for msg in data.get("messages", []):
                    if msg.get("type") == "status_update":
                        status = msg.get("status_update", {}).get("agent_status")
                        if status == "stopped":
                            # Check all messages for structured output
                            params2 = {"task_id": task_id, "order": "asc", "limit": 50}
                            async with session.get(f"{BASE_URL}/task.listMessages", headers=HEADERS, params=params2) as r2:
                                d2 = await r2.json()
                                for m in d2.get("messages", []):
                                    if m.get("type") == "structured_output_result":
                                        return m["structured_output_result"]
                            return {"success": False, "error": "Stopped without structured output"}
                        elif status == "error":
                            return {"success": False, "error": "Task errored"}
        except Exception as e:
            print(f"  [WARN] Poll: {e}")
        await asyncio.sleep(POLL_INTERVAL)
    return {"success": False, "error": "Timeout"}


def parse_response(response: dict) -> dict:
    """Parse structured output response into thinking + signal."""
    if not response.get("success"):
        return {"thinking": "", "signal": None}
    
    value = response.get("value", {})
    thinking = value.pop("thinking", "")
    
    return {"thinking": thinking, "signal": value}


def postprocess_signal(signal: dict) -> dict:
    """Post-process signal to enforce score ranges and zero inactive tickers."""
    sv = signal.get("signal_vector", {})
    
    # Zero inactive tickers
    for t in ["MSFT", "GOOGL", "NVDA"]:
        if t in sv:
            sv[t] = {"score": 0.0, "reasoning": "Inactive: quantum revenue exposure too low for meaningful signal."}
    
    # Clip score ranges
    for t in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT"]:
        if t in sv:
            score = sv[t].get("score", 0)
            # If score is on wrong scale (>2 or <-2), normalize
            if abs(score) > 2.0:
                if abs(score) > 10:  # Likely on 0-100 scale
                    score = (score / 50.0) - 1.0  # Map 0-100 to -1 to +1, then scale
                    score = max(-2.0, min(2.0, score * 2))
                else:
                    score = max(-2.0, min(2.0, score))
            sv[t]["score"] = round(score, 2)
    
    if "IBM" in sv:
        score = sv["IBM"].get("score", 0)
        if abs(score) > 0.15:
            if abs(score) > 1:  # Wrong scale
                score = max(-0.15, min(0.15, score / 10.0))
            else:
                score = max(-0.15, min(0.15, score))
        sv["IBM"]["score"] = round(score, 2)
    
    if "HON" in sv:
        score = sv["HON"].get("score", 0)
        if abs(score) > 0.3:
            if abs(score) > 1:  # Wrong scale
                score = max(-0.3, min(0.3, score / 10.0))
            else:
                score = max(-0.3, min(0.3, score))
        sv["HON"]["score"] = round(score, 2)
    
    signal["signal_vector"] = sv
    return signal


def validate_signal(signal: dict) -> list:
    """Validate a signal dict against V5 spec."""
    issues = []
    if not signal:
        return ["No signal parsed"]
    
    sv = signal.get("signal_vector", {})
    expected = {"IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"}
    actual = set(sv.keys())
    
    if not actual.issuperset(expected):
        missing = expected - actual
        issues.append(f"Missing tickers: {missing}")
    
    # Check inactive are 0.0
    for t in ["MSFT", "GOOGL", "NVDA"]:
        if t in sv and sv[t].get("score", 0) != 0.0:
            issues.append(f"{t} score should be 0.0, got {sv[t].get('score')}")
    
    # Check ranges
    for t in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT"]:
        if t in sv and abs(sv[t].get("score", 0)) > 2.0:
            issues.append(f"{t} score {sv[t]['score']} exceeds [-2.0, 2.0]")
    if "IBM" in sv and abs(sv["IBM"].get("score", 0)) > 0.15:
        issues.append(f"IBM score {sv['IBM']['score']} exceeds [-0.15, 0.15]")
    if "HON" in sv and abs(sv["HON"].get("score", 0)) > 0.3:
        issues.append(f"HON score {sv['HON']['score']} exceeds [-0.3, 0.3]")
    
    return issues


async def process_example(session, semaphore, idx, record, total):
    """Process a single training example."""
    async with semaphore:
        start_time = time.time()
        
        user_msg = build_user_message(record)
        title = record.get("title", record.get("scenario", ""))[:50]
        print(f"  [{idx+1}/{total}] {title}...")
        
        task_id = await create_task(session, user_msg)
        if not task_id:
            result = {
                "idx": idx, "success": False, "error": "Failed to create task",
                "category": record.get("category", ""),
                "timestamp": datetime.now().isoformat()
            }
            await append_result(OUTPUT_FILE, result)
            return result
        
        response = await poll_result(session, task_id)
        elapsed = time.time() - start_time
        
        parsed = parse_response(response)
        
        if parsed["signal"]:
            parsed["signal"] = postprocess_signal(parsed["signal"])
            issues = validate_signal(parsed["signal"])
            result = {
                "idx": idx,
                "success": True,
                "thinking": parsed["thinking"],
                "signal": parsed["signal"],
                "validation_issues": issues,
                "task_id": task_id,
                "category": record.get("category", ""),
                "source": record.get("source", ""),
                "title": record.get("title", record.get("scenario", "")),
                "date": record.get("date", ""),
                "text": record.get("text", record.get("scenario", "")),
                "market_context": get_full_market_context(record.get("date", "")),
                "time_seconds": elapsed,
                "timestamp": datetime.now().isoformat()
            }
            await append_result(OUTPUT_FILE, result)
            think_len = len(parsed["thinking"].split())
            status = "SUCCESS" if not issues else f"WARN({len(issues)} issues)"
            print(f"  [{idx+1}/{total}] {status} ({elapsed:.0f}s, think={think_len} words)")
            return result
        else:
            result = {
                "idx": idx, "success": False,
                "error": response.get("error", "Failed to parse"),
                "task_id": task_id,
                "category": record.get("category", ""),
                "time_seconds": elapsed,
                "timestamp": datetime.now().isoformat()
            }
            await append_result(OUTPUT_FILE, result)
            print(f"  [{idx+1}/{total}] FAILED ({elapsed:.0f}s): {response.get('error', '?')}")
            return result


async def main(start_idx=0, end_idx=None):
    # Load all source data
    print("Loading source data...")
    all_records = []
    
    with open(DATA_TRAINING / "manus_teacher_combined.jsonl") as f:
        combined = [json.loads(l) for l in f if l.strip()]
    combined_success = [r for r in combined if r.get("success") and r.get("signal")]
    all_records.extend(combined_success)
    
    arxiv_file = DATA_TRAINING / "manus_arxiv_rebalance.jsonl"
    if arxiv_file.exists():
        with open(arxiv_file) as f:
            arxiv = [json.loads(l) for l in f if l.strip()]
        all_records.extend([r for r in arxiv if r.get("success") and r.get("signal")])
    
    qnt_file = DATA_TRAINING / "manus_qnt_examples.jsonl"
    if qnt_file.exists():
        with open(qnt_file) as f:
            qnt = [json.loads(l) for l in f if l.strip()]
        all_records.extend([r for r in qnt if r.get("success") and r.get("signal")])
    
    if end_idx is None:
        end_idx = len(all_records)
    
    records_to_process = all_records[start_idx:end_idx]
    
    # Check existing progress (resume support)
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.add(r.get("idx"))
    
    pending = [(i + start_idx, r) for i, r in enumerate(records_to_process) if (i + start_idx) not in existing]
    
    print(f"=" * 60)
    print(f"V5 TRAINING DATA GENERATION")
    print(f"=" * 60)
    print(f"Total records: {len(all_records)}")
    print(f"Processing range: [{start_idx}, {end_idx})")
    print(f"Already done: {len(existing)}")
    print(f"Pending: {len(pending)}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"Project: Training Tasks ({PROJECT_ID})")
    print(f"Output: {OUTPUT_FILE}")
    print(f"=" * 60)
    print()
    
    if not pending:
        print("All examples already generated!")
        return
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            process_example(session, semaphore, idx, record, len(pending))
            for idx, record in pending
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    with_thinking = sum(1 for r in results if isinstance(r, dict) and r.get("success") and r.get("thinking"))
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {successes}/{len(pending)} successful, {with_thinking} with thinking blocks")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    args = parser.parse_args()
    
    asyncio.run(main(args.start, args.end))
