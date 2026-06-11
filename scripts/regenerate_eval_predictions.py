"""
Regenerate evaluation predictions using the updated prompt (v4).

CRITICAL: No future information leakage.
- Market context uses ONLY data up to the article's publication date
- Prompt explicitly forbids looking up anything after the article date
- No reference to what happened to stocks after the article

Uses:
- Updated ticker universe (MSFT/GOOGL/NVDA inactive, QNT added)
- Tech-validation vs competitive-threat rule
- Minimum conviction threshold
- ArXiv cap rules
- Market context (historical only, up to article date)

Usage:
    python scripts/regenerate_eval_predictions.py
"""

import asyncio
import aiohttp
import json
import time
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import SHARED_CONTEXT, SIGNAL_SCHEMA
from src.market_context import get_market_context

# ============================================================
# Configuration
# ============================================================

API_KEY = "sk-3oFNwU2aOnhvss4PtIMzRIEX5XvFMyLjkV8BTw_kAMkZKVAHu9HKMD3NIIe2Wuwt3XnWF_Qh3ppmJ_qZ_z7CsuFoqTYq"
BASE_URL = "https://api.manus.ai/v2"
HEADERS = {"x-manus-api-key": API_KEY, "Content-Type": "application/json"}

MAX_CONCURRENT = 10
CREATION_DELAY = 2.0
POLL_INTERVAL = 30
MAX_POLL_TIME = 900

DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"
MARKET_DIR = PROJECT_ROOT / "data" / "market"

OUTPUT_FILE = DATA_EVAL / "predictions_manus_teacher_v2.jsonl"

# ============================================================
# Eval-specific prompt (NO future information)
# ============================================================

EVAL_PROMPT_TEMPLATE = """You are a senior quantitative analyst. Analyze this article and produce a cross-sectional trading signal vector.

**CRITICAL CONSTRAINT: You are analyzing this article as if today's date is {date} (the article's publication date). You MUST NOT:**
- Look up any information published after {date}
- Check current stock prices or any price movements after {date}
- Reference any events that occurred after {date}
- Use web browsing to look up what happened to these stocks after this date

**Base your analysis SOLELY on:**
- The article text provided below
- Your pre-existing knowledge about these companies and quantum computing technology
- General market dynamics and sector relationships
- The market context provided (which only includes data UP TO {date})

{shared_context}

{market_context}

**Your chain of thought should reason about the article's implications WITHOUT referencing any future events.**

---

**Article (analyze as if today is {date}):**
Title: {title}
Source: {source}
Date: {date}

{text}"""

EVAL_ARXIV_PROMPT_TEMPLATE = """You are a senior quantitative analyst. Analyze this academic paper and produce a cross-sectional trading signal vector.

**CRITICAL CONSTRAINT: You are analyzing this paper as if today's date is {date}. You MUST NOT look up anything published after {date}.**

**ArXiv-specific rules:**
- Most academic papers do NOT move stocks. Default to 0.0 unless clear commercial implications.
- Maximum absolute score: 0.5 (unless company-authored hardware result, then up to 1.0)
- Pure theory papers: all scores 0.0
- Incremental improvements: all scores 0.0

{shared_context}

{market_context}

---

**Paper (analyze as if today is {date}):**
Title: {title}
Source: arxiv
Date: {date}

{text}"""


# ============================================================
# Async API (reused pattern)
# ============================================================

_file_lock = asyncio.Lock()


async def append_result(filepath, record):
    async with _file_lock:
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")


async def create_task(session, prompt, schema, retries=3):
    for attempt in range(retries):
        try:
            payload = {
                "message": {"content": prompt},
                "structured_output_schema": schema,
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
                for msg in data.get("messages", []):
                    if msg.get("type") == "structured_output_result":
                        return msg["structured_output_result"]
                for msg in data.get("messages", []):
                    if msg.get("type") == "status_update":
                        status = msg.get("status_update", {}).get("agent_status")
                        if status == "stopped":
                            params2 = {"task_id": task_id, "order": "asc"}
                            async with session.get(f"{BASE_URL}/task.listMessages", headers=HEADERS, params=params2) as r2:
                                d2 = await r2.json()
                                for m in d2.get("messages", []):
                                    if m.get("type") == "structured_output_result":
                                        return m["structured_output_result"]
                            return {"success": False, "error": "Stopped without output"}
                        elif status == "error":
                            return {"success": False, "error": "Task errored"}
        except Exception as e:
            print(f"  [WARN] Poll: {e}")
        await asyncio.sleep(POLL_INTERVAL)
    return {"success": False, "error": "Timeout"}


async def process_article(session, semaphore, idx, article):
    """Process a single eval article."""
    async with semaphore:
        start_time = time.time()
        
        date = article["date"]
        source = article.get("source", "news")
        title = article.get("title", "")
        text = article.get("text", "")
        
        # Compute market context UP TO the article date only (no future leakage)
        market_context = get_market_context(date, market_dir=MARKET_DIR)
        
        # Select prompt template based on source
        if source == "arxiv":
            prompt = EVAL_ARXIV_PROMPT_TEMPLATE.format(
                date=date,
                shared_context=SHARED_CONTEXT,
                market_context=market_context if market_context else "Market context: unavailable for this date.",
                title=title,
                source=source,
                text=text
            )
        else:
            prompt = EVAL_PROMPT_TEMPLATE.format(
                date=date,
                shared_context=SHARED_CONTEXT,
                market_context=market_context if market_context else "Market context: unavailable for this date.",
                title=title,
                source=source,
                text=text
            )
        
        print(f"  [eval][{idx}] Starting: {title[:50]}...")
        
        task_id = await create_task(session, prompt, SIGNAL_SCHEMA)
        if not task_id:
            record = {
                "article_idx": idx, "date": date, "title": title, "source": source,
                "status": "error", "error": "Failed to create task",
                "time_seconds": time.time() - start_time,
                "timestamp": datetime.now().isoformat()
            }
            await append_result(OUTPUT_FILE, record)
            return record
        
        result = await poll_result(session, task_id)
        elapsed = time.time() - start_time
        
        success = result.get("success", False)
        signal = result.get("value") if success else None
        
        record = {
            "article_idx": idx,
            "date": date,
            "title": title,
            "source": source,
            "status": "success" if success else "error",
            "signal": signal,
            "error": result.get("error"),
            "task_id": task_id,
            "time_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }
        
        await append_result(OUTPUT_FILE, record)
        
        status_str = "SUCCESS" if success else "FAILED"
        print(f"  [eval][{idx}] {status_str} ({elapsed:.0f}s)")
        return record


async def main():
    # Load eval articles
    eval_file = DATA_RAW / "articles_eval.jsonl"
    articles = []
    with open(eval_file) as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    articles = articles[:421]
    
    # Check existing progress (resume support)
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.add(r.get("article_idx"))
    
    pending = [(i, a) for i, a in enumerate(articles) if i not in existing]
    
    print(f"Regenerating eval predictions with updated prompt (v4)")
    print(f"Total articles: {len(articles)}")
    print(f"Already done: {len(existing)}")
    print(f"Pending: {len(pending)}")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"Output: {OUTPUT_FILE}")
    print(f"Agent profile: manus-1.6-max")
    print()
    print("ANTI-CHEATING MEASURES:")
    print("  - Market context uses ONLY data up to article date")
    print("  - Prompt forbids looking up anything after article date")
    print("  - No web browsing for future information")
    print()
    
    if not pending:
        print("All predictions already generated!")
        return
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [process_article(session, semaphore, idx, article) for idx, article in pending]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "success")
    print(f"\nComplete: {successes}/{len(pending)} successful")
    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    asyncio.run(main())
