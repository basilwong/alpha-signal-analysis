"""
Retry the 5 tasks that had major validation issues (score range violations).
Re-runs them with extra emphasis on score bounds.
"""

import asyncio
import aiohttp
import json
import time
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))
from manus_teacher_concurrent import (
    API_KEY, BASE_URL, HEADERS, SIGNAL_SCHEMA, SHARED_CONTEXT,
    DATA_TRAINING, create_task_async, poll_for_result_async,
    RateLimiter, validate_signal_format
)

# The 5 tasks that need retrying
RETRY_TASKS = [
    # Real articles: idx 33, 69, 114, 144
    {"file": "manus_real_articles.jsonl", "idx": 33},
    {"file": "manus_real_articles.jsonl", "idx": 69},
    {"file": "manus_real_articles.jsonl", "idx": 114},
    {"file": "manus_real_articles.jsonl", "idx": 144},
    # Paraphrased: idx 38
    {"file": "manus_paraphrased.jsonl", "idx": 38},
]

EXTRA_BOUNDS_EMPHASIS = """

**CRITICAL SCORE BOUNDS - YOU MUST RESPECT THESE:**
- IONQ, RGTI, QBTS, QUBT: [-2.0, +2.0]
- HON: [-0.3, +0.3] MAXIMUM
- IBM: [-0.15, +0.15] MAXIMUM
- NVDA: [-0.10, +0.10] MAXIMUM
- GOOGL, MSFT: [-0.05, +0.05] MAXIMUM

These bounds reflect revenue exposure. MSFT and GOOGL have <0.1% quantum revenue, so even major quantum news cannot move their score beyond +/-0.05. NVDA is capped at +/-0.10. Violating these bounds is INCORRECT."""


async def retry_task(session, record, rate_limiter):
    """Retry a single task with extra bounds emphasis."""
    category = record.get("category")
    
    if category == "real_articles":
        prompt = f"""You are a senior quantitative analyst specializing in the quantum computing sector. Analyze this article and produce a cross-sectional trading signal vector.

**IMPORTANT: Research the context thoroughly.** Look up the companies mentioned, check their stock performance around the article date ({record['date']}), verify claims, and understand competitive dynamics. Use web browsing.

{SHARED_CONTEXT}
{EXTRA_BOUNDS_EMPHASIS}

**Your chain of thought MUST include:**
1. What is this article actually saying? (separate fact from hype)
2. Which technology approach does this relate to?
3. How significant is this relative to the company's roadmap?
4. How quickly will the market price this in?
5. What are the second-order effects on competitors?

---

**Article:**
Title: {record['title']}
Source: {record['source']}
Date: {record['date']}

{record['text']}"""
    
    elif category == "paraphrased":
        # Load the original article
        articles = []
        with open(DATA_TRAINING.parent / "raw" / "articles_train.jsonl") as f:
            for line in f:
                if line.strip():
                    articles.append(json.loads(line))
        
        article = articles[record["article_idx"]]
        style = record.get("style", "Analyst research note")
        
        prompt = f"""You are a financial content editor AND a quantitative analyst. Your task has two parts:

**Part 1: Rewrite the following article** in a completely different style. Choose this style:
- {style}

The rewritten version must convey the SAME factual information but in a completely different tone, structure, and vocabulary. Do NOT add new information that wasn't in the original.

**Part 2: Analyze the rewritten article** and produce a cross-sectional trading signal vector. The signal should be IDENTICAL or very close to what you'd produce for the original article (since the underlying facts are the same).

{SHARED_CONTEXT}
{EXTRA_BOUNDS_EMPHASIS}

**Your chain of thought should note that the style is different but the underlying signal is the same, and explain why.**

---

**Original article to rewrite:**
Title: {article['title']}
Source: {article['source']}
Date: {article['date']}

{article['text']}"""
    
    else:
        print(f"  Unknown category: {category}")
        return None
    
    async with rate_limiter.semaphore:
        start_time = time.time()
        task_id = await create_task_async(session, prompt, SIGNAL_SCHEMA, rate_limiter)
        if not task_id:
            return {"success": False, "error": "Failed to create task"}
        
        print(f"  Task created: {task_id}")
        result = await poll_for_result_async(session, task_id)
        elapsed = time.time() - start_time
        
        success = result.get("success", False)
        signal = result.get("value") if success else None
        
        validation = None
        if success and signal:
            is_valid, issues, severity = validate_signal_format(signal)
            validation = {"is_valid": is_valid, "issues": issues, "severity": severity}
            if issues:
                print(f"  STILL HAS ISSUES: {issues}")
            else:
                print(f"  PASS - all scores within bounds")
        
        return {
            "task_id": task_id,
            "success": success,
            "signal": signal,
            "validation": validation,
            "time_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }


async def main():
    print("Retrying 5 tasks with major validation issues...")
    print("=" * 60)
    
    rate_limiter = RateLimiter(5)
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for task_info in RETRY_TASKS:
            filepath = DATA_TRAINING / task_info["file"]
            idx = task_info["idx"]
            
            # Load the original record
            with open(filepath) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            
            record = None
            for l in lines:
                if l.get("article_idx") == idx:
                    record = l
                    break
            
            if not record:
                print(f"  Could not find idx={idx} in {task_info['file']}")
                continue
            
            print(f"\n[RETRY] {task_info['file']} idx={idx}")
            print(f"  Original issue: {record.get('validation', {}).get('issues', [])}")
            
            result = await retry_task(session, record, rate_limiter)
            
            if result and result.get("success"):
                # Update the record in the file
                record["signal"] = result["signal"]
                record["validation"] = result["validation"]
                record["task_id"] = result["task_id"]
                record["retry"] = True
                record["retry_timestamp"] = result["timestamp"]
                
                # Rewrite the file with the updated record
                updated_lines = []
                for l in lines:
                    if l.get("article_idx") == idx:
                        updated_lines.append(record)
                    else:
                        updated_lines.append(l)
                
                with open(filepath, "w") as f:
                    for l in updated_lines:
                        f.write(json.dumps(l) + "\n")
                
                print(f"  Updated {task_info['file']} idx={idx}")
            else:
                print(f"  RETRY FAILED: {result.get('error', 'unknown') if result else 'None returned'}")
    
    # Regenerate combined file
    print("\n" + "=" * 60)
    print("Regenerating combined training file...")
    combined_file = DATA_TRAINING / "manus_teacher_combined.jsonl"
    training_files = [
        DATA_TRAINING / "manus_real_articles.jsonl",
        DATA_TRAINING / "manus_multi_turn.jsonl",
        DATA_TRAINING / "manus_synthetic.jsonl",
        DATA_TRAINING / "manus_paraphrased.jsonl",
        DATA_TRAINING / "manus_negatives.jsonl",
        DATA_TRAINING / "manus_edge_cases.jsonl",
    ]
    
    total = 0
    with open(combined_file, "w") as out:
        for tf in training_files:
            if tf.exists():
                with open(tf) as f:
                    for line in f:
                        if line.strip():
                            total += 1
                            out.write(line)
    
    print(f"Combined file regenerated: {total} examples")
    
    # Final validation check
    print("\n=== Final Validation Summary ===")
    for tf in training_files:
        if tf.exists():
            with open(tf) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            has_validation = [l for l in lines if l.get('validation')]
            major = sum(1 for l in has_validation if l['validation'].get('severity') == 'major')
            if major > 0:
                print(f"  {tf.name}: {major} major issues remaining")
                for l in has_validation:
                    if l['validation'].get('severity') == 'major':
                        print(f"    idx={l['article_idx']}: {l['validation']['issues']}")
            else:
                print(f"  {tf.name}: ALL PASS")


if __name__ == "__main__":
    asyncio.run(main())
