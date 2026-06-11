"""
Fix technology-validation vs competitive-threat inconsistency.

The market data is clear: when Google announced Willow (Dec 9, 2024),
RGTI surged +89% in 5 days. Technology validation dominates competitive threat
for pre-revenue quantum stocks.

This script:
1. Identifies examples where same-tech competitors are scored bearish on tech breakthroughs
2. Regenerates those specific examples via Manus API with the corrected rule
3. Updates the training data

The rule: Technology breakthroughs by a large company (Google, IBM, Microsoft)
VALIDATE the approach and are BULLISH for smaller same-technology competitors.
Only business wins (contracts, partnerships, market share) are bearish for competitors.

Usage:
    python scripts/fix_tech_validation_inconsistency.py --identify   # Just list flagged examples
    python scripts/fix_tech_validation_inconsistency.py --regenerate # Regenerate via Manus API
"""

import json
import asyncio
import aiohttp
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_TRAINING = PROJECT_ROOT / "data" / "training"

from src.prompts import SHARED_CONTEXT, SIGNAL_SCHEMA

API_KEY = "sk-3oFNwU2aOnhvss4PtIMzRIEX5XvFMyLjkV8BTw_kAMkZKVAHu9HKMD3NIIe2Wuwt3XnWF_Qh3ppmJ_qZ_z7CsuFoqTYq"
BASE_URL = "https://api.manus.ai/v2"
HEADERS = {"x-manus-api-key": API_KEY, "Content-Type": "application/json"}

# The technology validation rule to add to prompts
TECH_VALIDATION_RULE = """
**Technology validation vs. competitive threat (CRITICAL RULE):**
When a large company (Google, IBM, Microsoft) achieves a technology breakthrough:
- This VALIDATES the technology approach and is BULLISH for smaller same-technology competitors
- Example: Google Willow (superconducting) breakthrough → RGTI surged +89% in 5 days
- The market prices "this approach is viable" BEFORE it prices "who wins within the approach"
- Pre-revenue quantum stocks move on sector validation, not competitive positioning

Only score same-tech competitors BEARISH when:
- The news is a business win (contract, partnership, exclusive deal) — zero-sum
- The competitor demonstrates a proprietary advantage others cannot replicate
- The article explicitly discusses competitive displacement

Default: technology breakthrough by Google/IBM → BULLISH for RGTI (same tech)
Default: technology breakthrough by Quantinuum → BULLISH for IONQ/QNT (same tech)
"""


def identify_inconsistent_examples():
    """Find examples where the teacher applied competitive-threat when it should be tech-validation."""
    
    with open(DATA_TRAINING / "manus_teacher_combined.jsonl") as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    flagged = []
    
    for line_idx, r in enumerate(records):
        if not r.get("success") or not r.get("signal"):
            continue
        
        text = (r.get("title", "") + " " + r.get("text", "") + " " + r.get("scenario", "")).lower()
        sv = r["signal"]["signal_vector"]
        
        # Pattern: Google/IBM superconducting tech breakthrough → RGTI should be bullish
        is_sc_tech_breakthrough = (
            ("google" in text or "ibm" in text) and
            any(w in text for w in ["breakthrough", "achieves", "demonstrates", "error correction", 
                                     "logical qubit", "fidelity", "processor", "willow", "below threshold"]) and
            not any(w in text for w in ["contract", "partnership", "revenue", "earnings", 
                                         "acquisition", "hires", "workforce", "layoff"])
        )
        
        if is_sc_tech_breakthrough:
            rgti_score = sv.get("RGTI", {}).get("score", 0)
            if rgti_score < -0.3:
                flagged.append({
                    "line_idx": line_idx,
                    "article_idx": r.get("article_idx"),
                    "category": r.get("category"),
                    "title": r.get("title", r.get("scenario", ""))[:70],
                    "rgti_score": rgti_score,
                    "issue": "SC tech validation scored as competitive threat",
                    "record": r,
                })
    
    return flagged


async def regenerate_example(session, record, semaphore):
    """Regenerate a single example with the corrected rule."""
    async with semaphore:
        # Build prompt with the tech validation rule
        article_text = record.get("text", record.get("scenario", ""))
        title = record.get("title", "")
        date = record.get("date", "2025-01-01")
        source = record.get("source", "news")
        
        prompt = f"""You are a senior quantitative analyst specializing in the quantum computing sector. Analyze this article and produce a cross-sectional trading signal vector.

{SHARED_CONTEXT}

{TECH_VALIDATION_RULE}

**Your chain of thought MUST include:**
1. What is this article actually saying?
2. Is this a TECHNOLOGY VALIDATION event or a COMPETITIVE THREAT event?
3. If technology validation: same-tech competitors should be BULLISH (the approach works!)
4. If competitive threat: same-tech competitors may be bearish (zero-sum business win)
5. Apply the rule: Google/IBM SC breakthrough → RGTI bullish (validated by market: RGTI +89% on Willow)

---

**Article:**
Title: {title}
Source: {source}
Date: {date}

{article_text}"""
        
        try:
            payload = {
                "message": {"content": prompt},
                "structured_output_schema": SIGNAL_SCHEMA,
                "agent_profile": "manus-1.6-max"
            }
            async with session.post(f"{BASE_URL}/task.create", headers=HEADERS, json=payload) as resp:
                if resp.status == 429:
                    await asyncio.sleep(60)
                    async with session.post(f"{BASE_URL}/task.create", headers=HEADERS, json=payload) as resp2:
                        data = await resp2.json()
                else:
                    data = await resp.json()
                
                if not data.get("ok"):
                    return None
                
                task_id = data["task_id"]
            
            # Poll
            start = time.time()
            while time.time() - start < 900:
                await asyncio.sleep(30)
                params = {"task_id": task_id, "order": "desc", "limit": 20}
                async with session.get(f"{BASE_URL}/task.listMessages", headers=HEADERS, params=params) as resp:
                    poll_data = await resp.json()
                    if not poll_data.get("ok"):
                        continue
                    for msg in poll_data.get("messages", []):
                        if msg.get("type") == "structured_output_result":
                            result = msg["structured_output_result"]
                            if result.get("success"):
                                return result["value"]
                            return None
                    for msg in poll_data.get("messages", []):
                        if msg.get("type") == "status_update":
                            status = msg.get("status_update", {}).get("agent_status")
                            if status in ("stopped", "error"):
                                return None
            return None
        except Exception as e:
            print(f"  Error: {e}")
            return None


async def regenerate_flagged(flagged_examples):
    """Regenerate all flagged examples."""
    print(f"Regenerating {len(flagged_examples)} examples...")
    
    semaphore = asyncio.Semaphore(5)  # Conservative concurrency
    connector = aiohttp.TCPConnector(limit=10)
    timeout = aiohttp.ClientTimeout(total=60)
    
    results = {}
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for i, flagged in enumerate(flagged_examples):
            print(f"  [{i+1}/{len(flagged_examples)}] Regenerating idx={flagged['article_idx']}...")
            new_signal = await regenerate_example(session, flagged["record"], semaphore)
            
            if new_signal:
                new_rgti = new_signal.get("signal_vector", {}).get("RGTI", {}).get("score", 0)
                print(f"    Old RGTI={flagged['rgti_score']:+.2f} → New RGTI={new_rgti:+.2f}")
                results[flagged["line_idx"]] = new_signal
            else:
                print(f"    FAILED to regenerate")
            
            await asyncio.sleep(2)  # Rate limit buffer
    
    return results


def apply_regenerated(results: dict):
    """Apply regenerated signals to the training data."""
    combined_file = DATA_TRAINING / "manus_teacher_combined.jsonl"
    
    with open(combined_file) as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    updated = 0
    for line_idx, new_signal in results.items():
        if line_idx < len(records):
            records[line_idx]["signal"] = new_signal
            records[line_idx]["regenerated_reason"] = "tech_validation_inconsistency"
            records[line_idx]["regenerated_timestamp"] = datetime.now().isoformat()
            updated += 1
    
    # Write atomically
    temp_file = combined_file.with_suffix(".jsonl.tmp")
    with open(temp_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    temp_file.rename(combined_file)
    
    print(f"Updated {updated} examples in {combined_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--identify", action="store_true", help="Just identify flagged examples")
    parser.add_argument("--regenerate", action="store_true", help="Regenerate flagged examples via Manus API")
    args = parser.parse_args()
    
    flagged = identify_inconsistent_examples()
    
    print(f"Flagged examples: {len(flagged)}")
    print()
    
    for f in flagged:
        print(f"  [{f['category']}] idx={f['article_idx']}: RGTI={f['rgti_score']:+.2f}")
        print(f"    {f['title']}")
        print()
    
    if args.regenerate:
        print("=" * 60)
        results = asyncio.run(regenerate_flagged(flagged))
        
        if results:
            print(f"\nSuccessfully regenerated: {len(results)}/{len(flagged)}")
            apply_regenerated(results)
        else:
            print("No examples were successfully regenerated.")
    elif not args.identify:
        print("Use --identify to just list, or --regenerate to fix via Manus API")


if __name__ == "__main__":
    main()
