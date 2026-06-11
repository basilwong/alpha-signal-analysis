"""
Manus Teacher Pipeline: Use Manus API as the teacher model for generating
high-quality training data with deep research and reasoning.

Each article is sent as a Manus task with structured output. The agent:
1. Reads the article
2. Researches the companies and context (web browsing)
3. Reasons about sector impact
4. Produces a structured signal vector JSON

Usage:
    python scripts/generate_training_data_manus.py --input data/raw/articles.jsonl --output data/training/manus_teacher.jsonl --limit 5
    python scripts/generate_training_data_manus.py --input data/raw/articles.jsonl --output data/training/manus_teacher.jsonl --resume
"""

import json
import time
import argparse
import requests
from pathlib import Path

MANUS_API_KEY = "sk-3oFNwU2aOnhvss4PtIMzRIEX5XvFMyLjkV8BTw_kAMkZKVAHu9HKMD3NIIe2Wuwt3XnWF_Qh3ppmJ_qZ_z7CsuFoqTYq"
MANUS_BASE_URL = "https://api.manus.ai/v2"

HEADERS = {
    "x-manus-api-key": MANUS_API_KEY,
    "Content-Type": "application/json",
}

# The structured output schema for our signal vector
SIGNAL_VECTOR_SCHEMA = {
    "type": "object",
    "properties": {
        "signal_vector": {
            "type": "object",
            "properties": {
                "IONQ": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                },
                "RGTI": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                },
                "QBTS": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                },
                "QUBT": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                },
                "IBM": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                },
                "GOOGL": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                },
                "MSFT": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                },
                "HON": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                },
                "NVDA": {
                    "type": "object",
                    "properties": {
                        "score": {"type": "number"},
                        "reasoning": {"type": "string"}
                    },
                    "required": ["score", "reasoning"],
                    "additionalProperties": False
                }
            },
            "required": ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"],
            "additionalProperties": False
        },
        "event_type": {"type": "string"},
        "time_horizon": {
            "type": "string",
            "enum": ["intraday", "2-5 days", "1-2 weeks", "1+ month"]
        },
        "signal_decay": {
            "type": "string",
            "enum": ["fast", "medium", "slow"]
        },
        "information_novelty": {
            "type": "string",
            "enum": ["high", "medium", "low"]
        },
        "technical_translation": {"type": "string"},
        "signal_rationale": {"type": "string"},
        "chain_of_thought": {"type": "string"}
    },
    "required": [
        "signal_vector", "event_type", "time_horizon", "signal_decay",
        "information_novelty", "technical_translation", "signal_rationale",
        "chain_of_thought"
    ],
    "additionalProperties": False
}


TASK_PROMPT_TEMPLATE = """You are a senior quantitative analyst specializing in the quantum computing sector. Your job is to analyze news articles and research papers, then produce a cross-sectional trading signal vector for all public quantum computing companies.

**IMPORTANT: Research the context before producing your signal.** Look up the companies mentioned, check their recent stock performance, verify any claims made in the article, and understand the competitive dynamics.

**The quantum computing universe (9 tickers):**
- IONQ: IonQ (trapped-ion approach, 100% quantum revenue, pure-play)
- RGTI: Rigetti Computing (superconducting approach, 100% quantum revenue, pure-play)
- QBTS: D-Wave Quantum (quantum annealing approach, 100% quantum revenue, pure-play)
- QUBT: Quantum Computing Inc. (neutral atom approach, 100% quantum revenue, pure-play)
- IBM: International Business Machines (superconducting, quantum is ~2% of revenue)
- GOOGL: Alphabet/Google (superconducting, quantum is <0.1% of revenue)
- MSFT: Microsoft (topological approach, quantum is <0.1% of revenue)
- HON: Honeywell/Quantinuum (trapped-ion, quantum is ~5% of revenue via Quantinuum subsidiary)
- NVDA: NVIDIA (adjacent/enabler, sells simulation hardware, quantum is ~1% of revenue)

**Scoring guidelines:**
- Scores range from -2.0 (strongly bearish) to +2.0 (strongly bullish)
- Pure-play companies (IONQ, RGTI, QBTS, QUBT) can receive full range scores
- Diversified companies must be scaled by their quantum revenue exposure:
  - HON: max +/-0.3
  - IBM: max +/-0.15
  - NVDA: max +/-0.10
  - GOOGL, MSFT: max +/-0.05
- If the article is NOT about quantum computing, all scores should be 0.0

**Technology competitive dynamics:**
- Trapped-ion breakthroughs → bullish IONQ/HON, bearish RGTI/IBM/GOOGL (superconducting competitors)
- Superconducting breakthroughs → bullish RGTI/IBM/GOOGL, bearish IONQ/HON (trapped-ion competitors)
- Error correction advances → benefit ALL gate-based approaches
- Government funding → broadly bullish for entire sector
- Negative news about one company → slightly bullish for direct competitors

**Your chain of thought should include:**
1. What is this article actually saying? (separate fact from hype)
2. Which technology approach does this relate to?
3. How significant is this relative to the company's roadmap?
4. How quickly will the market price this in?
5. What are the second-order effects on competitors?

---

**Article to analyze:**

Title: {title}
Source: {source}
Date: {date}

{text}

---

Please research this article's context, reason about its implications for each company in the quantum computing universe, and produce your signal vector. Be specific in your reasoning for each ticker.
"""


def create_task(article):
    """Create a Manus task for a single article."""
    prompt = TASK_PROMPT_TEMPLATE.format(
        title=article.get("title", "Untitled"),
        source=article.get("source", "news"),
        date=article.get("date", "unknown"),
        text=article.get("text", ""),
    )

    payload = {
        "message": {
            "content": prompt,
        },
        "structured_output_schema": SIGNAL_VECTOR_SCHEMA,
    }

    resp = requests.post(f"{MANUS_BASE_URL}/task.create", headers=HEADERS, json=payload)
    resp.raise_for_status()
    data = resp.json()

    if not data.get("ok"):
        raise Exception(f"Task creation failed: {data.get('error', {}).get('message', 'unknown')}")

    return data["task_id"]


def poll_task(task_id, timeout=600, poll_interval=15):
    """Poll a task until it completes or times out."""
    start = time.time()
    while time.time() - start < timeout:
        resp = requests.get(
            f"{MANUS_BASE_URL}/task.listMessages",
            headers=HEADERS,
            params={"task_id": task_id, "order": "desc"},
        )
        resp.raise_for_status()
        data = resp.json()

        if not data.get("ok"):
            raise Exception(f"Poll failed: {data.get('error', {}).get('message', 'unknown')}")

        messages = data.get("messages", [])

        # Check for structured output result
        for msg in messages:
            if msg.get("type") == "structured_output_result":
                return msg.get("structured_output_result", {})

        # Check if task stopped
        for msg in messages:
            if msg.get("type") == "task_stopped":
                # Task stopped but no structured output yet, wait a bit more
                time.sleep(5)
                # Try one more time
                resp2 = requests.get(
                    f"{MANUS_BASE_URL}/task.listMessages",
                    headers=HEADERS,
                    params={"task_id": task_id, "order": "desc"},
                )
                data2 = resp2.json()
                for msg2 in data2.get("messages", []):
                    if msg2.get("type") == "structured_output_result":
                        return msg2.get("structured_output_result", {})
                return {"success": False, "error": "Task stopped without structured output"}

        time.sleep(poll_interval)

    return {"success": False, "error": f"Timeout after {timeout}s"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input articles JSONL file")
    parser.add_argument("--output", required=True, help="Output training data JSONL file")
    parser.add_argument("--limit", type=int, default=None, help="Max articles to process")
    parser.add_argument("--resume", action="store_true", help="Skip already-processed articles")
    parser.add_argument("--timeout", type=int, default=600, help="Max seconds per task")
    parser.add_argument("--poll-interval", type=int, default=15, help="Seconds between polls")
    args = parser.parse_args()

    # Load articles
    articles = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)

    if args.limit:
        articles = articles[:args.limit]

    print(f"Total articles: {len(articles)}")
    print(f"Output: {args.output}")

    # Resume support
    completed_indices = set()
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    if r.get("status") == "success":
                        completed_indices.add(r.get("article_idx"))
        print(f"Resuming: {len(completed_indices)} already done")

    remaining = [a for a in articles if a["idx"] not in completed_indices]
    print(f"Remaining: {len(remaining)}")

    # Process
    success = 0
    errors = 0
    mode = "a" if args.resume else "w"

    with open(args.output, mode) as f_out:
        for i, article in enumerate(remaining):
            idx = article["idx"]
            title = article.get("title", "Untitled")[:60]
            print(f"\n[{i+1}/{len(remaining)}] Article {idx}: {title}")

            try:
                # Create task
                task_id = create_task(article)
                print(f"  Task created: {task_id}")

                # Poll for result
                result = poll_task(task_id, timeout=args.timeout, poll_interval=args.poll_interval)

                if result.get("success"):
                    signal = result["value"]
                    output = {
                        "article_idx": idx,
                        "date": article.get("date", ""),
                        "title": article.get("title", ""),
                        "source": article.get("source", "news"),
                        "status": "success",
                        "signal": signal,
                        "task_id": task_id,
                    }
                    f_out.write(json.dumps(output) + "\n")
                    f_out.flush()
                    success += 1
                    print(f"  SUCCESS: {signal.get('event_type', 'N/A')}")
                else:
                    output = {
                        "article_idx": idx,
                        "date": article.get("date", ""),
                        "title": article.get("title", ""),
                        "source": article.get("source", "news"),
                        "status": "error",
                        "error": result.get("error", "unknown"),
                        "task_id": task_id,
                    }
                    f_out.write(json.dumps(output) + "\n")
                    f_out.flush()
                    errors += 1
                    print(f"  ERROR: {result.get('error', 'unknown')}")

            except Exception as e:
                output = {
                    "article_idx": idx,
                    "date": article.get("date", ""),
                    "title": article.get("title", ""),
                    "source": article.get("source", "news"),
                    "status": "error",
                    "error": str(e)[:300],
                }
                f_out.write(json.dumps(output) + "\n")
                f_out.flush()
                errors += 1
                print(f"  EXCEPTION: {str(e)[:100]}")

            # Progress
            if (i + 1) % 10 == 0:
                print(f"\n  --- Progress: {i+1}/{len(remaining)} | Success: {success} | Errors: {errors} ---")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {success} success, {errors} errors")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
