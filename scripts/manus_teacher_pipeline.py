"""
Manus Teacher Pipeline: Generate ~1,000 High-Quality Training Examples
Uses the Manus API to create tasks that analyze articles and produce structured trading signals.

Usage:
    python scripts/manus_teacher_pipeline.py --phase 1   # Real articles
    python scripts/manus_teacher_pipeline.py --phase 2   # Multi-turn follow-ups
    python scripts/manus_teacher_pipeline.py --phase 3   # Synthetic articles
    python scripts/manus_teacher_pipeline.py --phase 4   # Paraphrased articles
    python scripts/manus_teacher_pipeline.py --phase 5   # Negative examples
    python scripts/manus_teacher_pipeline.py --phase 6   # Edge cases
    python scripts/manus_teacher_pipeline.py --phase 7   # Evaluation predictions
    python scripts/manus_teacher_pipeline.py --phase all # Run all phases sequentially
"""

import json
import time
import argparse
import random
import os
import sys
from pathlib import Path
from datetime import datetime

import requests

# ============================================================
# Configuration
# ============================================================

API_KEY = "sk-3oFNwU2aOnhvss4PtIMzRIEX5XvFMyLjkV8BTw_kAMkZKVAHu9HKMD3NIIe2Wuwt3XnWF_Qh3ppmJ_qZ_z7CsuFoqTYq"
BASE_URL = "https://api.manus.ai/v2"
HEADERS = {"x-manus-api-key": API_KEY, "Content-Type": "application/json"}

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_TRAINING = PROJECT_ROOT / "data" / "training"
DATA_EVAL = PROJECT_ROOT / "data" / "eval"

# Ensure output directories exist
DATA_TRAINING.mkdir(parents=True, exist_ok=True)
DATA_EVAL.mkdir(parents=True, exist_ok=True)

# Timing
POLL_INTERVAL = 30  # seconds between polls
MAX_POLL_TIME = 900  # 15 minutes max per task
RETRY_DELAY = 60  # seconds before retry on failure
MAX_RETRIES = 3

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
# Structured Output Schemas
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

VALIDATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_valid": {"type": "boolean"},
        "issues": {"type": "array", "items": {"type": "string"}},
        "severity": {"type": "string", "enum": ["pass", "minor", "major", "critical"]}
    },
    "required": ["is_valid", "issues", "severity"],
    "additionalProperties": False
}

CONTAMINATION_SCHEMA = {
    "type": "object",
    "properties": {
        "is_contaminated": {"type": "boolean"},
        "contamination_details": {"type": ["string", "null"]},
        "confidence": {"type": "string", "enum": ["high", "medium", "low"]}
    },
    "required": ["is_contaminated", "contamination_details", "confidence"],
    "additionalProperties": False
}

# ============================================================
# API Helpers
# ============================================================

def create_task(prompt: str, schema: dict, retries: int = MAX_RETRIES) -> str:
    """Create a Manus task and return the task_id."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{BASE_URL}/task.create",
                headers=HEADERS,
                json={"message": {"content": prompt}, "structured_output_schema": schema},
                timeout=30
            )
            data = resp.json()
            if data.get("ok"):
                return data["task_id"]
            else:
                print(f"  [WARN] Task creation failed (attempt {attempt+1}): {data.get('error', {}).get('message', 'unknown')}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  [ERROR] Task creation exception (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    return None


def send_message(task_id: str, message: str, schema: dict, retries: int = MAX_RETRIES) -> bool:
    """Send a follow-up message to an existing task."""
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{BASE_URL}/task.sendMessage",
                headers=HEADERS,
                json={"task_id": task_id, "message": {"content": message}, "structured_output_schema": schema},
                timeout=30
            )
            data = resp.json()
            if data.get("ok"):
                return True
            else:
                print(f"  [WARN] sendMessage failed (attempt {attempt+1}): {data.get('error', {}).get('message', 'unknown')}")
                if attempt < retries - 1:
                    time.sleep(RETRY_DELAY)
        except Exception as e:
            print(f"  [ERROR] sendMessage exception (attempt {attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(RETRY_DELAY)
    return False


def poll_for_result(task_id: str, max_time: int = MAX_POLL_TIME) -> dict:
    """Poll a task until it completes and return the structured output result."""
    start = time.time()
    while time.time() - start < max_time:
        try:
            resp = requests.get(
                f"{BASE_URL}/task.listMessages",
                headers=HEADERS,
                params={"task_id": task_id, "order": "desc", "limit": 20},
                timeout=30
            )
            data = resp.json()
            if not data.get("ok"):
                print(f"  [WARN] Poll error: {data.get('error', {}).get('message', 'unknown')}")
                time.sleep(POLL_INTERVAL)
                continue

            messages = data.get("messages", [])
            
            # Check for structured output result
            for msg in messages:
                if msg.get("type") == "structured_output_result":
                    result = msg["structured_output_result"]
                    return result
                    
            # Check for status updates
            for msg in messages:
                if msg.get("type") == "status_update":
                    status = msg.get("status_update", {}).get("agent_status")
                    if status == "stopped":
                        # Task stopped but no structured output found - check all messages
                        resp2 = requests.get(
                            f"{BASE_URL}/task.listMessages",
                            headers=HEADERS,
                            params={"task_id": task_id, "order": "asc"},
                            timeout=30
                        )
                        all_msgs = resp2.json().get("messages", [])
                        for m in all_msgs:
                            if m.get("type") == "structured_output_result":
                                return m["structured_output_result"]
                        return {"success": False, "error": "Task stopped without structured output"}
                    elif status == "error":
                        return {"success": False, "error": "Task errored"}

        except Exception as e:
            print(f"  [WARN] Poll exception: {e}")
        
        time.sleep(POLL_INTERVAL)
    
    return {"success": False, "error": "Polling timeout"}


def run_task(prompt: str, schema: dict) -> dict:
    """Create a task, poll for results, and return the output."""
    task_id = create_task(prompt, schema)
    if not task_id:
        return {"success": False, "error": "Failed to create task"}
    
    print(f"  Task created: {task_id}")
    result = poll_for_result(task_id)
    return {"task_id": task_id, **result}


def run_followup_task(task_id: str, message: str, schema: dict) -> dict:
    """Send a follow-up message and poll for results."""
    success = send_message(task_id, message, schema)
    if not success:
        return {"success": False, "error": "Failed to send follow-up message"}
    
    result = poll_for_result(task_id)
    return {"task_id": task_id, **result}


# ============================================================
# Validation
# ============================================================

def validate_format(output_json: dict) -> dict:
    """Run format validation on a structured output."""
    prompt = f"""You are a data quality auditor. Validate this structured output meets ALL requirements:

**Schema compliance:**
1. All 9 tickers present in signal_vector
2. Each ticker has "score" (number) and "reasoning" (non-empty string)
3. All required top-level fields present

**Score range compliance:**
4. IONQ, RGTI, QBTS, QUBT scores within [-2.0, +2.0]
5. HON within [-0.3, +0.3]
6. IBM within [-0.15, +0.15]
7. NVDA within [-0.10, +0.10]
8. GOOGL, MSFT within [-0.05, +0.05]

**Quality checks:**
9. Each ticker's reasoning is specific and unique (not copy-pasted)
10. technical_translation is 2+ sentences explaining commercial implications
11. signal_rationale explains WHY these scores (not just restating them)
12. chain_of_thought shows step-by-step reasoning (5+ sentences)
13. event_type is descriptive (not generic like "news")
14. For non-quantum articles: ALL scores are 0.0

**Output to validate:** {json.dumps(output_json)}"""
    
    return run_task(prompt, VALIDATION_SCHEMA)


def validate_contamination(date: str, title: str, output_json: dict) -> dict:
    """Run contamination check on evaluation predictions."""
    prompt = f"""You are a data quality auditor. Check whether this analysis contains future information leakage.

The article was published on {date}. The analysis should ONLY reference information available on or before that date.

**Check for:**
1. References to stock price movements AFTER {date}
2. References to events/announcements published AFTER {date}
3. Phrases like "as we now know", "it turned out that", "subsequently"
4. Specific price targets matching actual future outcomes
5. References to quarterly results not yet published on {date}

**Article date:** {date}
**Article title:** {title}
**Model's analysis:** {json.dumps(output_json)}"""
    
    return run_task(prompt, CONTAMINATION_SCHEMA)


# ============================================================
# Output helpers
# ============================================================

def load_existing_results(filepath: Path) -> list:
    """Load existing results for resume support."""
    results = []
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line))
    return results


def append_result(filepath: Path, result: dict):
    """Append a single result to a JSONL file."""
    with open(filepath, "a") as f:
        f.write(json.dumps(result) + "\n")


def get_completed_indices(filepath: Path) -> set:
    """Get set of already-completed article indices."""
    indices = set()
    if filepath.exists():
        with open(filepath) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "article_idx" in data:
                        indices.add(data["article_idx"])
    return indices


# ============================================================
# Phase 1: Real Articles (190 examples)
# ============================================================

def run_phase1_real_articles():
    """Process real articles from articles_train.jsonl."""
    print("\n" + "="*60)
    print("PHASE 1: Real Articles (190 examples)")
    print("="*60)
    
    output_file = DATA_TRAINING / "manus_real_articles.jsonl"
    completed = get_completed_indices(output_file)
    print(f"Already completed: {len(completed)} articles")
    
    # Load articles
    articles = []
    with open(DATA_RAW / "articles_train.jsonl") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    # Process first 190
    articles = articles[:190]
    print(f"Total to process: {len(articles)} (skipping {len(completed)} already done)")
    
    validation_buffer = []
    
    for idx, article in enumerate(articles):
        if idx in completed:
            continue
        
        print(f"\n[{idx+1}/190] Processing: {article['title'][:60]}...")
        
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
        
        result = run_task(prompt, SIGNAL_SCHEMA)
        
        record = {
            "article_idx": idx,
            "category": "real_article",
            "title": article["title"],
            "date": article["date"],
            "source": article["source"],
            "text": article["text"],
            "task_id": result.get("task_id"),
            "success": result.get("success", False),
            "signal": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(output_file, record)
        validation_buffer.append(record)
        
        # Run format validation every 10 tasks
        if len(validation_buffer) >= 10:
            _run_batch_validation(validation_buffer, output_file)
            validation_buffer = []
        
        print(f"  Status: {'SUCCESS' if record['success'] else 'FAILED'}")
    
    # Final validation
    if validation_buffer:
        _run_batch_validation(validation_buffer, output_file)
    
    print(f"\nPhase 1 complete. Results in: {output_file}")


# ============================================================
# Phase 2: Multi-Turn Follow-ups (170 examples)
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

def run_phase2_multi_turn():
    """Create follow-up questions on Phase 1 results."""
    print("\n" + "="*60)
    print("PHASE 2: Multi-Turn Follow-ups (170 examples)")
    print("="*60)
    
    output_file = DATA_TRAINING / "manus_multi_turn.jsonl"
    completed = get_completed_indices(output_file)
    print(f"Already completed: {len(completed)} follow-ups")
    
    # Load Phase 1 results
    phase1_file = DATA_TRAINING / "manus_real_articles.jsonl"
    phase1_results = load_existing_results(phase1_file)
    successful = [r for r in phase1_results if r.get("success")]
    
    if not successful:
        print("ERROR: No successful Phase 1 results found. Run Phase 1 first.")
        return
    
    # Select 170 from successful results
    to_process = successful[:170]
    print(f"Total to process: {len(to_process)} (skipping {len(completed)} already done)")
    
    for idx, phase1_result in enumerate(to_process):
        if idx in completed:
            continue
        
        print(f"\n[{idx+1}/170] Follow-up for: {phase1_result['title'][:60]}...")
        
        task_id = phase1_result.get("task_id")
        signal = phase1_result.get("signal", {})
        
        # Pick a follow-up question
        q_template = FOLLOWUP_QUESTIONS[idx % len(FOLLOWUP_QUESTIONS)]
        
        # Fill in template variables if needed
        if "{ticker}" in q_template and signal:
            sv = signal.get("signal_vector", {})
            # Find the ticker with the highest absolute score
            max_ticker = max(sv.keys(), key=lambda t: abs(sv[t].get("score", 0)))
            max_score = sv[max_ticker]["score"]
            question = q_template.format(ticker=max_ticker, score=max_score)
        else:
            question = q_template
        
        # Send follow-up to the same task
        if task_id:
            result = run_followup_task(task_id, question, FOLLOWUP_SCHEMA)
        else:
            # If no task_id, create a new task with context
            full_prompt = f"""You previously analyzed this article:
Title: {phase1_result['title']}
Date: {phase1_result['date']}

Your analysis produced this signal vector:
{json.dumps(signal, indent=2)}

Now answer this follow-up question:
{question}"""
            result = run_task(full_prompt, FOLLOWUP_SCHEMA)
        
        record = {
            "article_idx": idx,
            "category": "multi_turn",
            "original_article_idx": phase1_result.get("article_idx"),
            "original_title": phase1_result["title"],
            "original_signal": signal,
            "follow_up_question": question,
            "task_id": result.get("task_id"),
            "success": result.get("success", False),
            "follow_up_response": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(output_file, record)
        print(f"  Status: {'SUCCESS' if record['success'] else 'FAILED'}")
    
    print(f"\nPhase 2 complete. Results in: {output_file}")


# ============================================================
# Phase 3: Synthetic Articles (200 examples)
# ============================================================

SYNTHETIC_SCENARIOS = [
    # Technical milestones (50)
    "IonQ achieves 35 algorithmic qubits on their latest Forte Enterprise processor, a 40% improvement over the previous generation",
    "IonQ demonstrates a quantum processor with 99.9% two-qubit gate fidelity, approaching the threshold needed for fault-tolerant computation",
    "IonQ announces a breakthrough in ion shuttling that enables their modular architecture to scale to 1000+ qubits",
    "Rigetti demonstrates 99.5% two-qubit gate fidelity on their 84-qubit Ankaa-3 processor",
    "Rigetti announces a new superconducting processor architecture that reduces crosstalk by 80%",
    "Rigetti achieves quantum volume of 512 on their latest chip, doubling their previous record",
    "Google's quantum team publishes a paper showing 10 logical qubits with surface codes on their 105-qubit Willow chip",
    "Google demonstrates a quantum algorithm that outperforms classical computers on a commercially relevant optimization problem",
    "Google announces a 1000-qubit quantum processor roadmap with delivery expected in 2027",
    "D-Wave announces a 7000-qubit advantage system that solves a real-world logistics problem 100x faster than classical",
    "D-Wave demonstrates quantum speedup on a drug discovery optimization problem for a major pharmaceutical company",
    "D-Wave releases benchmarks showing their annealer outperforms gate-based systems on combinatorial optimization",
    "Microsoft's topological qubit team publishes definitive evidence of non-abelian anyons in their hardware",
    "Microsoft announces their topological qubit has achieved a coherence time 1000x longer than superconducting qubits",
    "Microsoft demonstrates a logical qubit using topological protection, validating their decade-long research bet",
    "Quantinuum achieves record quantum volume of 1048576 (2^20) on their H3 trapped-ion processor",
    "Quantinuum demonstrates the first fault-tolerant quantum algorithm on their commercial hardware",
    "Quantinuum announces a 100-qubit trapped-ion processor with all-to-all connectivity",
    "A university team demonstrates quantum error correction below the threshold on superconducting qubits using a novel code",
    "A university team demonstrates quantum error correction below the threshold on trapped-ion qubits",
    "Researchers at MIT publish a paper showing a 10x improvement in quantum error correction efficiency",
    "A team at Caltech demonstrates a new quantum algorithm that exponentially speeds up machine learning inference",
    "IBM announces a 1000-qubit processor called Flamingo with improved error rates",
    "IBM demonstrates quantum utility on a financial portfolio optimization problem",
    "IBM announces a breakthrough in quantum interconnects allowing multiple processors to work together",
    "NVIDIA releases a quantum circuit simulator that can simulate 50 qubits in real-time on a single GPU",
    "NVIDIA announces a hybrid quantum-classical computing platform integrated with their DGX systems",
    "Quantum Computing Inc. demonstrates a 256-qubit neutral atom processor with 99.5% single-qubit gate fidelity",
    "QUBT announces a breakthrough in atom loading efficiency that enables scaling to 10,000 qubits",
    "A Chinese research team claims quantum supremacy on a problem 10 billion times faster than classical",
    "Researchers demonstrate quantum teleportation over 100km of fiber optic cable with 99% fidelity",
    "A new error correction code is discovered that requires 10x fewer physical qubits per logical qubit",
    "Scientists demonstrate the first quantum network connecting three quantum computers in different cities",
    "A breakthrough in photonic quantum computing achieves room-temperature operation with high fidelity",
    "Researchers demonstrate a quantum memory that maintains coherence for over 1 hour",
    "A new quantum algorithm is published that breaks a widely-used post-quantum cryptography candidate",
    "Scientists achieve quantum entanglement between 100 qubits simultaneously for the first time",
    "A team demonstrates quantum sensing with precision 1000x beyond the classical limit",
    "Researchers publish a method to convert between different qubit types with 99% fidelity",
    "A breakthrough in quantum software compilation reduces circuit depth by 90% for practical algorithms",
    "IonQ publishes results showing their barium qubit technology achieves record coherence times",
    "Rigetti demonstrates real-time quantum error correction on their latest processor",
    "D-Wave announces a gate-based quantum processor, pivoting from pure annealing",
    "Quantinuum demonstrates quantum machine learning that outperforms classical on a real dataset",
    "IBM announces a quantum-centric supercomputer combining 10,000 qubits with classical HPC",
    "Google achieves quantum error correction with a logical error rate below 10^-6",
    "Microsoft demonstrates a topological qubit operating at higher temperatures than competitors",
    "NVIDIA announces quantum hardware simulation accuracy within 0.1% of real quantum processors",
    "A startup demonstrates a quantum processor using a completely novel qubit technology",
    "Researchers achieve fault-tolerant quantum computation with only 100 physical qubits",
    # Business/financial events (50)
    "IonQ reports Q3 2025 revenue of $12.5M, beating estimates by 15% with strong enterprise demand",
    "IonQ announces a $100M contract with the US Air Force for quantum computing services",
    "IonQ raises $500M in a secondary offering at $35 per share to fund manufacturing expansion",
    "IonQ announces partnership with Amazon Web Services to offer quantum computing via Braket",
    "IonQ CEO Peter Chapman resigns; former Google Quantum AI lead appointed as replacement",
    "Rigetti reports Q3 2025 revenue of $4.2M, missing estimates by 20% with slower enterprise adoption",
    "Rigetti announces a $50M contract with the Department of Energy for quantum simulation",
    "Rigetti raises $200M in Series F at a $2.5B valuation from sovereign wealth funds",
    "Rigetti announces partnership with Microsoft Azure to offer quantum computing services",
    "Rigetti announces 25% workforce reduction to extend runway amid slower revenue growth",
    "D-Wave reports Q3 2025 revenue of $8.1M, beating estimates by 10% with growing optimization revenue",
    "D-Wave announces a $75M contract with a major logistics company for quantum optimization",
    "D-Wave raises $150M through a PIPE deal at a 20% premium to market price",
    "D-Wave announces partnership with Google Cloud for quantum optimization services",
    "D-Wave CEO Alan Baratz steps down; board appoints former McKinsey partner as new CEO",
    "Quantum Computing Inc. reports Q3 2025 revenue of $1.8M, beating estimates by 25%",
    "QUBT announces a $30M contract with a defense contractor for quantum sensing applications",
    "QUBT raises $100M in a secondary offering to fund their neutral atom processor development",
    "QUBT announces partnership with IBM to integrate their neutral atom qubits with IBM's ecosystem",
    "QUBT announces 15% workforce reduction amid pivot from photonic to neutral atom technology",
    "Honeywell announces Quantinuum has reached $100M annual recurring revenue",
    "Quantinuum announces a $300M funding round led by JPMorgan at a $10B valuation",
    "Quantinuum announces partnership with NVIDIA for hybrid quantum-classical computing",
    "Honeywell announces plans to spin off Quantinuum as a separate public company",
    "Quantinuum CEO Rajeeb Hazra departs; Honeywell veteran takes over leadership",
    "IBM announces quantum computing revenue reached $500M annually across its quantum division",
    "IBM announces a $1B investment in quantum computing research over the next 5 years",
    "IBM partners with Goldman Sachs to deploy quantum algorithms for derivatives pricing",
    "Google announces $2B investment in quantum computing, doubling its quantum team",
    "Google partners with Volkswagen to use quantum computing for battery material simulation",
    "Microsoft announces Azure Quantum has reached 10,000 enterprise customers",
    "Microsoft invests $500M in Quantinuum for exclusive access to their trapped-ion technology",
    "NVIDIA reports quantum simulation revenue of $200M in Q3, up 150% year-over-year",
    "NVIDIA announces a dedicated quantum computing business unit with 500 employees",
    "Activist investor Carl Icahn takes 8% stake in IonQ, pushes for strategic review",
    "Short seller Hindenburg Research publishes report on Rigetti claiming inflated revenue metrics",
    "Goldman Sachs initiates coverage of quantum computing sector with Overweight on IonQ",
    "Morgan Stanley downgrades entire quantum computing sector citing valuation concerns",
    "IonQ announces acquisition of a quantum software startup for $200M in stock",
    "Rigetti announces merger with another quantum hardware company to combine technologies",
    "D-Wave announces acquisition of a classical optimization company to build hybrid solutions",
    "Amazon announces entry into quantum computing hardware with a superconducting approach",
    "Apple announces a quantum computing research lab focused on post-quantum cryptography",
    "Samsung invests $1B in quantum computing, partners with IonQ for mobile applications",
    "A major pension fund discloses 5% position in quantum computing ETF",
    "Quantum computing ETF sees $500M in inflows in a single week after sector catalyst",
    "IonQ stock added to the S&P 500 index, triggering forced buying from index funds",
    "Rigetti receives a going-concern warning from auditors in their annual report",
    "D-Wave announces a stock buyback program worth $100M",
    "QUBT announces a reverse stock split to maintain NASDAQ listing requirements",
    # Government/regulatory (30)
    "US DOE announces $5B quantum computing funding program over 10 years",
    "EU passes quantum technology sovereignty act with $3B budget for European quantum companies",
    "China announces quantum computing export restrictions on key components and materials",
    "DARPA awards $200M contract to IonQ for quantum computing applications in defense",
    "Congress holds hearing on quantum computing national security implications, considers export controls",
    "US government mandates all federal agencies adopt post-quantum cryptography by 2027",
    "National Science Foundation announces $1B quantum workforce development program",
    "Japan announces $2B quantum computing initiative, partners with IBM and Google",
    "UK government announces $1.5B National Quantum Strategy with focus on commercialization",
    "India announces $1B quantum computing mission, establishes national quantum lab",
    "DARPA awards $150M contract to Rigetti for quantum simulation of chemical processes",
    "Department of Defense designates quantum computing as critical technology, restricts foreign investment",
    "SEC proposes new disclosure requirements for quantum computing companies regarding technical claims",
    "FDA approves first quantum computing-assisted drug design, validates commercial use case",
    "US-China quantum computing cooperation agreement signed, easing export restrictions",
    "NATO establishes quantum computing center of excellence, awards contracts to Western companies",
    "Australia announces $800M quantum computing investment, partners with IonQ for government applications",
    "Canada establishes $500M quantum computing fund, supports domestic companies",
    "South Korea announces $1B quantum computing program, partners with Google and Samsung",
    "US Congress passes Quantum Computing Advancement Act with $10B in funding over 5 years",
    "European Commission designates quantum computing as strategic autonomy priority",
    "US Treasury adds quantum computing companies to CFIUS review list for foreign acquisitions",
    "Pentagon awards $500M quantum computing contract split between IonQ, Rigetti, and IBM",
    "White House releases executive order on quantum computing competitiveness",
    "NIST finalizes post-quantum cryptography standards, creating urgency for quantum-safe migration",
    "China demonstrates quantum communication satellite network, raising national security concerns",
    "US bans export of quantum computing technology to certain countries",
    "EU establishes quantum computing regulatory sandbox for financial services",
    "Singapore announces $300M quantum computing hub, attracts IonQ and Quantinuum",
    "Israel announces $200M quantum computing defense program",
    # Competitive dynamics (40)
    "IonQ acquires a quantum software company for $500M to build a full-stack quantum platform",
    "Rigetti acquires a cryogenics company for $100M to vertically integrate their supply chain",
    "Google acquires a quantum error correction startup for $1B",
    "Microsoft acquires a neutral atom quantum startup for $2B, diversifying from topological approach",
    "Amazon acquires a photonic quantum computing company for $800M",
    "IonQ announces it is exploring superconducting qubits as a complementary technology",
    "Rigetti announces pivot to focus exclusively on quantum error correction rather than NISQ applications",
    "D-Wave announces development of a gate-based quantum processor alongside their annealer",
    "Microsoft announces backup plan using superconducting qubits while topological research continues",
    "QUBT announces pivot from photonic to neutral atom technology after breakthrough results",
    "New quantum startup PsiQuantum raises $600M to build a photonic quantum computer",
    "New quantum startup raises $300M to compete with IonQ using a novel trapped-ion architecture",
    "New quantum startup backed by Elon Musk raises $1B for quantum computing",
    "AWS announces quantum computing service using Rigetti's hardware exclusively",
    "Google Cloud announces quantum computing service using IonQ's hardware",
    "Microsoft Azure announces exclusive partnership with Quantinuum for quantum services",
    "Benchmark shows IonQ's processor outperforms Rigetti's on quantum chemistry simulation",
    "Benchmark shows Rigetti's processor outperforms IonQ's on variational algorithms",
    "Benchmark shows D-Wave's annealer outperforms all gate-based systems on optimization",
    "Benchmark shows Google's processor achieves highest quantum volume ever recorded",
    "Independent study shows quantum computers are still 5-10 years from practical advantage",
    "Independent study shows quantum computers have already achieved advantage on specific problems",
    "Major cloud provider drops quantum computing service citing lack of customer demand",
    "All major cloud providers announce quantum computing price cuts of 50%",
    "A major enterprise customer publicly abandons quantum computing pilot, citing lack of ROI",
    "Fortune 500 company reports $10M in savings from quantum computing optimization",
    "Quantum computing industry consolidation: two major players announce merger",
    "IonQ and Quantinuum announce cross-licensing agreement for trapped-ion patents",
    "Rigetti and IBM announce collaboration on superconducting qubit standards",
    "D-Wave sues IonQ for patent infringement related to quantum annealing techniques",
    "QUBT announces breakthrough that makes their neutral atom approach 10x cheaper than competitors",
    "A major quantum computing company loses key patent case, opening technology to competitors",
    "Intel announces return to quantum computing with a new silicon spin qubit processor",
    "Fujitsu announces a 1000-qubit superconducting processor, entering the Western market",
    "Alibaba shuts down its quantum computing lab, citing geopolitical tensions",
    "A quantum computing company is caught fabricating benchmark results",
    "Two quantum computing companies announce incompatible standards, splitting the ecosystem",
    "A major university quantum lab announces it will only work with open-source quantum platforms",
    "Quantum computing patent filings surge 300% year-over-year, signaling increased competition",
    "A breakthrough in classical algorithms reduces the advantage of quantum computing for key applications",
    # Market/sector events (30)
    "Quantum computing ETF sees $1B in inflows in a single day after major breakthrough announcement",
    "Quantum computing ETF sees $500M in outflows as investors rotate to AI stocks",
    "Major hedge fund Citadel discloses 10% position in IonQ",
    "Renaissance Technologies takes large short position in quantum computing stocks",
    "Analyst at Goldman Sachs initiates coverage of quantum sector with bullish outlook, $50 price target on IonQ",
    "Analyst at Morgan Stanley initiates coverage with bearish outlook, says quantum is overhyped",
    "Jensen Huang says quantum computing is '15-20 years away from being useful' at GTC conference",
    "Jensen Huang reverses course, says quantum computing will be useful within 5 years",
    "Elon Musk tweets that quantum computing is 'the next big thing' and announces Tesla quantum lab",
    "Warren Buffett says he doesn't understand quantum computing and won't invest in it",
    "Short seller Muddy Waters publishes report on IonQ claiming technology doesn't work as advertised",
    "Short seller Citron Research publishes report on D-Wave claiming quantum annealing is obsolete",
    "Quantum computing stocks all drop 20% in a single day with no apparent news catalyst",
    "Quantum computing stocks all surge 30% after surprise government announcement",
    "Options market shows unusual call buying in quantum computing stocks ahead of conference",
    "Insider selling detected at multiple quantum computing companies simultaneously",
    "Quantum computing becomes the most discussed sector on Reddit's WallStreetBets",
    "A major index provider announces creation of a quantum computing sub-index",
    "Quantum computing companies collectively raise $5B in Q3, a record quarter",
    "IPO market for quantum computing companies dries up as investors demand profitability",
    "Quantum computing sector market cap exceeds $100B for the first time",
    "Credit Suisse publishes comprehensive quantum computing sector report projecting $65B TAM by 2030",
    "BlackRock launches dedicated quantum computing fund with $2B in initial capital",
    "Vanguard adds quantum computing stocks to its technology ETF",
    "Quantum computing stocks decouple from broader tech sector, trading on their own catalysts",
    "Retail investor interest in quantum computing reaches all-time high on Robinhood",
    "Institutional ownership of quantum computing stocks reaches 70%, up from 40% a year ago",
    "Quantum computing sector experiences a flash crash, recovering within minutes",
    "A major sovereign wealth fund announces $5B allocation to quantum computing companies",
    "Quantum computing becomes the best-performing tech sub-sector for the third consecutive quarter",
]

def run_phase3_synthetic():
    """Generate and analyze synthetic articles."""
    print("\n" + "="*60)
    print("PHASE 3: Synthetic Articles (200 examples)")
    print("="*60)
    
    output_file = DATA_TRAINING / "manus_synthetic.jsonl"
    completed = get_completed_indices(output_file)
    print(f"Already completed: {len(completed)} articles")
    
    scenarios = SYNTHETIC_SCENARIOS[:200]
    print(f"Total to process: {len(scenarios)} (skipping {len(completed)} already done)")
    
    validation_buffer = []
    
    for idx, scenario in enumerate(scenarios):
        if idx in completed:
            continue
        
        print(f"\n[{idx+1}/200] Scenario: {scenario[:60]}...")
        
        prompt = f"""You are a financial journalist AND a quantitative analyst. Your task has two parts:

**Part 1: Generate a realistic news article** about the following quantum computing scenario. The article should read like a real financial news piece (Reuters/Bloomberg style) with specific details, quotes, and context. Make it 150-300 words.

**Scenario:** {scenario}

**Part 2: Analyze the article you just wrote** and produce a cross-sectional trading signal vector for all 9 quantum computing tickers.

{SHARED_CONTEXT}

**Your chain of thought should explain your reasoning for each ticker's score.**"""
        
        result = run_task(prompt, SIGNAL_SCHEMA)
        
        record = {
            "article_idx": idx,
            "category": "synthetic",
            "scenario": scenario,
            "task_id": result.get("task_id"),
            "success": result.get("success", False),
            "signal": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(output_file, record)
        validation_buffer.append(record)
        
        if len(validation_buffer) >= 10:
            _run_batch_validation(validation_buffer, output_file)
            validation_buffer = []
        
        print(f"  Status: {'SUCCESS' if record['success'] else 'FAILED'}")
    
    if validation_buffer:
        _run_batch_validation(validation_buffer, output_file)
    
    print(f"\nPhase 3 complete. Results in: {output_file}")


# ============================================================
# Phase 4: Paraphrased Articles (190 examples)
# ============================================================

PARAPHRASE_STYLES = [
    "Formal SEC filing language",
    "Casual tech blog post",
    "Twitter/X thread (series of short posts)",
    "Analyst research note",
    "Reddit r/investing discussion post",
]

def run_phase4_paraphrased():
    """Rewrite articles in different styles and analyze."""
    print("\n" + "="*60)
    print("PHASE 4: Paraphrased Articles (190 examples)")
    print("="*60)
    
    output_file = DATA_TRAINING / "manus_paraphrased.jsonl"
    completed = get_completed_indices(output_file)
    print(f"Already completed: {len(completed)} articles")
    
    # Load articles
    articles = []
    with open(DATA_RAW / "articles_train.jsonl") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    articles = articles[:190]
    print(f"Total to process: {len(articles)} (skipping {len(completed)} already done)")
    
    validation_buffer = []
    
    for idx, article in enumerate(articles):
        if idx in completed:
            continue
        
        style = PARAPHRASE_STYLES[idx % len(PARAPHRASE_STYLES)]
        print(f"\n[{idx+1}/190] Paraphrasing ({style}): {article['title'][:50]}...")
        
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
        
        result = run_task(prompt, SIGNAL_SCHEMA)
        
        record = {
            "article_idx": idx,
            "category": "paraphrased",
            "original_title": article["title"],
            "original_date": article["date"],
            "style": style,
            "task_id": result.get("task_id"),
            "success": result.get("success", False),
            "signal": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(output_file, record)
        validation_buffer.append(record)
        
        if len(validation_buffer) >= 10:
            _run_batch_validation(validation_buffer, output_file)
            validation_buffer = []
        
        print(f"  Status: {'SUCCESS' if record['success'] else 'FAILED'}")
    
    if validation_buffer:
        _run_batch_validation(validation_buffer, output_file)
    
    print(f"\nPhase 4 complete. Results in: {output_file}")


# ============================================================
# Phase 5: Negative Examples (150 examples)
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

def run_phase5_negatives():
    """Find and analyze non-quantum articles (all scores should be 0.0)."""
    print("\n" + "="*60)
    print("PHASE 5: Negative Examples (150 examples)")
    print("="*60)
    
    output_file = DATA_TRAINING / "manus_negatives.jsonl"
    completed = get_completed_indices(output_file)
    print(f"Already completed: {len(completed)} articles")
    
    print(f"Total to process: 150 (skipping {len(completed)} already done)")
    
    validation_buffer = []
    
    for idx in range(150):
        if idx in completed:
            continue
        
        topic = NEGATIVE_TOPICS[idx % len(NEGATIVE_TOPICS)]
        # Vary date ranges
        start_year = 2024 + (idx % 3)
        start_month = 1 + (idx % 12)
        start_date = f"{start_year}-{start_month:02d}-01"
        end_date = f"{start_year}-{start_month:02d}-28"
        
        print(f"\n[{idx+1}/150] Finding non-quantum article about: {topic}...")
        
        prompt = f"""You are a quantitative analyst specializing in the quantum computing sector. Your task has two parts:

**Part 1: Find a real news article** from the web that is about technology or finance but is NOT related to quantum computing. Find an article about: {topic}

Browse the web and find a real article published between {start_date} and {end_date}. Copy its title and a 100-200 word summary.

**Part 2: Analyze this article** as if it were submitted to your quantum computing signal system. Since it is NOT about quantum computing, ALL ticker scores should be 0.0 (or very close to zero). Explain in your reasoning why each ticker is unaffected.

{SHARED_CONTEXT}

**IMPORTANT: All scores MUST be 0.0 for this article. The chain of thought should explain why this article has no relevance to the quantum computing sector.**"""
        
        result = run_task(prompt, SIGNAL_SCHEMA)
        
        record = {
            "article_idx": idx,
            "category": "negative",
            "topic": topic,
            "date_range": f"{start_date} to {end_date}",
            "task_id": result.get("task_id"),
            "success": result.get("success", False),
            "signal": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(output_file, record)
        validation_buffer.append(record)
        
        if len(validation_buffer) >= 10:
            _run_batch_validation(validation_buffer, output_file)
            validation_buffer = []
        
        print(f"  Status: {'SUCCESS' if record['success'] else 'FAILED'}")
    
    if validation_buffer:
        _run_batch_validation(validation_buffer, output_file)
    
    print(f"\nPhase 5 complete. Results in: {output_file}")


# ============================================================
# Phase 6: Edge Cases (100 examples)
# ============================================================

EDGE_CASES = [
    "IonQ announces a partnership with Rigetti to co-develop a hybrid trapped-ion/superconducting system. This is unprecedented cooperation between direct competitors.",
    "A peer-reviewed paper shows quantum computers are 1000x slower than classical for a specific optimization problem previously thought to favor quantum. However, it's a very narrow problem class.",
    "NVIDIA announces a quantum simulation chip that makes physical quantum computers unnecessary for most near-term applications. Is this good for NVDA but bad for quantum hardware companies? Or does it validate the field?",
    "China bans export of rare earth materials used in superconducting quantum computers. This hurts RGTI/IBM/GOOGL but IONQ/HON use different materials (trapped ions).",
    "A major quantum computing company (unnamed in the leak) is about to announce bankruptcy. Market speculation is rampant but no one knows which company.",
    "New theoretical result proves fault-tolerant quantum computing requires 10x more qubits than previously thought. This delays everyone, but who is closest to the new threshold?",
    "Amazon announces entry into quantum computing hardware with a massive $5B investment. This is a new competitor but also validates the market opportunity.",
    "IonQ's chief scientist and 5 key researchers leave to join a stealth startup. Brain drain is concerning, but the startup might eventually be acquired.",
    "Quantum computing stocks all drop 20% in a single day with no news. Pure market sentiment shift, not fundamental. How should signals reflect this?",
    "A preprint claims to break RSA encryption with a 100-qubit quantum computer. If true, this is massive for the sector. But preprints are often wrong.",
    "Google publishes a paper showing their quantum processor made a critical error in their previous quantum supremacy claim. The original result was partially wrong.",
    "A quantum computing company announces record revenue but also reveals their technology has a fundamental scaling limitation they hadn't previously disclosed.",
    "Two quantum computing companies announce a merger, but antitrust regulators signal they may block it. The combined entity would dominate the market.",
    "A major customer publicly states they're switching from IonQ to Rigetti, citing better performance. But IonQ disputes the benchmarks used.",
    "Quantum computing is mentioned positively in a State of the Union address, but no specific funding is announced. Is this bullish or just political theater?",
    "A Nobel Prize is awarded for theoretical work that could eventually benefit quantum computing, but the practical applications are decades away.",
    "Insurance companies announce they will no longer cover quantum computing companies due to technology risk. This raises cost of capital for the sector.",
    "A major cybersecurity incident is attributed to a quantum computer, but experts debate whether it was actually quantum or just advanced classical computing.",
    "The CEO of IonQ makes controversial political statements, leading to calls for boycotts. The technology is unaffected but brand reputation suffers.",
    "A documentary about quantum computing goes viral on Netflix, dramatically increasing retail investor interest but also raising concerns about a bubble.",
    "Quantum computing companies collectively miss earnings expectations in the same quarter, but all raise forward guidance significantly.",
    "A major pension fund announces it's divesting from all quantum computing stocks due to ESG concerns about energy consumption.",
    "A leaked internal memo from Google suggests their quantum team is demoralized and considering pivoting to a different approach.",
    "The quantum computing sector experiences a 'short squeeze' where heavily shorted stocks surge 50% in a day on minimal news.",
    "A respected quantum physicist publishes an op-ed arguing that useful quantum computing is impossible due to fundamental physics limitations.",
    "IonQ and D-Wave announce they are suing each other for patent infringement, creating uncertainty for both companies.",
    "A quantum computing ETF announces it will rebalance, removing QUBT and adding a new company. This triggers forced selling/buying.",
    "China demonstrates a quantum computer that appears to outperform all Western systems, but the results cannot be independently verified.",
    "A major cloud provider announces it's shutting down its quantum computing service due to lack of demand, but another provider simultaneously launches one.",
    "Quantum computing companies receive a wave of analyst upgrades and downgrades on the same day, with no consensus on the sector's direction.",
    "A breakthrough in room-temperature superconductivity is announced (later retracted). During the announcement period, how should quantum stocks react?",
    "The US and China sign a quantum computing cooperation agreement, but hawks in Congress threaten to block implementation.",
    "A quantum computing company announces it has achieved quantum advantage, but the claim is disputed by academic researchers.",
    "NVIDIA announces it will stop supporting quantum computing simulation on its GPUs, focusing instead on AI workloads.",
    "A major bank announces it has achieved a quantum computing use case in production, but won't disclose which hardware provider they used.",
    "Quantum computing stocks surge after a social media influencer with 10M followers endorses the sector. Is this sustainable?",
    "A fire destroys a quantum computing company's primary research facility. No injuries, but years of research may be lost.",
    "The Federal Reserve mentions quantum computing as a potential systemic risk to financial markets in their stability report.",
    "A quantum computing company's stock is halted for volatility after rumors of a breakthrough that the company neither confirms nor denies.",
    "Two countries announce competing quantum internet projects, potentially fragmenting the global quantum ecosystem.",
    "A major pharmaceutical company announces quantum computing saved them $1B in drug development, but competitors question the methodology.",
    "Quantum computing hiring surges 500% year-over-year, but most positions remain unfilled due to talent shortage.",
    "A quantum computing company announces a stock split, historically associated with bullish sentiment but no fundamental change.",
    "The quantum computing sector's total addressable market estimate is revised downward by 50% by a major consulting firm.",
    "A quantum computing company is added to a major stock index, triggering passive fund buying, but the company's fundamentals haven't changed.",
    "Quantum computing companies face a new tax on quantum hardware imports, raising costs for US-based manufacturers.",
    "A major university announces it will offer free quantum computing access to all students, potentially disrupting commercial quantum cloud services.",
    "A quantum computing company's founder sells 50% of their shares, but states it's for 'personal financial planning' not lack of confidence.",
    "The quantum computing sector experiences its first insider trading scandal, raising governance concerns across the industry.",
    "A breakthrough in classical computing (new chip architecture) narrows the gap with quantum for certain problems, but quantum still leads on others.",
    "Quantum computing companies collectively file 1000 patents in a single quarter, but critics say most are defensive and low-quality.",
    "A major quantum computing conference is cancelled due to lack of submissions, raising questions about the pace of progress.",
    "A quantum computing company announces it will open-source its entire software stack, potentially commoditizing a key competitive advantage.",
    "The quantum computing sector receives its first credit rating from Moody's, with most companies rated as speculative grade.",
    "A quantum computing company announces a partnership with a controversial government, raising ethical concerns.",
    "Quantum computing stocks become the most volatile sector in the market, with average daily moves exceeding 5%.",
    "A quantum computing company's auditor resigns unexpectedly, raising concerns about financial reporting integrity.",
    "The quantum computing sector experiences a 'death cross' technical pattern, historically associated with further declines.",
    "A quantum computing company announces it has been secretly working on a breakthrough for 3 years and will reveal results next month.",
    "Multiple quantum computing companies announce secondary offerings on the same day, flooding the market with new shares.",
    "A quantum computing company's key patent expires, potentially allowing competitors to use their technology freely.",
    "The quantum computing sector's correlation with Bitcoin reaches an all-time high, suggesting speculative rather than fundamental trading.",
    "A quantum computing company announces a dividend for the first time, signaling either maturity or lack of growth opportunities.",
    "Quantum computing is excluded from a major technology index rebalancing, reducing passive investment flows to the sector.",
    "A quantum computing company announces it has been hacked, with proprietary research potentially stolen by a nation-state actor.",
    "The quantum computing sector experiences a 'golden cross' technical pattern after months of decline, potentially signaling a trend reversal.",
    "A quantum computing company's board fires the CEO and entire C-suite simultaneously, citing 'strategic differences'.",
    "Quantum computing companies collectively burn through $2B in cash in a single quarter, raising sustainability concerns.",
    "A quantum computing company announces it will pivot entirely to quantum-as-a-service, abandoning hardware development.",
    "The quantum computing sector receives a wave of patent troll lawsuits, potentially costing companies hundreds of millions.",
    "A quantum computing company announces a breakthrough but simultaneously reveals it's running low on cash and needs to raise capital.",
    "Quantum computing stocks surge after being mentioned in a popular TV show, but the mention was actually negative.",
    "A quantum computing company's co-founder publicly disagrees with the CEO about the company's technical direction.",
    "The quantum computing sector's short interest reaches an all-time high, creating potential for a short squeeze.",
    "A quantum computing company announces it has achieved profitability, but only through one-time government grants.",
    "Quantum computing companies face a new regulatory requirement to disclose error rates, potentially embarrassing some players.",
    "A quantum computing company announces a 'quantum winter' is coming and recommends investors reduce exposure to the sector.",
    "The quantum computing sector experiences unusual options activity suggesting someone knows something the market doesn't.",
    "A quantum computing company's stock price diverges significantly from its peers for no apparent reason.",
    "Quantum computing companies collectively announce they will form an industry consortium to set standards.",
    "A quantum computing company announces it will accept Bitcoin as payment for quantum computing services.",
    "The quantum computing sector's average P/S ratio exceeds 100x, raising bubble concerns among value investors.",
    "A quantum computing company announces a major contract win, but the contract includes performance milestones that may be impossible to meet.",
    "Quantum computing stocks react differently to the same news: some surge while others drop, confusing analysts.",
    "A quantum computing company announces it will go private in a management buyout at a 30% premium.",
    "The quantum computing sector experiences its first major fraud case, with a company accused of fabricating results.",
    "A quantum computing company announces a partnership with a defense contractor, raising concerns about dual-use technology.",
    "Quantum computing companies face a talent war, with key researchers being poached by AI companies offering 3x salaries.",
    "A quantum computing company's stock is added to the 'meme stock' category by financial media, changing its trading dynamics.",
    "The quantum computing sector receives contradictory analyst reports on the same day: one says buy everything, another says sell everything.",
    "A quantum computing company announces it has solved a problem that was thought to be impossible, but hasn't published the proof yet.",
    "Quantum computing companies collectively announce they will reduce R&D spending to focus on near-term revenue.",
    "A quantum computing company's largest customer announces they're building their own quantum computer in-house.",
    "The quantum computing sector experiences a liquidity crisis as market makers reduce their presence in quantum stocks.",
    "A quantum computing company announces a breakthrough in quantum networking that could make their hardware the standard for quantum internet.",
    "Quantum computing stocks become correlated with interest rate expectations, suggesting the market views them as duration assets.",
    "A quantum computing company announces it will merge with a classical computing company, creating a hybrid quantum-classical entity.",
    "The quantum computing sector's implied volatility reaches levels typically seen before major announcements.",
    "A quantum computing company's key technology works perfectly in lab conditions but fails consistently in real-world deployments, raising questions about whether lab results are meaningful.",
    "Multiple quantum computing CEOs simultaneously sell large blocks of shares on the same day, but each cites different personal reasons for the sale.",
]

def run_phase6_edge_cases():
    """Generate and analyze ambiguous/edge case scenarios."""
    print("\n" + "="*60)
    print("PHASE 6: Edge Cases (100 examples)")
    print("="*60)
    
    output_file = DATA_TRAINING / "manus_edge_cases.jsonl"
    completed = get_completed_indices(output_file)
    print(f"Already completed: {len(completed)} edge cases")
    
    scenarios = EDGE_CASES[:100]
    print(f"Total to process: {len(scenarios)} (skipping {len(completed)} already done)")
    
    validation_buffer = []
    
    for idx, edge_case in enumerate(scenarios):
        if idx in completed:
            continue
        
        print(f"\n[{idx+1}/100] Edge case: {edge_case[:60]}...")
        
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
        
        result = run_task(prompt, SIGNAL_SCHEMA)
        
        record = {
            "article_idx": idx,
            "category": "edge_case",
            "scenario": edge_case,
            "task_id": result.get("task_id"),
            "success": result.get("success", False),
            "signal": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(output_file, record)
        validation_buffer.append(record)
        
        if len(validation_buffer) >= 10:
            _run_batch_validation(validation_buffer, output_file)
            validation_buffer = []
        
        print(f"  Status: {'SUCCESS' if record['success'] else 'FAILED'}")
    
    if validation_buffer:
        _run_batch_validation(validation_buffer, output_file)
    
    print(f"\nPhase 6 complete. Results in: {output_file}")


# ============================================================
# Phase 7: Evaluation Predictions (421 examples)
# ============================================================

def run_phase7_eval_predictions():
    """Run predictions on evaluation articles (no future information)."""
    print("\n" + "="*60)
    print("PHASE 7: Evaluation Predictions (421 examples)")
    print("="*60)
    
    output_file = DATA_EVAL / "predictions_manus_teacher.jsonl"
    contamination_file = DATA_EVAL / "validation_contamination.jsonl"
    completed = get_completed_indices(output_file)
    print(f"Already completed: {len(completed)} predictions")
    
    # Load eval articles
    articles = []
    with open(DATA_RAW / "articles_eval.jsonl") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))
    
    # Use first 421
    articles = articles[:421]
    print(f"Total to process: {len(articles)} (skipping {len(completed)} already done)")
    
    validation_buffer = []
    contamination_buffer = []
    
    for idx, article in enumerate(articles):
        if idx in completed:
            continue
        
        print(f"\n[{idx+1}/421] Eval: {article['title'][:60]}...")
        
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
        
        result = run_task(prompt, SIGNAL_SCHEMA)
        
        record = {
            "article_idx": idx,
            "category": "eval_prediction",
            "title": article["title"],
            "date": article["date"],
            "source": article["source"],
            "text": article["text"],
            "task_id": result.get("task_id"),
            "success": result.get("success", False),
            "signal": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(output_file, record)
        validation_buffer.append(record)
        contamination_buffer.append(record)
        
        # Format validation every 10
        if len(validation_buffer) >= 10:
            _run_batch_validation(validation_buffer, output_file)
            validation_buffer = []
        
        # Contamination check every 10
        if len(contamination_buffer) >= 10:
            _run_contamination_checks(contamination_buffer, contamination_file)
            contamination_buffer = []
        
        print(f"  Status: {'SUCCESS' if record['success'] else 'FAILED'}")
    
    # Final checks
    if validation_buffer:
        _run_batch_validation(validation_buffer, output_file)
    if contamination_buffer:
        _run_contamination_checks(contamination_buffer, contamination_file)
    
    print(f"\nPhase 7 complete. Results in: {output_file}")


# ============================================================
# Batch Validation Helpers
# ============================================================

def _run_batch_validation(records: list, output_file: Path):
    """Run format validation on a batch of records."""
    print(f"\n  [VALIDATION] Running format checks on {len(records)} records...")
    
    validation_file = DATA_EVAL / "validation_format.jsonl"
    
    for record in records:
        if not record.get("success") or not record.get("signal"):
            continue
        
        # Quick local validation (no API call needed for obvious checks)
        signal = record["signal"]
        issues = []
        
        sv = signal.get("signal_vector", {})
        required_tickers = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]
        
        for ticker in required_tickers:
            if ticker not in sv:
                issues.append(f"Missing ticker: {ticker}")
            else:
                score = sv[ticker].get("score", 0)
                # Check score ranges
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
        
        # Determine severity
        if not issues:
            severity = "pass"
        elif any("Missing" in i for i in issues):
            severity = "critical"
        elif any("exceeds" in i for i in issues):
            severity = "major"
        else:
            severity = "minor"
        
        validation_record = {
            "article_idx": record.get("article_idx"),
            "category": record.get("category"),
            "is_valid": len(issues) == 0,
            "issues": issues,
            "severity": severity,
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(validation_file, validation_record)
        
        if severity == "critical":
            print(f"    [CRITICAL] Article {record.get('article_idx')}: {issues}")


def _run_contamination_checks(records: list, contamination_file: Path):
    """Run contamination checks on evaluation predictions."""
    print(f"\n  [CONTAMINATION] Checking {len(records)} eval predictions...")
    
    for record in records:
        if not record.get("success") or not record.get("signal"):
            continue
        
        # Quick heuristic check (save API calls for obvious cases)
        signal = record["signal"]
        chain = signal.get("chain_of_thought", "")
        rationale = signal.get("signal_rationale", "")
        
        contamination_phrases = [
            "as we now know", "it turned out", "subsequently", "in hindsight",
            "later revealed", "would eventually", "as history showed",
            "the stock went on to", "price eventually"
        ]
        
        is_contaminated = False
        details = None
        
        for phrase in contamination_phrases:
            if phrase.lower() in chain.lower() or phrase.lower() in rationale.lower():
                is_contaminated = True
                details = f"Found contamination phrase: '{phrase}'"
                break
        
        contamination_record = {
            "article_idx": record.get("article_idx"),
            "title": record.get("title"),
            "date": record.get("date"),
            "is_contaminated": is_contaminated,
            "contamination_details": details,
            "confidence": "high" if is_contaminated else "low",
            "timestamp": datetime.now().isoformat()
        }
        
        append_result(contamination_file, contamination_record)
        
        if is_contaminated:
            print(f"    [CONTAMINATED] Article {record.get('article_idx')}: {details}")


# ============================================================
# Final Aggregation
# ============================================================

def run_final_aggregation():
    """Combine all training files and report statistics."""
    print("\n" + "="*60)
    print("FINAL AGGREGATION")
    print("="*60)
    
    # Combine training files
    combined_file = DATA_TRAINING / "manus_teacher_combined.jsonl"
    total_count = 0
    success_count = 0
    
    training_files = [
        DATA_TRAINING / "manus_real_articles.jsonl",
        DATA_TRAINING / "manus_multi_turn.jsonl",
        DATA_TRAINING / "manus_synthetic.jsonl",
        DATA_TRAINING / "manus_paraphrased.jsonl",
        DATA_TRAINING / "manus_negatives.jsonl",
        DATA_TRAINING / "manus_edge_cases.jsonl",
    ]
    
    with open(combined_file, "w") as out:
        for tf in training_files:
            if tf.exists():
                with open(tf) as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line)
                            total_count += 1
                            if data.get("success"):
                                success_count += 1
                            out.write(line)
    
    # Report statistics
    print(f"\nTraining Data Statistics:")
    print(f"  Total examples: {total_count}")
    print(f"  Successful: {success_count}")
    print(f"  Success rate: {success_count/max(total_count,1)*100:.1f}%")
    
    # Per-category breakdown
    for tf in training_files:
        if tf.exists():
            with open(tf) as f:
                lines = [json.loads(l) for l in f if l.strip()]
            successes = sum(1 for l in lines if l.get("success"))
            print(f"  {tf.name}: {len(lines)} total, {successes} successful")
    
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
    
    # Contamination report
    contamination_file = DATA_EVAL / "validation_contamination.jsonl"
    if contamination_file.exists():
        with open(contamination_file) as f:
            contam_lines = [json.loads(l) for l in f if l.strip()]
        contaminated = sum(1 for l in contam_lines if l.get("is_contaminated"))
        print(f"\nContamination Check:")
        print(f"  Checked: {len(contam_lines)}")
        print(f"  Contaminated: {contaminated}")
        print(f"  Contamination rate: {contaminated/max(len(contam_lines),1)*100:.1f}%")
    
    # Format validation report
    validation_file = DATA_EVAL / "validation_format.jsonl"
    if validation_file.exists():
        with open(validation_file) as f:
            val_lines = [json.loads(l) for l in f if l.strip()]
        by_severity = {}
        for v in val_lines:
            sev = v.get("severity", "unknown")
            by_severity[sev] = by_severity.get(sev, 0) + 1
        print(f"\nFormat Validation:")
        print(f"  Total checked: {len(val_lines)}")
        for sev, count in sorted(by_severity.items()):
            print(f"  {sev}: {count}")
    
    print(f"\nCombined training file: {combined_file}")
    print(f"Total training examples: {success_count}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Manus Teacher Pipeline")
    parser.add_argument("--phase", type=str, default="all",
                       help="Phase to run: 1-7 or 'all'")
    parser.add_argument("--aggregate", action="store_true",
                       help="Run final aggregation only")
    args = parser.parse_args()
    
    if args.aggregate:
        run_final_aggregation()
        return
    
    phases = {
        "1": run_phase1_real_articles,
        "2": run_phase2_multi_turn,
        "3": run_phase3_synthetic,
        "4": run_phase4_paraphrased,
        "5": run_phase5_negatives,
        "6": run_phase6_edge_cases,
        "7": run_phase7_eval_predictions,
    }
    
    if args.phase == "all":
        for phase_num in ["1", "2", "3", "4", "5", "6", "7"]:
            phases[phase_num]()
        run_final_aggregation()
    elif args.phase in phases:
        phases[args.phase]()
    else:
        print(f"Unknown phase: {args.phase}. Use 1-7 or 'all'.")
        sys.exit(1)


if __name__ == "__main__":
    main()
