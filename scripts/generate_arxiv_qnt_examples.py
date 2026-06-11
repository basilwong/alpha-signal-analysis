"""
Generate new training examples for Fix 13 (arXiv rebalancing) and Fix 15 (QNT examples).
Uses the Manus API with the updated prompts and schema.

Fix 13: 70 arXiv examples (10 important, 45 incremental, 15 unrelated)
Fix 15: 35 QNT competitive dynamics examples

Usage:
    python scripts/generate_arxiv_qnt_examples.py --type arxiv
    python scripts/generate_arxiv_qnt_examples.py --type qnt
    python scripts/generate_arxiv_qnt_examples.py --type all
"""

import asyncio
import aiohttp
import json
import time
import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.prompts import SHARED_CONTEXT, SIGNAL_SCHEMA, ARXIV_PROMPT, QNT_SCENARIO_PROMPT

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

DATA_TRAINING = PROJECT_ROOT / "data" / "training"

# ============================================================
# ArXiv Scenarios (Fix 13)
# ============================================================

ARXIV_IMPORTANT = [
    "IonQ researchers demonstrate 35 algorithmic qubits with 99.7% two-qubit gate fidelity on barium qubits, published in Nature",
    "IBM Quantum team achieves below-threshold error correction on 127-qubit Eagle processor using heavy-hex code, published in Science",
    "Rigetti publishes Nature paper: 99.5% CZ gate fidelity on 84-qubit Ankaa-3, enabling practical error correction",
    "Quantinuum demonstrates 50 logical qubits with real-time error correction on H2 processor, published in Nature Physics",
    "IonQ and Duke University demonstrate distributed quantum computing across 4 networked trapped-ion nodes, published in Physical Review Letters",
    "IBM researchers demonstrate quantum utility for materials simulation exceeding classical methods on 127 qubits, published in Nature",
    "Rigetti team publishes first demonstration of fault-tolerant variational quantum eigensolver on superconducting hardware in Science",
    "Quantinuum achieves quantum volume 2^21 (2097152), demonstrating exponential scaling, published in PRX Quantum",
    "IonQ publishes results showing barium qubit T2 coherence times exceeding 10 seconds, a 100x improvement, in Nature Physics",
    "D-Wave publishes peer-reviewed quantum speedup on real-world logistics optimization for FedEx in Nature Computational Science",
]

ARXIV_INCREMENTAL = [
    "Improved bounds on quantum circuit depth for approximate optimization algorithms",
    "Noise characterization and mitigation in superconducting transmon qubits at millikelvin temperatures",
    "Variational quantum eigensolver convergence analysis for molecular hydrogen",
    "Quantum error correction with repetition codes: a pedagogical review and numerical study",
    "Benchmarking quantum volume across different qubit modalities: a comparative study",
    "Theoretical analysis of cross-talk in multi-qubit superconducting processors",
    "Machine learning approaches for quantum state tomography with limited measurements",
    "Quantum approximate optimization algorithm performance on random 3-SAT instances",
    "Characterization of decoherence channels in trapped-ion quantum processors",
    "Optimal control pulses for two-qubit gates in transmon architectures",
    "Quantum circuit compilation techniques for near-term devices: a survey",
    "Error budget analysis for surface code quantum computing at scale",
    "Hybrid quantum-classical algorithms for combinatorial optimization: current status",
    "Quantum random number generation using photonic integrated circuits",
    "Scalability analysis of neutral atom quantum computing architectures",
    "Quantum annealing performance on MAX-CUT: empirical scaling analysis",
    "Noise-resilient variational quantum algorithms for chemistry applications",
    "Tensor network methods for simulating quantum circuits with 50+ qubits",
    "Quantum error mitigation techniques for NISQ-era computations",
    "Comparative study of qubit connectivity topologies for quantum processors",
    "Quantum machine learning for classification tasks: a benchmark study",
    "Efficient quantum state preparation using parameterized circuits",
    "Quantum computing for financial portfolio optimization: complexity analysis",
    "Superconducting qubit fabrication yield optimization using machine learning",
    "Quantum entanglement distribution over metropolitan fiber networks",
    "Measurement-based quantum computation with cluster states: implementation challenges",
    "Quantum algorithms for linear systems: practical considerations and limitations",
    "Crosstalk mitigation strategies in multi-qubit superconducting processors",
    "Quantum error correction overhead estimates for practical algorithms",
    "Trapped-ion qubit addressing using integrated photonics: design and simulation",
    "Quantum computing resource estimation for cryptographically relevant problems",
    "Adiabatic quantum computation equivalence to circuit model: constructive proof",
    "Quantum sensing with NV centers: sensitivity limits and applications",
    "Quantum communication protocols for distributed quantum computing",
    "Variational quantum simulation of lattice gauge theories",
    "Quantum advantage in sampling problems: updated classical bounds",
    "Fault-tolerant quantum computation with constant overhead: asymptotic analysis",
    "Quantum computing for drug discovery: current limitations and future prospects",
    "Topological quantum error correction codes: a comprehensive review",
    "Quantum walk algorithms for graph problems: complexity analysis",
    "Neutral atom quantum computing: scalability roadmap and engineering challenges",
    "Quantum computing education: curriculum design for undergraduate programs",
    "Benchmarking quantum simulators against exact diagonalization",
    "Quantum computing for climate modeling: feasibility assessment",
    "Quantum algorithms for optimization: QAOA vs quantum annealing comparison",
]

ARXIV_UNRELATED = [
    "Quantum gravity and holographic entanglement entropy in AdS/CFT correspondence",
    "Topological phases and edge states in 2D condensed matter systems",
    "Quantum information scrambling in black hole evaporation models",
    "Bell inequality violations in photonic systems at room temperature: loophole-free test",
    "Quantum key distribution security proofs under realistic channel conditions",
    "Quantum thermodynamics of small systems: work extraction and fluctuation theorems",
    "Quantum chaos in many-body systems: spectral statistics and eigenstate thermalization",
    "Quantum field theory on curved spacetime: Hawking radiation corrections",
    "Quantum biology: coherent energy transfer in photosynthetic complexes",
    "Quantum foundations: new interpretations of the measurement problem",
    "Quantum optics with single atoms in optical cavities: strong coupling regime",
    "Quantum magnetism in frustrated lattice systems: spin liquid candidates",
    "Quantum phase transitions in ultracold atomic gases",
    "Quantum metrology beyond the Heisenberg limit using non-Gaussian states",
    "Quantum simulation of high-energy physics: lattice QCD on quantum hardware",
]

# ============================================================
# QNT Scenarios (Fix 15)
# ============================================================

QNT_SECTOR_WIDE = [
    "US DOE announces $3B trapped-ion quantum computing initiative, funding both academic and commercial programs",
    "Academic paper demonstrates trapped-ion qubits maintaining coherence for 1 hour, a fundamental physics breakthrough",
    "Google announces superconducting processor with 1000 qubits and below-threshold error rates, leapfrogging trapped-ion approaches",
    "Congress passes Quantum Computing Advancement Act with $5B funding over 5 years for all approaches",
    "New theoretical result shows trapped-ion approach has fundamental advantage over superconducting for error correction scaling",
    "China demonstrates 100-qubit trapped-ion processor, intensifying global competition in the trapped-ion space",
    "Major enterprise survey shows 60% of quantum-interested Fortune 500 companies prefer trapped-ion approach",
    "Quantum computing ETF sees $500M inflows in a single week after sector-wide positive sentiment",
    "Jensen Huang says trapped-ion quantum computers will be commercially useful within 3 years at GTC keynote",
    "EU announces $2B quantum computing sovereignty fund with specific carve-out for trapped-ion technology",
    "Short seller publishes report claiming all quantum computing companies are overvalued by 80%",
    "Quantum computing stocks drop 15% sector-wide on macro fears and rising interest rates, no quantum-specific news",
]

QNT_COMPETITIVE = [
    "QNT wins $200M US Air Force contract for quantum computing services, beating IONQ in final round of bidding",
    "IONQ wins $150M contract with JPMorgan for quantum optimization, QNT was the other finalist",
    "QNT announces 99.99% single-qubit gate fidelity on H3 processor, surpassing IONQ's published 99.95% results",
    "IONQ demonstrates 50 algorithmic qubits on Forte Enterprise, maintaining lead over QNT's 40 on H2",
    "QNT reports first quarter as public company: revenue of $45M, beating estimates by 20%",
    "IONQ misses Q3 revenue estimates by 15%, cites delayed enterprise deployments and longer sales cycles",
    "QNT's chief scientist and 3 key researchers leave to join IONQ, citing better research freedom",
    "IONQ announces exclusive partnership with AWS for trapped-ion quantum services on Braket",
    "QNT announces exclusive partnership with Microsoft Azure for enterprise quantum services",
    "Independent benchmark study shows QNT's H3 processor outperforms IONQ's Forte on quantum chemistry workloads",
    "IONQ announces acquisition of a quantum networking startup for $200M, expanding beyond pure computation",
    "QNT raises $500M secondary offering at $80/share to fund next-generation H4 processor manufacturing",
    "IONQ's CEO makes controversial comments about QNT's technology, QNT responds with benchmark data",
    "QNT announces 30% workforce reduction to extend runway and focus on core H-series processor development",
    "Major enterprise customer publicly switches from IONQ to QNT for production workloads, citing better error rates",
    "IONQ patents a novel ion-shuttling technique that QNT's architecture cannot replicate without licensing",
    "QNT announces breakthrough in photonic interconnects enabling modular scaling to 1000+ qubits",
    "Analyst initiates QNT at Overweight with $90 target, IONQ at Underweight citing valuation gap",
]

QNT_MIXED = [
    "IONQ and QNT announce a joint venture to develop quantum networking standards and interoperability protocols",
    "A major trapped-ion patent held by QNT expires, allowing IONQ and others to use the technology freely",
    "QNT's IPO lockup period expires in 180 days, insiders expected to sell significant shares",
    "Both IONQ and QNT miss earnings in the same quarter, raising concerns about trapped-ion commercialization timeline",
    "A new trapped-ion startup backed by SoftBank raises $1B, competing with both IONQ and QNT using a novel architecture",
]


# ============================================================
# Async API Helpers (reused from manus_teacher_concurrent.py)
# ============================================================

async def create_task_async(session, prompt, schema, retries=3):
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
                    print(f"  [WARN] Create failed: {data.get('error', {}).get('message', '?')}")
                    if attempt < retries - 1:
                        await asyncio.sleep(60)
        except Exception as e:
            print(f"  [ERROR] Create exception: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(60)
    return None


async def poll_for_result_async(session, task_id, max_time=MAX_POLL_TIME):
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
                            async with session.get(f"{BASE_URL}/task.listMessages", headers=HEADERS, params=params2) as resp2:
                                data2 = await resp2.json()
                                for m in data2.get("messages", []):
                                    if m.get("type") == "structured_output_result":
                                        return m["structured_output_result"]
                            return {"success": False, "error": "Stopped without output"}
                        elif status == "error":
                            return {"success": False, "error": "Task errored"}
        except Exception as e:
            print(f"  [WARN] Poll exception: {e}")
        await asyncio.sleep(POLL_INTERVAL)
    return {"success": False, "error": "Timeout"}


# ============================================================
# Generation Logic
# ============================================================

_file_lock = asyncio.Lock()

async def append_result(filepath, record):
    async with _file_lock:
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")


async def process_task(session, semaphore, idx, prompt, schema, metadata, output_file, category):
    async with semaphore:
        start_time = time.time()
        print(f"  [{category}][{idx}] Starting...")
        
        task_id = await create_task_async(session, prompt, schema)
        if not task_id:
            record = {**metadata, "article_idx": idx, "category": category,
                      "success": False, "error": "Failed to create task",
                      "timestamp": datetime.now().isoformat()}
            await append_result(output_file, record)
            return record
        
        print(f"  [{category}][{idx}] Task: {task_id}")
        result = await poll_for_result_async(session, task_id)
        elapsed = time.time() - start_time
        
        record = {
            **metadata,
            "article_idx": idx,
            "category": category,
            "task_id": task_id,
            "success": result.get("success", False),
            "signal": result.get("value") if result.get("success") else None,
            "error": result.get("error"),
            "time_seconds": elapsed,
            "timestamp": datetime.now().isoformat()
        }
        
        await append_result(output_file, record)
        status = "SUCCESS" if record["success"] else "FAILED"
        print(f"  [{category}][{idx}] {status} ({elapsed:.0f}s)")
        return record


async def run_arxiv_generation():
    """Generate 70 arXiv training examples."""
    output_file = DATA_TRAINING / "manus_arxiv_rebalance.jsonl"
    
    # Check existing
    existing = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.add(r.get("article_idx"))
    
    print(f"ArXiv generation: {len(existing)} already done")
    
    tasks = []
    
    # 10 important (idx 0-9)
    for i, scenario in enumerate(ARXIV_IMPORTANT):
        if i in existing:
            continue
        prompt = ARXIV_PROMPT.format(
            shared_context=SHARED_CONTEXT,
            market_context="",
            title=scenario,
            date="2025-06-15",
            text=f"Title: {scenario}\n\nThis is a significant research paper with concrete experimental results."
        )
        tasks.append((i, prompt, {"scenario": scenario, "arxiv_tier": "important"}))
    
    # 45 incremental (idx 10-54)
    for i, scenario in enumerate(ARXIV_INCREMENTAL):
        idx = i + 10
        if idx in existing:
            continue
        prompt = ARXIV_PROMPT.format(
            shared_context=SHARED_CONTEXT,
            market_context="",
            title=scenario,
            date="2025-08-01",
            text=f"Title: {scenario}\n\nAbstract: This paper presents incremental improvements in quantum computing methodology. The results are primarily of academic interest with limited near-term commercial implications."
        )
        tasks.append((idx, prompt, {"scenario": scenario, "arxiv_tier": "incremental"}))
    
    # 15 unrelated (idx 55-69)
    for i, scenario in enumerate(ARXIV_UNRELATED):
        idx = i + 55
        if idx in existing:
            continue
        prompt = ARXIV_PROMPT.format(
            shared_context=SHARED_CONTEXT,
            market_context="",
            title=scenario,
            date="2025-09-01",
            text=f"Title: {scenario}\n\nAbstract: This paper addresses fundamental questions in quantum physics that are not related to quantum computing hardware or commercial applications."
        )
        tasks.append((idx, prompt, {"scenario": scenario, "arxiv_tier": "unrelated"}))
    
    print(f"ArXiv: {len(tasks)} tasks to run")
    
    if not tasks:
        print("All arXiv examples already generated!")
        return
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        coros = [
            process_task(session, semaphore, idx, prompt, SIGNAL_SCHEMA, meta, output_file, "arxiv")
            for idx, prompt, meta in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
    
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    print(f"\nArXiv complete: {successes}/{len(tasks)} successful")


async def run_qnt_generation():
    """Generate 35 QNT competitive dynamics examples."""
    output_file = DATA_TRAINING / "manus_qnt_examples.jsonl"
    
    existing = set()
    if output_file.exists():
        with open(output_file) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.add(r.get("article_idx"))
    
    print(f"QNT generation: {len(existing)} already done")
    
    all_scenarios = QNT_SECTOR_WIDE + QNT_COMPETITIVE + QNT_MIXED
    tasks = []
    
    for i, scenario in enumerate(all_scenarios):
        if i in existing:
            continue
        
        # Determine category
        if i < len(QNT_SECTOR_WIDE):
            qnt_type = "sector_wide"
        elif i < len(QNT_SECTOR_WIDE) + len(QNT_COMPETITIVE):
            qnt_type = "competitive"
        else:
            qnt_type = "mixed"
        
        prompt = QNT_SCENARIO_PROMPT.format(
            shared_context=SHARED_CONTEXT,
            scenario=scenario
        )
        tasks.append((i, prompt, {"scenario": scenario, "qnt_type": qnt_type}))
    
    print(f"QNT: {len(tasks)} tasks to run")
    
    if not tasks:
        print("All QNT examples already generated!")
        return
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        coros = [
            process_task(session, semaphore, idx, prompt, SIGNAL_SCHEMA, meta, output_file, "qnt")
            for idx, prompt, meta in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
    
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    print(f"\nQNT complete: {successes}/{len(tasks)} successful")


# ============================================================
# Main
# ============================================================

async def async_main(gen_type):
    if gen_type in ("arxiv", "all"):
        await run_arxiv_generation()
    if gen_type in ("qnt", "all"):
        await run_qnt_generation()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["arxiv", "qnt", "all"], default="all")
    args = parser.parse_args()
    
    print(f"Generating {args.type} examples...")
    print(f"Max concurrent: {MAX_CONCURRENT}")
    print(f"Agent profile: manus-1.6-max")
    print()
    
    asyncio.run(async_main(args.type))


if __name__ == "__main__":
    main()
