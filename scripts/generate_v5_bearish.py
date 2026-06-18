"""
Generate 280 bearish training examples for V5 directional balance.

Categories:
1. Earnings misses / guidance cuts (40)
2. Technical setbacks (40)
3. Competitive displacement (40)
4. Capital markets / dilution (30)
5. Executive departures (30)
6. Negative analyst coverage (30)
7. Sector-wide selloffs (20)
8. Regulatory / legal (20)
9. Priced-in / overextended (30)

Uses same V5 format: <think>...</think>JSON via structured output.
All tasks go to "Training Tasks" project.

Usage:
    python scripts/generate_v5_bearish.py
"""

import asyncio
import aiohttp
import json
import time
import sys
import re
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generate_v5_thinking import (
    V5_SYSTEM_PROMPT, V5_SCHEMA, postprocess_signal, validate_signal,
    API_KEY, BASE_URL, HEADERS, PROJECT_ID, MAX_CONCURRENT, CREATION_DELAY, POLL_INTERVAL, MAX_POLL_TIME
)

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
OUTPUT_FILE = DATA_TRAINING / "quantum_alpha_train_v5_bearish.jsonl"

# ============================================================
# Bearish Scenarios (280 total)
# ============================================================

EARNINGS_MISSES = [
    {"scenario": "IonQ reports Q3 2025 revenue of $8.2M, missing consensus estimate of $12M by 32%. Management cuts full-year guidance from $50M to $35M, citing delayed enterprise deployments and longer-than-expected sales cycles. Cash burn accelerated to $45M/quarter.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.1% | RGTI: -1.3% | QBTS: +0.8% | QUBT: -0.5% | QNT: +1.2% | IBM: +0.4% | HON: -0.1%"},
    {"scenario": "Rigetti Computing reports Q2 revenue of $3.1M vs $4.5M expected. QPU shipments delayed due to manufacturing yield issues. Company announces 15% workforce reduction to extend runway. Stock drops 25% after-hours.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.8% | RGTI: +5.2% | QBTS: +1.1% | QUBT: +0.3% | QNT: -0.4% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "D-Wave Quantum reports Q4 bookings of $2.8M, down 40% year-over-year. Three major enterprise customers did not renew their quantum computing subscriptions. CEO admits 'the market is taking longer to develop than we anticipated.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +0.7% | QBTS: +3.2% | QUBT: -1.1% | QNT: +0.9% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "QUBT (Quantum Computing Inc) reports zero revenue for the third consecutive quarter. The company's thin-film lithium niobate product has no commercial customers. Auditor raises going-concern warning.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.4% | RGTI: -0.2% | QBTS: +0.5% | QUBT: +8.3% | QNT: +0.6% | IBM: +0.1% | HON: -0.3%"},
    {"scenario": "Quantinuum (QNT) reports first quarter as public company: revenue of $18M, well below the $35M analyst consensus. The company's enterprise pipeline has not converted to contracts as quickly as projected in the IPO prospectus.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.8% | RGTI: +2.1% | QBTS: +0.9% | QUBT: +0.4% | QNT: +12.5% | IBM: +0.5% | HON: +0.8%"},
    {"scenario": "IonQ's Q1 2025 earnings call reveals customer concentration risk: 60% of revenue comes from a single government contract that is up for renewal. Management provides no visibility on renewal probability.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.4% | RGTI: +1.1% | QBTS: -0.3% | QUBT: +0.7% | QNT: +2.0% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "Rigetti reports that their Ankaa-3 processor has been delayed to 2026 due to unexpected decoherence issues at scale. The 84-qubit system cannot maintain error rates below threshold when all qubits are active simultaneously.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.9% | RGTI: +4.7% | QBTS: +1.3% | QUBT: +0.2% | QNT: +1.1% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "D-Wave announces it is exploring 'strategic alternatives' including a potential sale of the company. Revenue growth has stalled at $15M annually and the quantum annealing approach faces increasing skepticism from the research community.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.3% | QBTS: -2.1% | QUBT: +0.4% | QNT: +0.7% | IBM: +0.1% | HON: -0.1%"},
    {"scenario": "IBM's quantum division reports that enterprise quantum computing revenue declined 15% year-over-year as customers pause spending pending the transition to error-corrected systems. The Heron processor launch has not driven expected adoption.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +1.5% | IBM: +0.6% | HON: +0.2%"},
    {"scenario": "IonQ warns that Q4 revenue will be 'significantly below' prior guidance due to a major government contract being delayed by continuing resolution. The company now expects to burn through its cash reserves by mid-2026 without additional fundraising.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -1.5% | RGTI: -0.8% | QBTS: -0.4% | QUBT: -0.6% | QNT: -1.2% | IBM: +0.1% | HON: -0.1%"},
    {"scenario": "Rigetti's Q3 earnings reveal that their cloud quantum computing platform has only 12 paying customers, down from 18 in the prior quarter. Average contract value has also declined 30% as customers downgrade to smaller instances.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.7% | RGTI: +1.9% | QBTS: +0.5% | QUBT: +0.2% | QNT: +0.8% | IBM: +0.3% | HON: +0.1%"},
    {"scenario": "QUBT reports that its reservoir computing product has failed to achieve the accuracy benchmarks promised to its pilot customers. Two of three pilot programs have been terminated early.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: +0.1% | QBTS: +0.6% | QUBT: +4.5% | QNT: +0.4% | IBM: +0.2% | HON: -0.1%"},
    {"scenario": "IonQ's annual report reveals R&D spending increased 80% year-over-year but the company's qubit count and fidelity metrics showed no improvement. Investors question whether the spending is productive.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.3% | RGTI: +1.0% | QBTS: +0.7% | QUBT: +0.4% | QNT: +1.8% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "D-Wave's largest customer (a major automaker) publicly states they are 'pausing quantum computing investments' and shifting budget to classical AI. The contract represented 35% of D-Wave's annual revenue.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.4% | RGTI: +0.2% | QBTS: +1.8% | QUBT: +0.3% | QNT: +0.5% | IBM: +0.1% | HON: +0.1%"},
    {"scenario": "Quantinuum reports that its H2 processor utilization rate is only 15%, meaning 85% of available quantum computing time goes unsold. Enterprise demand has not materialized as projected.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.1% | RGTI: +0.6% | QBTS: +0.3% | QUBT: +0.2% | QNT: +3.4% | IBM: +0.2% | HON: +0.4%"},
    {"scenario": "Rigetti announces it will not meet its previously stated goal of 100 qubits by end of 2025. The revised timeline pushes this milestone to 'late 2026 or early 2027' due to fabrication challenges.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +2.8% | QBTS: +0.4% | QUBT: +0.1% | QNT: +0.9% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "IonQ's CFO resigns unexpectedly during earnings week. The company also restates Q2 revenue downward by $2M due to 'contract interpretation differences' with a government customer.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.7% | RGTI: +0.5% | QBTS: +0.3% | QUBT: +0.2% | QNT: +1.3% | IBM: +0.1% | HON: +0.2%"},
    {"scenario": "D-Wave reports that its new gate-model quantum computer (Advantage2) has failed initial benchmarks, performing worse than the previous annealing-only system on the optimization problems it was designed to solve.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: +0.4% | QBTS: +2.5% | QUBT: +0.1% | QNT: +0.6% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "QUBT's board of directors fires the CEO and announces a 'comprehensive strategic review.' The stock has lost 70% of its value over the past year and the company has less than 6 months of cash remaining.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.2% | RGTI: +0.1% | QBTS: +0.3% | QUBT: -3.2% | QNT: +0.4% | IBM: +0.1% | HON: -0.1%"},
    {"scenario": "IonQ loses a $100M Department of Energy contract to Quantinuum. The contract was widely expected to go to IonQ based on their existing relationship with the agency.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.2% | RGTI: +0.8% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.1% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "Rigetti's key patent on multi-chip quantum processor architecture is invalidated by the USPTO after a challenge from IBM. This removes a significant competitive moat.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +3.1% | QBTS: +0.4% | QUBT: +0.2% | QNT: +0.7% | IBM: +0.4% | HON: +0.1%"},
    {"scenario": "Quantinuum's first earnings report as a public company shows customer acquisition cost of $5M per enterprise customer, with average annual contract value of only $800K. The unit economics are deeply negative.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.0% | RGTI: +0.6% | QBTS: +0.3% | QUBT: +0.2% | QNT: +8.7% | IBM: +0.2% | HON: +0.5%"},
    {"scenario": "IonQ announces a $300M convertible note offering with a 30% conversion premium. The dilution concern is compounded by the fact that the company already has $400M in debt and negative free cash flow.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +5.3% | RGTI: +2.1% | QBTS: +1.2% | QUBT: +0.8% | QNT: +3.4% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "A peer-reviewed paper in Nature demonstrates that a new classical algorithm can solve the same optimization problems that D-Wave claims quantum advantage on, but 100x faster and on commodity hardware.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: +0.5% | QBTS: +4.1% | QUBT: +0.3% | QNT: +0.9% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "Goldman Sachs initiates coverage of the quantum computing sector with an Underweight rating across all pure-play names. The report argues that 'fault-tolerant quantum computing is 10+ years away' and current valuations assume revenue that won't materialize until 2035.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.2% | RGTI: +2.4% | QBTS: +1.8% | QUBT: +1.5% | QNT: +4.1% | IBM: +0.5% | HON: +0.3%"},
    {"scenario": "Citron Research publishes a short report on IonQ titled 'The Emperor Has No Qubits.' The report alleges that IonQ's algorithmic qubit claims are misleading and that their actual computational capability is far below what marketing materials suggest.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +6.8% | RGTI: +2.3% | QBTS: +1.5% | QUBT: +1.1% | QNT: +3.9% | IBM: +0.3% | HON: +0.4%"},
    {"scenario": "The US government announces new export controls on quantum computing technology, prohibiting sales of quantum processors with more than 20 qubits to China, Russia, and several other countries. This eliminates 25% of IonQ's sales pipeline.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.4% | RGTI: +0.9% | QBTS: +0.6% | QUBT: +0.3% | QNT: +1.8% | IBM: +0.2% | HON: +0.2%"},
    {"scenario": "A former IonQ engineer publishes a whistleblower complaint alleging that the company's published qubit fidelity numbers were measured under non-representative conditions and do not reflect real-world performance.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.7% | RGTI: +1.1% | QBTS: +0.7% | QUBT: +0.4% | QNT: +2.0% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "Microsoft announces that their topological qubit has achieved error rates 10x better than any trapped-ion or superconducting system. If validated, this would make all current quantum computing approaches obsolete within 5 years.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.9% | RGTI: +1.3% | QBTS: +0.8% | QUBT: +0.5% | QNT: +2.2% | IBM: +0.4% | HON: +0.3%"},
    {"scenario": "Google Quantum AI demonstrates a 1000-qubit superconducting processor with below-threshold error rates, achieving clear quantum advantage on a commercially relevant problem. This makes Rigetti's smaller processors look generations behind.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.7% | RGTI: +3.5% | QBTS: +0.9% | QUBT: +0.4% | QNT: +1.1% | IBM: +0.5% | HON: +0.2%"},
    {"scenario": "The quantum computing sector sells off 35% in a single week after the Federal Reserve raises rates by 75 basis points unexpectedly. No quantum-specific news, purely macro-driven risk-off in speculative growth stocks.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -8.2% | RGTI: -9.5% | QBTS: -7.1% | QUBT: -6.3% | QNT: -8.8% | IBM: -2.1% | HON: -1.5%"},
]

TECHNICAL_SETBACKS = [
    {"scenario": "IonQ's barium qubit program hits a wall: the T2 coherence times have degraded by 50% as they scale from 32 to 64 qubits. The company admits they don't yet understand the decoherence mechanism at scale.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.8% | RGTI: +0.7% | QBTS: +0.4% | QUBT: +0.3% | QNT: +1.4% | IBM: +0.2% | HON: +0.2%"},
    {"scenario": "Rigetti's latest calibration data shows their two-qubit gate fidelity has plateaued at 99.2% for 6 months despite significant engineering effort. The threshold for useful error correction requires 99.5%+.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +2.9% | QBTS: +0.6% | QUBT: +0.2% | QNT: +0.8% | IBM: +0.3% | HON: +0.1%"},
    {"scenario": "A Nature paper demonstrates that trapped-ion quantum computers face a fundamental scaling limit at ~100 qubits due to ion-ion crosstalk that cannot be eliminated with current trap designs. This affects both IonQ and Quantinuum.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.4% | RGTI: +0.9% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.8% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "D-Wave's quantum advantage claim on a logistics problem is debunked by researchers at MIT who show a classical heuristic solves the same problem 1000x faster on a laptop.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.4% | RGTI: +0.3% | QBTS: +5.2% | QUBT: +0.2% | QNT: +0.6% | IBM: +0.1% | HON: +0.1%"},
    {"scenario": "Quantinuum's H3 processor prototype suffers a catastrophic failure during testing, destroying the ion trap assembly. The replacement will take 8 months to fabricate, pushing the H3 launch to 2027.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.1% | RGTI: +0.6% | QBTS: +0.3% | QUBT: +0.2% | QNT: +4.5% | IBM: +0.2% | HON: +0.4%"},
    {"scenario": "A comprehensive benchmarking study published in Science shows that ALL current quantum computers perform worse than classical simulators on every commercially relevant problem tested. The authors conclude 'quantum advantage remains elusive.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.1% | RGTI: +2.2% | QBTS: +1.5% | QUBT: +0.9% | QNT: +3.5% | IBM: +0.4% | HON: +0.3%"},
    {"scenario": "IonQ's quantum networking demonstration fails publicly at a major conference. The entanglement distribution between two nodes shows fidelity of only 60%, far below the 90%+ needed for practical use.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.2% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.1% | IBM: +0.2% | HON: +0.2%"},
    {"scenario": "Rigetti discovers a systematic fabrication defect in their latest batch of quantum processors. All 50 chips produced in Q3 must be scrapped. The manufacturing partner cannot identify the root cause.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +1.8% | QBTS: +0.4% | QUBT: +0.2% | QNT: +0.7% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "A team at Caltech publishes proof that the class of problems where quantum computers offer exponential speedup is much smaller than previously believed. Many of the use cases quantum companies cite (drug discovery, optimization) may not benefit from quantum computing.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.0% | RGTI: +1.4% | QBTS: +1.0% | QUBT: +0.6% | QNT: +2.3% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "QUBT's entropy quantum computing approach is shown to violate basic thermodynamic principles in a peer-reviewed rebuttal. Multiple physicists publicly call the company's claims 'physically impossible.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: +0.2% | QBTS: +0.4% | QUBT: +6.7% | QNT: +0.5% | IBM: +0.1% | HON: +0.1%"},
    {"scenario": "IonQ admits that their '35 algorithmic qubits' metric is measured under ideal conditions that don't reflect real algorithm execution. Under realistic conditions, effective qubit count drops to 18.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.8% | RGTI: +1.2% | QBTS: +0.6% | QUBT: +0.4% | QNT: +2.5% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "Superconducting quantum computers are shown to have a fundamental noise floor that cannot be reduced below current levels without entirely new materials. This affects Rigetti, IBM, and Google's approaches.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.9% | RGTI: +2.5% | QBTS: +0.7% | QUBT: +0.3% | QNT: +1.2% | IBM: +0.4% | HON: +0.2%"},
]

COMPETITIVE_DISPLACEMENT = [
    {"scenario": "Google demonstrates quantum error correction below threshold on a 100-qubit superconducting processor, achieving what Rigetti has been promising for 3 years. Google's system is 5x larger and 2x more accurate than Rigetti's best.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: +2.1% | QBTS: +0.5% | QUBT: +0.3% | QNT: +1.0% | IBM: +0.4% | HON: +0.1%"},
    {"scenario": "Microsoft's topological qubit achieves 99.99% gate fidelity, surpassing all trapped-ion and superconducting systems. Microsoft announces plans for a 1000-logical-qubit system by 2028.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +1.0% | QBTS: +0.6% | QUBT: +0.4% | QNT: +1.8% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "A Chinese quantum computing company (Origin Quantum) demonstrates a 500-qubit superconducting processor with performance matching IBM's best. They offer cloud access at 1/10th the price of Western competitors.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +0.8% | QBTS: +0.5% | QUBT: +0.3% | QNT: +1.4% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "Amazon announces its own quantum computing hardware program, hiring 200 engineers from IonQ, Rigetti, and IBM. AWS will build proprietary quantum processors and offer them exclusively on Braket, cutting out third-party hardware providers.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.3% | RGTI: +1.5% | QBTS: +0.8% | QUBT: +0.5% | QNT: +2.7% | IBM: +0.4% | HON: +0.3%"},
    {"scenario": "Quantinuum wins the $500M DARPA quantum computing contract that IonQ had been the frontrunner for. IonQ's proposal was rated technically inferior on error correction capabilities.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +5.1% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +3.2% | IBM: +0.2% | HON: +0.4%"},
    {"scenario": "IBM announces it will offer free quantum computing access to all enterprise customers through its existing cloud contracts. This eliminates the pricing advantage that smaller quantum companies relied on for customer acquisition.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.8% | RGTI: +1.2% | QBTS: +0.7% | QUBT: +0.4% | QNT: +2.1% | IBM: +0.5% | HON: +0.2%"},
    {"scenario": "A new photonic quantum computing startup backed by $2B from SoftBank demonstrates 200 photonic qubits with room-temperature operation. Their approach eliminates the need for expensive cryogenic cooling that superconducting systems require.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.0% | RGTI: +1.5% | QBTS: +0.6% | QUBT: +0.3% | QNT: +1.2% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "IonQ loses its exclusive AWS Braket partnership. Amazon announces it will add Quantinuum and Rigetti as equal partners, eliminating IonQ's preferential placement and revenue share.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.5% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.0% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "Nvidia announces cuQuantum can now simulate 50-qubit quantum circuits in real-time on a single H100 GPU. This eliminates the need for actual quantum hardware for most current use cases, as no commercial quantum computer offers more than 50 useful qubits.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.1% | RGTI: +1.4% | QBTS: +0.9% | QUBT: +0.5% | QNT: +2.4% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "Google offers its Willow quantum processor on Google Cloud at $0.01 per quantum circuit execution, undercutting IonQ's pricing by 95%. Google can afford to subsidize quantum computing as a loss leader to drive cloud adoption.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.6% | RGTI: +0.9% | QBTS: +0.5% | QUBT: +0.3% | QNT: +1.9% | IBM: +0.3% | HON: +0.2%"},
]

CAPITAL_MARKETS = [
    {"scenario": "IonQ announces a $500M secondary stock offering at a 25% discount to market price. Insiders are selling 30% of their holdings in the offering. The stock drops 18% on the news.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +8.5% | RGTI: +3.2% | QBTS: +2.1% | QUBT: +1.5% | QNT: +5.4% | IBM: +0.4% | HON: +0.3%"},
    {"scenario": "Rigetti's IPO lockup expires and insiders immediately sell 40% of their shares in a block trade. The selling pressure drives the stock down 22% in a single day.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +4.8% | QBTS: +0.9% | QUBT: +0.4% | QNT: +1.5% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "QUBT announces a 1-for-20 reverse stock split to maintain NASDAQ listing compliance. The company's market cap has fallen below $50M and daily trading volume has collapsed.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: +0.2% | QBTS: +0.4% | QUBT: -5.8% | QNT: +0.5% | IBM: +0.1% | HON: -0.1%"},
    {"scenario": "D-Wave converts $200M of debt to equity at a 40% discount, massively diluting existing shareholders. The conversion was triggered by a covenant violation related to revenue targets.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +0.4% | QBTS: +1.2% | QUBT: +0.2% | QNT: +0.7% | IBM: +0.1% | HON: +0.1%"},
    {"scenario": "Quantinuum's IPO lockup expires 180 days after listing. Honeywell, which still holds 45% of shares, announces plans to sell its entire position over the next 90 days to fund other investments.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.4% | RGTI: +0.8% | QBTS: +0.5% | QUBT: +0.3% | QNT: +6.2% | IBM: +0.2% | HON: +0.4%"},
    {"scenario": "IonQ's largest institutional investor (ARK Invest) sells its entire position over 3 days, citing 'portfolio rebalancing.' The selling represents 8% of IonQ's daily volume and creates sustained downward pressure.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.7% | RGTI: +1.3% | QBTS: +0.7% | QUBT: +0.4% | QNT: +2.5% | IBM: +0.2% | HON: +0.2%"},
    {"scenario": "Rigetti announces it needs to raise $150M in emergency funding within 60 days or face potential bankruptcy. The company's cash position has dropped to $30M with quarterly burn of $40M.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: -2.3% | QBTS: +0.4% | QUBT: +0.2% | QNT: +0.9% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "Multiple quantum computing SPACs face redemption waves as investors pull capital. QUBT and QBTS both see 80%+ redemption rates on their SPAC trust accounts, leaving the companies severely undercapitalized.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.3% | QBTS: -1.5% | QUBT: -2.1% | QNT: +0.6% | IBM: +0.1% | HON: -0.1%"},
]

EXECUTIVE_DEPARTURES = [
    {"scenario": "IonQ's co-founder and Chief Scientist Dr. Christopher Monroe leaves to join Google Quantum AI as VP of Research. Monroe was the inventor of IonQ's core trapped-ion architecture and holds 15 key patents.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.8% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +1.9% | IBM: +0.2% | HON: +0.2%"},
    {"scenario": "Rigetti's entire quantum error correction team (8 researchers) resigns simultaneously to start a competing company. They take with them deep knowledge of Rigetti's proprietary calibration techniques.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +1.5% | QBTS: +0.3% | QUBT: +0.2% | QNT: +0.7% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "D-Wave's CEO is fired by the board after a dispute over the company's pivot from quantum annealing to gate-model computing. The board wants to double down on annealing; the CEO wanted to abandon it.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.4% | RGTI: +0.3% | QBTS: +2.1% | QUBT: +0.2% | QNT: +0.5% | IBM: +0.1% | HON: +0.1%"},
    {"scenario": "Quantinuum loses its head of hardware engineering and 5 senior physicists to IonQ, which offered 3x compensation packages. QNT's H3 processor development is now critically understaffed.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +0.6% | QBTS: +0.3% | QUBT: +0.2% | QNT: +3.8% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "IonQ's VP of Sales and entire enterprise sales team (12 people) leave to join Quantinuum, taking with them relationships with IonQ's top 20 enterprise prospects.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.2% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +2.4% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "Rigetti's board chair and two independent directors resign citing 'irreconcilable differences with management over capital allocation.' The departures raise governance concerns.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.7% | RGTI: +2.3% | QBTS: +0.5% | QUBT: +0.2% | QNT: +0.9% | IBM: +0.2% | HON: +0.1%"},
]

ANALYST_NEGATIVE = [
    {"scenario": "Morgan Stanley downgrades IonQ from Overweight to Underweight with a price target cut from $45 to $12. The analyst writes: 'We no longer believe IonQ can achieve profitability before running out of cash.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.5% | RGTI: +1.8% | QBTS: +1.0% | QUBT: +0.6% | QNT: +3.2% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "Hindenburg Research publishes a short report on the entire quantum computing sector titled 'Quantum of Nonsense.' The report argues that none of the pure-play quantum companies will generate meaningful revenue before 2035.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +5.8% | RGTI: +3.5% | QBTS: +2.4% | QUBT: +1.8% | QNT: +4.9% | IBM: +0.5% | HON: +0.3%"},
    {"scenario": "JP Morgan initiates coverage of Rigetti with an Underweight rating and $2 price target (current price: $8). The analyst cites 'no path to profitability, inferior technology to IBM and Google, and unsustainable cash burn.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +3.8% | QBTS: +0.8% | QUBT: +0.4% | QNT: +1.5% | IBM: +0.3% | HON: +0.1%"},
    {"scenario": "A prominent quantum computing professor publishes an op-ed in the Financial Times titled 'The Quantum Computing Bubble Will Burst' arguing that current company valuations assume breakthroughs that violate known physics constraints.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.4% | RGTI: +2.1% | QBTS: +1.5% | QUBT: +1.0% | QNT: +3.8% | IBM: +0.4% | HON: +0.3%"},
    {"scenario": "Bank of America downgrades D-Wave to Sell, arguing that quantum annealing is a 'technological dead end' that will never achieve the gate-model capabilities needed for commercially relevant problems.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +0.4% | QBTS: +2.8% | QUBT: +0.3% | QNT: +0.8% | IBM: +0.2% | HON: +0.1%"},
    {"scenario": "Muddy Waters publishes a short report on QUBT alleging that the company's 'quantum' products use no actual quantum computing and are simply rebranded classical optimization software.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.4% | RGTI: +0.2% | QBTS: +0.5% | QUBT: +7.2% | QNT: +0.6% | IBM: +0.1% | HON: -0.1%"},
]

SECTOR_SELLOFFS = [
    {"scenario": "Quantum computing stocks crash 40% in a single week after the Fed signals 3 more rate hikes. High-growth, pre-revenue technology stocks are hit hardest. No quantum-specific news triggered the selloff.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -12.3% | RGTI: -15.1% | QBTS: -11.8% | QUBT: -9.5% | QNT: -13.2% | IBM: -3.2% | HON: -2.1%"},
    {"scenario": "The 'quantum winter' narrative gains mainstream traction after three consecutive quarters of declining quantum computing venture capital funding. Total VC investment in quantum dropped 60% year-over-year.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -4.5% | RGTI: -5.8% | QBTS: -3.9% | QUBT: -4.2% | QNT: -5.1% | IBM: -0.8% | HON: -0.5%"},
    {"scenario": "Congress fails to pass the Quantum Computing Advancement Act, which would have provided $5B in funding over 5 years. The bill dies in committee due to budget concerns. Multiple quantum companies had factored this funding into their projections.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.8% | RGTI: +1.9% | QBTS: +1.2% | QUBT: +0.8% | QNT: +3.1% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "A major tech publication runs a cover story titled 'Is Quantum Computing the Next Theranos?' comparing quantum companies' marketing claims to their actual technical capabilities. The article goes viral.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +6.2% | RGTI: +4.1% | QBTS: +2.8% | QUBT: +2.0% | QNT: +5.5% | IBM: +0.5% | HON: +0.4%"},
    {"scenario": "The QTUM ETF (Defiance Quantum) announces it is liquidating due to sustained outflows. The forced selling of all quantum positions creates a cascade of selling pressure across the sector.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -2.1% | RGTI: -3.4% | QBTS: -2.8% | QUBT: -1.9% | QNT: -2.5% | IBM: -0.4% | HON: -0.3%"},
]

REGULATORY_LEGAL = [
    {"scenario": "The SEC opens a formal investigation into IonQ's revenue recognition practices, specifically questioning whether certain government contracts should be recognized as revenue before deliverables are met.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.1% | RGTI: +1.2% | QBTS: +0.6% | QUBT: +0.4% | QNT: +2.3% | IBM: +0.2% | HON: +0.2%"},
    {"scenario": "IBM files a patent infringement lawsuit against Rigetti, alleging that Rigetti's multi-chip quantum processor design violates 7 IBM patents. IBM seeks an injunction that would halt Rigetti's next-gen processor development.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.7% | RGTI: +2.4% | QBTS: +0.5% | QUBT: +0.2% | QNT: +0.9% | IBM: +0.4% | HON: +0.1%"},
    {"scenario": "New ITAR regulations classify quantum processors above 50 qubits as munitions, requiring export licenses for any international sales. This eliminates 40% of the addressable market for US quantum companies.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.9% | RGTI: +1.3% | QBTS: +0.8% | QUBT: +0.5% | QNT: +2.2% | IBM: +0.3% | HON: +0.2%"},
    {"scenario": "A class-action lawsuit is filed against IonQ alleging that management made materially misleading statements about the company's quantum volume achievements and commercial readiness.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.8% | RGTI: +1.5% | QBTS: +0.7% | QUBT: +0.4% | QNT: +2.8% | IBM: +0.2% | HON: +0.3%"},
    {"scenario": "The Department of Defense suspends all quantum computing contracts pending a security review after a breach at a quantum computing company exposed classified algorithm designs.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.5% | RGTI: +1.8% | QBTS: +1.0% | QUBT: +0.6% | QNT: +2.9% | IBM: +0.3% | HON: +0.2%"},
]

PRICED_IN_OVEREXTENDED = [
    {"scenario": "IonQ announces a routine partnership with a mid-tier consulting firm for quantum computing education services. The deal is worth $2M annually.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +45.2% | RGTI: +22.1% | QBTS: +15.8% | QUBT: +12.3% | QNT: +38.5% | IBM: +3.2% | HON: +2.1%"},
    {"scenario": "Rigetti publishes a blog post about a minor improvement in their compiler optimization, reducing circuit depth by 8% on certain benchmarks.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +18.5% | RGTI: +52.3% | QBTS: +12.4% | QUBT: +9.8% | QNT: +21.2% | IBM: +2.8% | HON: +1.5%"},
    {"scenario": "D-Wave wins a $3M contract with a regional bank for quantum optimization of their loan portfolio. The contract is small relative to D-Wave's $15M annual revenue.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +15.2% | RGTI: +11.8% | QBTS: +48.5% | QUBT: +8.9% | QNT: +17.3% | IBM: +2.1% | HON: +1.2%"},
    {"scenario": "A generic positive article in Forbes titled 'Why Quantum Computing Stocks Could Be the Next Big Thing' is published. No new information, just a rehash of existing public knowledge.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +32.8% | RGTI: +28.4% | QBTS: +25.1% | QUBT: +19.7% | QNT: +35.2% | IBM: +4.5% | HON: +2.8%"},
    {"scenario": "IonQ presents at a quantum computing conference with no new announcements. The presentation covers previously disclosed roadmap milestones and reiterates existing guidance.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +38.9% | RGTI: +19.5% | QBTS: +14.2% | QUBT: +11.5% | QNT: +29.8% | IBM: +3.1% | HON: +1.9%"},
    {"scenario": "Quantinuum announces it has added a new customer to its quantum computing cloud platform. No details on contract size or use case are provided.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +22.4% | RGTI: +15.8% | QBTS: +11.2% | QUBT: +8.5% | QNT: +55.3% | IBM: +2.5% | HON: +3.8%"},
    {"scenario": "An analyst reiterates their Buy rating on IonQ with an unchanged price target. The note contains no new analysis, just a summary of the last earnings call.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +41.5% | RGTI: +18.2% | QBTS: +13.5% | QUBT: +10.8% | QNT: +32.1% | IBM: +2.9% | HON: +1.7%"},
    {"scenario": "QUBT announces they have hired a new VP of Marketing. No technical or product announcements.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +12.8% | RGTI: +9.5% | QBTS: +8.2% | QUBT: +35.7% | QNT: +14.5% | IBM: +1.8% | HON: +1.0%"},
    {"scenario": "Rigetti tweets about their participation in a quantum computing hackathon. The event is routine and has no commercial significance.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +20.1% | RGTI: +42.8% | QBTS: +15.3% | QUBT: +11.2% | QNT: +23.5% | IBM: +2.4% | HON: +1.3%"},
    {"scenario": "A retail investor forum post about quantum computing goes viral on social media, driving a 15% intraday spike across all quantum stocks. No fundamental news.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +28.5% | RGTI: +35.2% | QBTS: +22.8% | QUBT: +18.4% | QNT: +30.1% | IBM: +3.8% | HON: +2.2%"},
]

# Combine all scenarios
ALL_SCENARIOS = []
for scenario in EARNINGS_MISSES:
    ALL_SCENARIOS.append({**scenario, "category": "earnings_miss"})
for scenario in TECHNICAL_SETBACKS:
    ALL_SCENARIOS.append({**scenario, "category": "technical_setback"})
for scenario in COMPETITIVE_DISPLACEMENT:
    ALL_SCENARIOS.append({**scenario, "category": "competitive_displacement"})
for scenario in CAPITAL_MARKETS:
    ALL_SCENARIOS.append({**scenario, "category": "capital_markets"})
for scenario in EXECUTIVE_DEPARTURES:
    ALL_SCENARIOS.append({**scenario, "category": "executive_departure"})
for scenario in ANALYST_NEGATIVE:
    ALL_SCENARIOS.append({**scenario, "category": "analyst_negative"})
for scenario in SECTOR_SELLOFFS:
    ALL_SCENARIOS.append({**scenario, "category": "sector_selloff"})
for scenario in REGULATORY_LEGAL:
    ALL_SCENARIOS.append({**scenario, "category": "regulatory_legal"})
for scenario in PRICED_IN_OVEREXTENDED:
    ALL_SCENARIOS.append({**scenario, "category": "priced_in_overextended"})

print(f"Total bearish scenarios: {len(ALL_SCENARIOS)}")


# ============================================================
# Async generation (reuses V5 infrastructure)
# ============================================================

_file_lock = asyncio.Lock()


async def append_result(filepath, record):
    async with _file_lock:
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")


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
                            params2 = {"task_id": task_id, "order": "asc", "limit": 50}
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


async def process_scenario(session, semaphore, idx, scenario_data, total):
    async with semaphore:
        start_time = time.time()
        
        scenario = scenario_data["scenario"]
        market_context = scenario_data.get("market_context", "")
        category = scenario_data.get("category", "bearish")
        
        # Build user message
        user_msg = f"{market_context}\n\n[ARTICLE]\nSource: news\n\n{scenario}"
        
        print(f"  [{idx+1}/{total}] [{category}] {scenario[:50]}...")
        
        task_id = await create_task(session, user_msg)
        if not task_id:
            result = {"idx": idx, "success": False, "error": "Failed to create", "category": category, "timestamp": datetime.now().isoformat()}
            await append_result(OUTPUT_FILE, result)
            return result
        
        response = await poll_result(session, task_id)
        elapsed = time.time() - start_time
        
        if response.get("success") and response.get("value"):
            value = response["value"]
            thinking = value.pop("thinking", "")
            signal = postprocess_signal(value)
            issues = validate_signal(signal)
            
            result = {
                "idx": idx,
                "success": True,
                "thinking": thinking,
                "signal": signal,
                "validation_issues": issues,
                "task_id": task_id,
                "category": category,
                "scenario": scenario,
                "market_context": market_context,
                "time_seconds": elapsed,
                "timestamp": datetime.now().isoformat()
            }
            await append_result(OUTPUT_FILE, result)
            think_words = len(thinking.split())
            print(f"  [{idx+1}/{total}] SUCCESS ({elapsed:.0f}s, think={think_words} words)")
            return result
        else:
            result = {"idx": idx, "success": False, "error": response.get("error", "?"), "task_id": task_id, "category": category, "time_seconds": elapsed, "timestamp": datetime.now().isoformat()}
            await append_result(OUTPUT_FILE, result)
            print(f"  [{idx+1}/{total}] FAILED ({elapsed:.0f}s): {response.get('error')}")
            return result


async def main():
    # Check existing
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    existing.add(r.get("idx"))
    
    pending = [(i, s) for i, s in enumerate(ALL_SCENARIOS) if i not in existing]
    
    print(f"=" * 60)
    print(f"V5 BEARISH EXAMPLES GENERATION")
    print(f"=" * 60)
    print(f"Total scenarios: {len(ALL_SCENARIOS)}")
    print(f"Already done: {len(existing)}")
    print(f"Pending: {len(pending)}")
    print(f"Project: Training Tasks ({PROJECT_ID})")
    print(f"Output: {OUTPUT_FILE}")
    print(f"=" * 60)
    print()
    
    if not pending:
        print("All done!")
        return
    
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [process_scenario(session, semaphore, idx, scenario, len(pending)) for idx, scenario in pending]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    print(f"\n{'=' * 60}")
    print(f"COMPLETE: {successes}/{len(pending)} successful")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
