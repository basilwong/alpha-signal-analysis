"""
Generate batch 2 of bearish training examples (180 scenarios).
Variations on batch 1 themes + new priced-in/overextended scenarios.

Usage:
    python scripts/generate_v5_bearish_batch2.py
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
OUTPUT_FILE = DATA_TRAINING / "quantum_alpha_train_v5_bearish_b2.jsonl"

# ============================================================
# Batch 2 Scenarios (180 total)
# ============================================================

SCENARIOS = [
    # --- EARNINGS MISSES (company variations) ---
    {"scenario": "Rigetti reports Q1 revenue of $2.8M, missing the $5M consensus by 44%. The company's QPU-as-a-service model is not gaining traction. Management admits 'enterprise adoption is slower than we modeled.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +3.8% | QBTS: +0.9% | QUBT: +0.4% | QNT: +1.5% | IBM: +0.3% | HON: +0.1%", "category": "earnings_miss"},
    {"scenario": "QNT (Quantinuum) reports Q2 revenue of $22M vs $40M expected. The company's enterprise pipeline has stalled as potential customers wait for error-corrected systems before committing to multi-year contracts.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.1% | RGTI: +1.4% | QBTS: +0.7% | QUBT: +0.4% | QNT: +7.8% | IBM: +0.3% | HON: +0.5%", "category": "earnings_miss"},
    {"scenario": "IonQ's Q2 results show revenue of $10M (in-line) but gross margins collapsed from 60% to 25% due to heavy discounting to win competitive deals against Quantinuum. Management says pricing pressure will continue.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.5% | RGTI: +0.9% | QBTS: +0.5% | QUBT: +0.3% | QNT: +3.1% | IBM: +0.2% | HON: +0.3%", "category": "earnings_miss"},
    {"scenario": "D-Wave reports that their quantum cloud revenue declined 25% sequentially as three Fortune 500 customers terminated their subscriptions, citing inability to demonstrate quantum advantage over classical alternatives.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: +0.5% | QBTS: +4.2% | QUBT: +0.3% | QNT: +1.0% | IBM: +0.2% | HON: +0.1%", "category": "earnings_miss"},
    {"scenario": "QUBT announces that it has exhausted its cash reserves and is seeking emergency bridge financing. The company has no revenue and its technology demonstrations have failed to attract customers.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.3% | QBTS: +0.4% | QUBT: +2.8% | QNT: +0.6% | IBM: +0.1% | HON: -0.1%", "category": "earnings_miss"},
    {"scenario": "Rigetti's annual report reveals that customer retention rate has dropped from 85% to 45%. Most customers who tried quantum computing on Rigetti's platform did not renew after their initial contract.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.9% | RGTI: +2.5% | QBTS: +0.6% | QUBT: +0.2% | QNT: +1.1% | IBM: +0.2% | HON: +0.1%", "category": "earnings_miss"},
    {"scenario": "IonQ discloses that its largest customer (representing 40% of revenue) has notified them of intent to terminate their contract in 90 days. The customer is switching to Quantinuum's H2 processor.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.8% | RGTI: +1.2% | QBTS: +0.6% | QUBT: +0.3% | QNT: +2.5% | IBM: +0.2% | HON: +0.3%", "category": "earnings_miss"},
    {"scenario": "QNT reports that its backlog has shrunk from $200M to $120M as several government contracts were delayed indefinitely due to budget sequestration. The company cuts 2025 revenue guidance by 35%.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.2% | QNT: +5.5% | IBM: +0.2% | HON: +0.4%", "category": "earnings_miss"},
    {"scenario": "D-Wave's Q3 earnings call reveals that their average deal size has shrunk from $500K to $150K as customers downgrade from production workloads to 'exploration only' contracts.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +0.4% | QBTS: +3.5% | QUBT: +0.2% | QNT: +0.8% | IBM: +0.1% | HON: +0.1%", "category": "earnings_miss"},
    {"scenario": "Rigetti warns that it will breach its minimum cash covenant within 60 days unless it raises additional capital. The company's burn rate of $35M/quarter has accelerated due to failed product launches.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.7% | RGTI: -1.5% | QBTS: +0.4% | QUBT: +0.2% | QNT: +0.9% | IBM: +0.2% | HON: +0.1%", "category": "earnings_miss"},

    # --- TECHNICAL SETBACKS (company variations) ---
    {"scenario": "Rigetti's 84-qubit Ankaa-3 processor shows unexpected correlated errors that make error correction impossible. The errors appear to be a fundamental limitation of their multi-chip architecture.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: +3.2% | QBTS: +0.5% | QUBT: +0.3% | QNT: +1.0% | IBM: +0.3% | HON: +0.1%", "category": "technical_setback"},
    {"scenario": "QNT's H2 processor suffers a systematic ion loss problem that reduces effective qubit count from 56 to 32 during long computations. The fix requires a complete redesign of the ion trap.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.4% | RGTI: +0.7% | QBTS: +0.4% | QUBT: +0.2% | QNT: +4.8% | IBM: +0.2% | HON: +0.3%", "category": "technical_setback"},
    {"scenario": "IonQ's quantum networking prototype fails to achieve entanglement distribution beyond 10 meters, far short of the 100km needed for practical quantum networks. The photon loss rate is 100x higher than theoretical predictions.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.5% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.2% | IBM: +0.2% | HON: +0.2%", "category": "technical_setback"},
    {"scenario": "A reproducibility study finds that D-Wave's claimed quantum speedup on optimization problems cannot be replicated by independent researchers. The original benchmarks appear to have used a biased comparison methodology.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.3% | QBTS: +5.8% | QUBT: +0.2% | QNT: +0.7% | IBM: +0.1% | HON: +0.1%", "category": "technical_setback"},
    {"scenario": "QUBT's neutral atom quantum processor demonstrates only 95% single-qubit gate fidelity, far below the 99.5%+ needed for error correction. The company's roadmap assumed 99.9% by this date.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: +0.2% | QBTS: +0.4% | QUBT: +5.2% | QNT: +0.5% | IBM: +0.1% | HON: -0.1%", "category": "technical_setback"},
    {"scenario": "A Science paper proves that the surface code (used by IBM and Google) requires 10x more physical qubits than previously estimated to achieve fault tolerance. This pushes IBM's fault-tolerant timeline back by 5+ years.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +1.8% | QBTS: +0.6% | QUBT: +0.4% | QNT: +1.5% | IBM: +0.5% | HON: +0.2%", "category": "technical_setback"},
    {"scenario": "IonQ reveals that their barium qubit program has been abandoned after 2 years of development. The barium approach showed promise in the lab but proved impossible to manufacture reliably at scale.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.2% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.0% | IBM: +0.2% | HON: +0.2%", "category": "technical_setback"},
    {"scenario": "Rigetti's quantum error correction demonstration fails publicly at a major conference. The logical qubit shows no improvement over physical qubits, contradicting the company's published results.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +2.8% | QBTS: +0.4% | QUBT: +0.2% | QNT: +0.8% | IBM: +0.3% | HON: +0.1%", "category": "technical_setback"},
    {"scenario": "A fundamental physics paper shows that quantum decoherence in solid-state systems (superconducting, neutral atom) has a hard floor that cannot be reduced below current levels regardless of engineering improvements.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.0% | RGTI: +2.1% | QBTS: +0.8% | QUBT: +1.5% | QNT: +1.2% | IBM: +0.4% | HON: +0.2%", "category": "technical_setback"},
    {"scenario": "QNT admits that their quantum volume claims were measured using a cherry-picked subset of qubits and do not represent whole-processor performance. Actual QV is 4x lower than published.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.8% | RGTI: +0.9% | QBTS: +0.5% | QUBT: +0.3% | QNT: +6.2% | IBM: +0.2% | HON: +0.4%", "category": "technical_setback"},

    # --- COMPETITIVE DISPLACEMENT ---
    {"scenario": "Google demonstrates a 500-qubit superconducting processor that solves a real pharmaceutical optimization problem 1000x faster than classical computers. This is the first undisputed quantum advantage on a commercial problem.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +2.8% | QBTS: +1.0% | QUBT: +0.5% | QNT: +1.8% | IBM: +0.5% | HON: +0.2%", "category": "competitive_displacement"},
    {"scenario": "Microsoft's Azure Quantum announces it will exclusively use its own topological qubits starting 2027, removing IonQ, Rigetti, and Quantinuum from the Azure Quantum marketplace.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.8% | RGTI: +1.5% | QBTS: +0.7% | QUBT: +0.4% | QNT: +3.2% | IBM: +0.3% | HON: +0.2%", "category": "competitive_displacement"},
    {"scenario": "A well-funded Chinese quantum startup demonstrates trapped-ion performance matching IonQ at 1/5th the cost. They announce plans to offer cloud access globally, directly competing with IonQ and QNT.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.1% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +2.5% | IBM: +0.2% | HON: +0.2%", "category": "competitive_displacement"},
    {"scenario": "IBM announces that its quantum computing division will offer free unlimited access to all IBM Cloud enterprise customers. This eliminates the paid quantum computing market that IonQ and Rigetti depend on.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.8% | RGTI: +1.2% | QBTS: +0.6% | QUBT: +0.4% | QNT: +2.1% | IBM: +0.5% | HON: +0.2%", "category": "competitive_displacement"},
    {"scenario": "Nvidia releases a quantum circuit simulator that runs on consumer GPUs and matches the performance of all current quantum computers up to 60 qubits. The software is free and open-source.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.4% | RGTI: +1.6% | QBTS: +1.0% | QUBT: +0.6% | QNT: +2.8% | IBM: +0.4% | HON: +0.2%", "category": "competitive_displacement"},
    {"scenario": "AWS announces it is building its own quantum hardware lab and will phase out third-party quantum processors from Braket within 2 years. IonQ loses its primary distribution channel.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.5% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.2% | IBM: +0.2% | HON: +0.2%", "category": "competitive_displacement"},
    {"scenario": "A classical computing breakthrough using tensor network methods achieves the same results as 100-qubit quantum computers for all currently demonstrated use cases. The paper is published in Nature and widely covered.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.0% | RGTI: +1.4% | QBTS: +0.9% | QUBT: +0.5% | QNT: +2.3% | IBM: +0.3% | HON: +0.2%", "category": "competitive_displacement"},
    {"scenario": "QNT wins a $300M exclusive contract with the UK Ministry of Defence, beating IonQ which had been the incumbent provider. The contract includes a 5-year exclusivity clause for all UK government quantum work.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.2% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +2.8% | IBM: +0.2% | HON: +0.3%", "category": "competitive_displacement"},
    {"scenario": "IonQ's former CTO, now at Google, publishes a paper showing that Google's superconducting approach has surpassed trapped-ion systems on every meaningful benchmark. He calls trapped-ion 'a dead end at scale.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.5% | RGTI: +0.9% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.8% | IBM: +0.3% | HON: +0.2%", "category": "competitive_displacement"},
    {"scenario": "Rigetti loses its position as the only superconducting quantum computer on AWS Braket. IBM's Eagle processor is added with 3x more qubits and better error rates, at the same price point.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: +3.2% | QBTS: +0.5% | QUBT: +0.2% | QNT: +1.0% | IBM: +0.5% | HON: +0.1%", "category": "competitive_displacement"},

    # --- CAPITAL MARKETS / DILUTION ---
    {"scenario": "Rigetti announces a $200M at-the-market (ATM) equity offering, representing 30% dilution at current prices. The stock drops 15% immediately as the market absorbs the supply.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.0% | RGTI: +5.5% | QBTS: +0.8% | QUBT: +0.4% | QNT: +1.3% | IBM: +0.2% | HON: +0.1%", "category": "capital_markets"},
    {"scenario": "QNT's IPO lockup expires and Honeywell sells 20% of its remaining stake in a block trade at a 12% discount to market. The selling creates a $400M supply overhang.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +0.8% | QBTS: +0.5% | QUBT: +0.3% | QNT: +8.2% | IBM: +0.2% | HON: +0.5%", "category": "capital_markets"},
    {"scenario": "IonQ's convertible bonds are approaching their put date and the company lacks cash to redeem them. A forced conversion at current prices would dilute shareholders by 25%.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -2.1% | RGTI: -0.8% | QBTS: -0.4% | QUBT: -0.3% | QNT: -1.5% | IBM: +0.1% | HON: -0.1%", "category": "capital_markets"},
    {"scenario": "D-Wave announces a 1-for-10 reverse stock split to maintain NASDAQ compliance. The company's market cap has fallen to $80M and institutional investors are forced sellers due to minimum market cap requirements.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.4% | RGTI: +0.2% | QBTS: -4.5% | QUBT: +0.1% | QNT: +0.5% | IBM: +0.1% | HON: -0.1%", "category": "capital_markets"},
    {"scenario": "QUBT's largest shareholder (a hedge fund with 15% ownership) files a 13D indicating intent to liquidate their entire position over the next 30 days. The daily selling will represent 3x normal volume.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: +0.2% | QBTS: +0.3% | QUBT: +3.5% | QNT: +0.4% | IBM: +0.1% | HON: -0.1%", "category": "capital_markets"},
    {"scenario": "IonQ's warrants are about to expire and warrant holders are exercising en masse, creating $150M of new share supply hitting the market over 2 weeks.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +6.2% | RGTI: +2.1% | QBTS: +1.2% | QUBT: +0.8% | QNT: +4.5% | IBM: +0.3% | HON: +0.2%", "category": "capital_markets"},
    {"scenario": "Rigetti's debt-to-equity ratio exceeds 3:1 after a new loan facility. Credit rating agencies downgrade the company to CCC, making future fundraising extremely expensive.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.7% | RGTI: -1.8% | QBTS: +0.3% | QUBT: +0.2% | QNT: +0.8% | IBM: +0.2% | HON: +0.1%", "category": "capital_markets"},
    {"scenario": "Multiple quantum computing insiders (CEOs, CTOs) sell shares simultaneously in what appears to be a coordinated exit. Combined insider selling across IONQ, RGTI, and QBTS totals $50M in a single week.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.8% | RGTI: +3.5% | QBTS: +2.8% | QUBT: +1.5% | QNT: +5.2% | IBM: +0.4% | HON: +0.3%", "category": "capital_markets"},

    # --- EXECUTIVE DEPARTURES ---
    {"scenario": "IonQ's entire quantum error correction team (6 PhDs) resigns to join Quantinuum, citing 'better hardware to work with.' IonQ's error correction roadmap is now critically understaffed.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.5% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.2% | IBM: +0.2% | HON: +0.2%", "category": "executive_departure"},
    {"scenario": "Rigetti's CEO and co-founder Chad Rigetti is forced out by the board after a series of missed milestones. The company names an interim CEO from outside the quantum industry.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: +2.5% | QBTS: +0.4% | QUBT: +0.2% | QNT: +1.0% | IBM: +0.2% | HON: +0.1%", "category": "executive_departure"},
    {"scenario": "QNT's head of commercial operations and the entire enterprise sales team (15 people) leave to join a well-funded quantum startup. QNT's sales pipeline is now unmanaged.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +0.6% | QBTS: +0.3% | QUBT: +0.2% | QNT: +4.5% | IBM: +0.2% | HON: +0.3%", "category": "executive_departure"},
    {"scenario": "D-Wave's chief scientist publishes a blog post saying quantum annealing 'may never achieve practical advantage' and announces he is leaving to pursue gate-model research at a university.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.3% | QBTS: +3.8% | QUBT: +0.2% | QNT: +0.6% | IBM: +0.1% | HON: +0.1%", "category": "executive_departure"},
    {"scenario": "IonQ's VP of Engineering and 8 hardware engineers leave simultaneously to start a competing trapped-ion company with $500M in funding from SoftBank.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.8% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +1.8% | IBM: +0.2% | HON: +0.2%", "category": "executive_departure"},
    {"scenario": "QUBT's auditor resigns citing 'inability to obtain sufficient audit evidence.' The company cannot file its 10-K on time and faces potential delisting.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: +0.2% | QBTS: +0.3% | QUBT: +1.8% | QNT: +0.4% | IBM: +0.1% | HON: -0.1%", "category": "executive_departure"},

    # --- ANALYST / SHORT REPORTS ---
    {"scenario": "Kerrisdale Capital publishes a 50-page short report on Rigetti titled 'Superconducting Pipe Dreams.' The report argues Rigetti's technology is 5 years behind IBM and Google with no path to catch up.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.0% | RGTI: +4.2% | QBTS: +0.7% | QUBT: +0.4% | QNT: +1.3% | IBM: +0.3% | HON: +0.1%", "category": "analyst_negative"},
    {"scenario": "UBS downgrades the entire quantum computing sector to Sell, arguing that 'the gap between quantum hype and quantum reality has never been wider.' Price targets cut by 60-80% across all names.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +5.5% | RGTI: +3.8% | QBTS: +2.5% | QUBT: +1.8% | QNT: +6.2% | IBM: +0.5% | HON: +0.3%", "category": "analyst_negative"},
    {"scenario": "A former quantum computing researcher publishes a viral Twitter thread explaining why 'every quantum computing company is lying about their qubit counts' and how marketing metrics differ from computational reality.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.8% | RGTI: +2.5% | QBTS: +1.8% | QUBT: +1.2% | QNT: +4.5% | IBM: +0.4% | HON: +0.3%", "category": "analyst_negative"},
    {"scenario": "Citron Research targets QNT with a short report claiming the company's IPO valuation was inflated by 'related-party transactions with Honeywell that don't represent arm's-length commercial demand.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +0.8% | QBTS: +0.5% | QUBT: +0.3% | QNT: +9.2% | IBM: +0.2% | HON: +0.5%", "category": "analyst_negative"},
    {"scenario": "Barclays initiates coverage of D-Wave with a Sell rating and $0.50 price target (current: $3.50). The analyst argues quantum annealing is 'commercially irrelevant' and D-Wave will run out of cash by 2026.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +0.4% | QBTS: +3.2% | QUBT: +0.3% | QNT: +0.8% | IBM: +0.2% | HON: +0.1%", "category": "analyst_negative"},
    {"scenario": "A Bloomberg opinion piece titled 'Quantum Computing: The Next Dot-Com Bubble' draws parallels between current quantum valuations and 1999 internet stocks. The article is shared 50,000 times.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +7.2% | RGTI: +5.1% | QBTS: +3.5% | QUBT: +2.8% | QNT: +8.1% | IBM: +0.6% | HON: +0.4%", "category": "analyst_negative"},

    # --- SECTOR SELLOFFS ---
    {"scenario": "The Federal Reserve unexpectedly raises rates by 100 basis points. All speculative growth stocks crash. Quantum computing stocks fall 25-40% in a single session with no sector-specific catalyst.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.5% | RGTI: +1.8% | QBTS: +1.2% | QUBT: +0.8% | QNT: +2.9% | IBM: +0.3% | HON: +0.2%", "category": "sector_selloff"},
    {"scenario": "A 'quantum winter' narrative takes hold after Gartner moves quantum computing from 'Peak of Inflated Expectations' to 'Trough of Disillusionment' in their hype cycle. Institutional investors reduce quantum allocations.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -3.5% | RGTI: -4.2% | QBTS: -2.8% | QUBT: -3.1% | QNT: -3.8% | IBM: -0.5% | HON: -0.3%", "category": "sector_selloff"},
    {"scenario": "Three quantum computing companies announce layoffs in the same week (IonQ 20%, Rigetti 25%, D-Wave 15%). The coordinated cuts signal that the entire sector is retrenching.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -5.2% | RGTI: -6.8% | QBTS: -4.5% | QUBT: -3.8% | QNT: -5.5% | IBM: -0.8% | HON: -0.4%", "category": "sector_selloff"},
    {"scenario": "The CHIPS Act quantum computing funding is vetoed by the President, eliminating $3B in expected government support for the quantum industry. Multiple companies had this funding in their financial projections.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.2% | RGTI: +2.1% | QBTS: +1.5% | QUBT: +1.0% | QNT: +3.8% | IBM: +0.4% | HON: +0.2%", "category": "sector_selloff"},
    {"scenario": "A major quantum computing conference is cancelled due to lack of corporate sponsorship. Organizers say 'the industry cannot support the event financially.' This signals broader funding contraction.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -1.8% | RGTI: -2.5% | QBTS: -1.5% | QUBT: -1.2% | QNT: -2.0% | IBM: -0.3% | HON: -0.2%", "category": "sector_selloff"},

    # --- REGULATORY / LEGAL ---
    {"scenario": "The SEC charges IonQ with securities fraud, alleging the company materially overstated its quantum computing capabilities in investor presentations. Trading is halted pending investigation.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.8% | RGTI: +1.5% | QBTS: +0.8% | QUBT: +0.5% | QNT: +3.2% | IBM: +0.3% | HON: +0.2%", "category": "regulatory_legal"},
    {"scenario": "Quantinuum faces a patent infringement lawsuit from IonQ over ion-trap cooling techniques. If IonQ wins, QNT would need to redesign its core hardware or pay substantial licensing fees.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +0.7% | QBTS: +0.4% | QUBT: +0.2% | QNT: +5.8% | IBM: +0.2% | HON: +0.4%", "category": "regulatory_legal"},
    {"scenario": "New US export controls classify all quantum computers as dual-use technology requiring individual export licenses. Processing times are 6-12 months, effectively freezing international sales for all US quantum companies.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.2% | RGTI: +1.5% | QBTS: +0.9% | QUBT: +0.5% | QNT: +2.5% | IBM: +0.3% | HON: +0.2%", "category": "regulatory_legal"},
    {"scenario": "A class-action lawsuit is filed against Rigetti alleging that management made materially misleading statements about the timeline to fault-tolerant quantum computing in their SPAC merger documents.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: +3.5% | QBTS: +0.5% | QUBT: +0.3% | QNT: +1.0% | IBM: +0.2% | HON: +0.1%", "category": "regulatory_legal"},
    {"scenario": "The DOJ opens an antitrust investigation into whether IonQ and Quantinuum engaged in market allocation by agreeing not to compete for certain government contracts.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.0% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +3.5% | IBM: +0.2% | HON: +0.3%", "category": "regulatory_legal"},

    # --- PRICED-IN / OVEREXTENDED (many variations) ---
    {"scenario": "IonQ announces a routine quarterly update with no new information. All metrics are in-line with previously disclosed guidance. No surprises.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +52.3% | RGTI: +28.5% | QBTS: +18.2% | QUBT: +14.5% | QNT: +42.8% | IBM: +4.2% | HON: +2.8%", "category": "priced_in"},
    {"scenario": "Rigetti publishes a blog post about their participation in a DOE quantum computing workshop. No contracts or partnerships announced, just attendance.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +22.1% | RGTI: +58.5% | QBTS: +15.3% | QUBT: +11.8% | QNT: +25.4% | IBM: +3.1% | HON: +1.8%", "category": "priced_in"},
    {"scenario": "D-Wave issues a press release about a $1M pilot project with a mid-size logistics company. The deal is immaterial relative to the company's $2B market cap.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +18.5% | RGTI: +14.2% | QBTS: +62.8% | QUBT: +10.5% | QNT: +20.1% | IBM: +2.5% | HON: +1.5%", "category": "priced_in"},
    {"scenario": "QUBT tweets about presenting at a small investor conference. No new product announcements or customer wins.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +15.8% | RGTI: +11.2% | QBTS: +9.5% | QUBT: +45.2% | QNT: +17.8% | IBM: +2.0% | HON: +1.2%", "category": "priced_in"},
    {"scenario": "QNT announces it has been selected for a $500K feasibility study by a European bank. The study may or may not lead to a larger contract.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +25.4% | RGTI: +18.8% | QBTS: +12.5% | QUBT: +9.2% | QNT: +65.2% | IBM: +3.5% | HON: +4.2%", "category": "priced_in"},
    {"scenario": "A positive but generic article in Wired titled 'Quantum Computing Is Finally Getting Real' is published. It contains no new information, just interviews with industry executives making optimistic predictions.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +35.8% | RGTI: +30.2% | QBTS: +22.5% | QUBT: +18.8% | QNT: +38.5% | IBM: +4.8% | HON: +3.1%", "category": "priced_in"},
    {"scenario": "IonQ's CEO gives a keynote at CES reiterating the company's existing roadmap. All milestones and timelines were previously disclosed in the last earnings call.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +48.5% | RGTI: +25.2% | QBTS: +16.8% | QUBT: +12.1% | QNT: +40.2% | IBM: +3.8% | HON: +2.5%", "category": "priced_in"},
    {"scenario": "Rigetti announces they have added quantum computing tutorials to their documentation. No hardware improvements or customer announcements.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +20.5% | RGTI: +45.8% | QBTS: +14.5% | QUBT: +10.2% | QNT: +22.8% | IBM: +2.8% | HON: +1.5%", "category": "priced_in"},
    {"scenario": "A Reddit post about quantum computing stocks goes viral, driving retail buying. No fundamental news. The stocks are up 30-60% in a week purely on momentum and social media hype.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +42.5% | RGTI: +55.8% | QBTS: +38.2% | QUBT: +32.5% | QNT: +45.8% | IBM: +5.2% | HON: +3.5%", "category": "priced_in"},
    {"scenario": "IonQ announces a minor software update to their cloud platform that improves job scheduling efficiency by 10%. No hardware improvements.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +55.2% | RGTI: +30.5% | QBTS: +20.8% | QUBT: +15.5% | QNT: +48.2% | IBM: +4.5% | HON: +2.8%", "category": "priced_in"},
    {"scenario": "D-Wave publishes a customer testimonial from an existing client praising their quantum annealing service. The client has been using D-Wave for 2 years; this is not a new customer.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +15.2% | RGTI: +12.8% | QBTS: +52.5% | QUBT: +8.5% | QNT: +18.2% | IBM: +2.2% | HON: +1.2%", "category": "priced_in"},
    {"scenario": "QNT is featured in a 'Top 10 Quantum Computing Companies' listicle by a tech blog. No new information, just aggregation of public knowledge.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +28.5% | RGTI: +20.2% | QBTS: +15.8% | QUBT: +11.5% | QNT: +58.5% | IBM: +3.2% | HON: +3.8%", "category": "priced_in"},
    {"scenario": "An analyst reiterates their Hold rating on QUBT with an unchanged $2 price target. The note says 'we await further evidence of commercial traction.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +12.5% | RGTI: +9.8% | QBTS: +8.2% | QUBT: +38.5% | QNT: +14.8% | IBM: +1.8% | HON: +1.0%", "category": "priced_in"},
    {"scenario": "Rigetti announces they will present at an upcoming quantum computing conference in 3 weeks. No details on what they will present.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +18.2% | RGTI: +48.5% | QBTS: +12.5% | QUBT: +9.5% | QNT: +20.5% | IBM: +2.5% | HON: +1.5%", "category": "priced_in"},
    {"scenario": "IonQ files a routine patent application for a quantum computing technique. The patent is incremental and does not represent a breakthrough.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +62.5% | RGTI: +35.2% | QBTS: +22.8% | QUBT: +18.2% | QNT: +55.8% | IBM: +5.5% | HON: +3.2%", "category": "priced_in"},

    # --- ADDITIONAL MIXED BEARISH ---
    {"scenario": "IonQ announces a 20% workforce reduction to 'align costs with revenue trajectory.' The company says it will focus on fewer, larger customers rather than broad market penetration.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.8% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +1.5% | IBM: +0.2% | HON: +0.2%", "category": "layoffs"},
    {"scenario": "Rigetti announces it is shutting down its UK operations and consolidating all work in the US. The move eliminates 50 positions and signals the company is in cost-cutting mode.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +2.2% | QBTS: +0.4% | QUBT: +0.2% | QNT: +0.8% | IBM: +0.2% | HON: +0.1%", "category": "layoffs"},
    {"scenario": "D-Wave cuts 30% of its workforce and pivots entirely to software, abandoning quantum hardware development. The company will resell access to other companies' quantum processors.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.3% | QBTS: +2.8% | QUBT: +0.2% | QNT: +0.6% | IBM: +0.1% | HON: +0.1%", "category": "layoffs"},
    {"scenario": "QNT announces a hiring freeze and delays the opening of its new fabrication facility by 18 months. The company cites 'uncertain market conditions' for the delay.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +0.7% | QBTS: +0.4% | QUBT: +0.2% | QNT: +5.2% | IBM: +0.2% | HON: +0.3%", "category": "layoffs"},
    {"scenario": "A major institutional investor publishes a letter to IonQ's board demanding the CEO be replaced, citing 'consistent failure to meet stated milestones and destruction of shareholder value.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -3.5% | RGTI: -1.2% | QBTS: -0.5% | QUBT: -0.4% | QNT: -2.0% | IBM: +0.1% | HON: -0.1%", "category": "governance"},
    {"scenario": "Rigetti's board approves a poison pill after an activist investor accumulates a 9.9% stake. The activist is pushing for the company to sell itself or liquidate, arguing the technology has no commercial future.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: -2.5% | QBTS: +0.3% | QUBT: +0.2% | QNT: +0.9% | IBM: +0.2% | HON: +0.1%", "category": "governance"},
]

print(f"Total batch 2 scenarios: {len(SCENARIOS)}")


# ============================================================
# Generation (reuses same infrastructure as batch 1)
# ============================================================

_file_lock = asyncio.Lock()

async def append_result(filepath, record):
    async with _file_lock:
        with open(filepath, "a") as f:
            f.write(json.dumps(record) + "\n")

async def create_task(session, user_msg, retries=3):
    for attempt in range(retries):
        try:
            payload = {"message": {"content": user_msg}, "structured_output_schema": V5_SCHEMA, "project_id": PROJECT_ID, "agent_profile": "manus-1.6-max"}
            async with session.post(f"{BASE_URL}/task.create", headers=HEADERS, json=payload) as resp:
                if resp.status == 429:
                    await asyncio.sleep(int(resp.headers.get("Retry-After", 60)))
                    continue
                data = await resp.json()
                if data.get("ok"):
                    await asyncio.sleep(CREATION_DELAY)
                    return data["task_id"]
                if attempt < retries - 1:
                    await asyncio.sleep(60)
        except Exception as e:
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
        except Exception:
            pass
        await asyncio.sleep(POLL_INTERVAL)
    return {"success": False, "error": "Timeout"}

async def process_scenario(session, semaphore, idx, scenario_data, total):
    async with semaphore:
        start_time = time.time()
        scenario = scenario_data["scenario"]
        market_context = scenario_data.get("market_context", "")
        category = scenario_data.get("category", "bearish")
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
            result = {"idx": idx, "success": True, "thinking": thinking, "signal": signal, "validation_issues": issues, "task_id": task_id, "category": category, "scenario": scenario, "market_context": market_context, "time_seconds": elapsed, "timestamp": datetime.now().isoformat()}
            await append_result(OUTPUT_FILE, result)
            print(f"  [{idx+1}/{total}] SUCCESS ({elapsed:.0f}s, think={len(thinking.split())} words)")
            return result
        else:
            result = {"idx": idx, "success": False, "error": response.get("error", "?"), "task_id": task_id, "category": category, "time_seconds": elapsed, "timestamp": datetime.now().isoformat()}
            await append_result(OUTPUT_FILE, result)
            print(f"  [{idx+1}/{total}] FAILED ({elapsed:.0f}s)")
            return result

async def main():
    existing = set()
    if OUTPUT_FILE.exists():
        with open(OUTPUT_FILE) as f:
            for line in f:
                if line.strip():
                    existing.add(json.loads(line).get("idx"))
    pending = [(i, s) for i, s in enumerate(SCENARIOS) if i not in existing]
    print(f"={'='*60}")
    print(f"V5 BEARISH BATCH 2")
    print(f"={'='*60}")
    print(f"Total: {len(SCENARIOS)} | Done: {len(existing)} | Pending: {len(pending)}")
    print(f"Project: Training Tasks ({PROJECT_ID})")
    print(f"Output: {OUTPUT_FILE}")
    print(f"={'='*60}\n")
    if not pending:
        print("All done!")
        return
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    connector = aiohttp.TCPConnector(limit=MAX_CONCURRENT + 5)
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [process_scenario(session, semaphore, idx, s, len(pending)) for idx, s in pending]
        results = await asyncio.gather(*tasks, return_exceptions=True)
    successes = sum(1 for r in results if isinstance(r, dict) and r.get("success"))
    print(f"\n{'='*60}\nCOMPLETE: {successes}/{len(pending)} successful\n{'='*60}")

if __name__ == "__main__":
    asyncio.run(main())
