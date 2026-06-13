"""
Generate robustness training examples:
1. Drawdown behavior (40) - stocks already deeply negative
2. Sideways/choppy market (20) - noise, small scores
3. Conflicting signals (20) - mixed news, moderate scores

Usage:
    python scripts/generate_v5_robustness.py
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

from scripts.generate_v5_thinking import (
    V5_SYSTEM_PROMPT, V5_SCHEMA, postprocess_signal, validate_signal,
    API_KEY, BASE_URL, HEADERS, PROJECT_ID, MAX_CONCURRENT, CREATION_DELAY, POLL_INTERVAL, MAX_POLL_TIME
)

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
OUTPUT_FILE = DATA_TRAINING / "quantum_alpha_train_v5_robustness.jsonl"

# ============================================================
# DRAWDOWN SCENARIOS (40)
# Stocks already deeply negative. Model must learn behavior in pain.
# ============================================================

DRAWDOWN_SCENARIOS = [
    # Bad news during drawdown (continuation - stay bearish)
    {"scenario": "IonQ reports that a key government contract has been delayed by 6 months due to budget sequestration. The company's cash runway is now 9 months without additional funding.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -18.5% | RGTI: -12.3% | QBTS: -8.5% | QUBT: -10.2% | QNT: -15.8% | IBM: -2.1% | HON: -1.5%", "category": "drawdown_continuation"},
    {"scenario": "Rigetti announces another round of layoffs (15% of remaining staff) as the company struggles to reduce burn rate. This is the third round of cuts in 6 months.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -22.1% | RGTI: -28.5% | QBTS: -15.2% | QUBT: -12.8% | QNT: -20.3% | IBM: -3.5% | HON: -2.2%", "category": "drawdown_continuation"},
    {"scenario": "D-Wave's auditor issues a going-concern warning in their quarterly filing. The company has $15M cash with $12M quarterly burn.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -15.2% | RGTI: -18.8% | QBTS: -32.5% | QUBT: -14.5% | QNT: -16.2% | IBM: -2.8% | HON: -1.8%", "category": "drawdown_continuation"},
    {"scenario": "QUBT receives a NASDAQ delisting warning as its stock price has been below $1 for 30 consecutive days. The company has 180 days to regain compliance.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -10.5% | RGTI: -8.2% | QBTS: -7.5% | QUBT: -25.8% | QNT: -11.2% | IBM: -1.5% | HON: -0.8%", "category": "drawdown_continuation"},
    {"scenario": "A second major analyst downgrades IonQ in the same week, citing 'no near-term catalysts and deteriorating competitive position.' The stock is already down 40% from its high.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -25.2% | RGTI: -15.5% | QBTS: -10.8% | QUBT: -8.5% | QNT: -22.1% | IBM: -2.5% | HON: -1.8%", "category": "drawdown_continuation"},
    {"scenario": "QNT's largest institutional holder files a 13F showing they sold 80% of their position last quarter. The selling has been ongoing for 3 months.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -12.8% | RGTI: -10.5% | QBTS: -8.2% | QUBT: -6.5% | QNT: -30.2% | IBM: -1.8% | HON: -2.5%", "category": "drawdown_continuation"},
    {"scenario": "Rigetti's latest processor benchmark shows no improvement over the version released 18 months ago. The company appears to have hit a technical wall.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -8.5% | RGTI: -35.2% | QBTS: -6.8% | QUBT: -5.5% | QNT: -9.2% | IBM: -1.2% | HON: -0.8%", "category": "drawdown_continuation"},
    {"scenario": "The quantum computing sector enters its 4th consecutive month of declines. Total sector market cap has fallen 55% from peak. A prominent VC says 'quantum winter is here.'", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -8.2% | RGTI: -9.5% | QBTS: -7.8% | QUBT: -6.2% | QNT: -8.8% | IBM: -1.5% | HON: -1.0%", "category": "drawdown_continuation"},
    {"scenario": "IonQ's co-founder sells $5M of personal stock in an open-market transaction. While small relative to his holdings, the timing during a 45% drawdown sends a negative signal.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -15.8% | RGTI: -11.2% | QBTS: -8.5% | QUBT: -7.2% | QNT: -14.5% | IBM: -2.0% | HON: -1.2%", "category": "drawdown_continuation"},
    {"scenario": "A Bloomberg article titled 'Quantum Computing Stocks: Is the Bottom In?' concludes 'probably not' based on deteriorating fundamentals and continued cash burn across the sector.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -20.5% | RGTI: -22.8% | QBTS: -18.2% | QUBT: -15.5% | QNT: -21.2% | IBM: -3.2% | HON: -2.0%", "category": "drawdown_continuation"},

    # Good news during drawdown (potential recovery - cautiously bullish)
    {"scenario": "IonQ wins a $50M contract with the Department of Defense despite the stock being down 45% from highs. This is the largest single contract in company history and validates their technology.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -18.5% | RGTI: -12.2% | QBTS: -9.5% | QUBT: -7.8% | QNT: -15.2% | IBM: -2.5% | HON: -1.5%", "category": "drawdown_recovery"},
    {"scenario": "Rigetti demonstrates a genuine quantum advantage on a drug discovery problem, published in Nature. The stock is down 60% from its high but this is a fundamental breakthrough.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -22.5% | RGTI: -35.8% | QBTS: -15.2% | QUBT: -12.5% | QNT: -20.8% | IBM: -3.8% | HON: -2.2%", "category": "drawdown_recovery"},
    {"scenario": "D-Wave secures a $100M strategic investment from a major tech company at a 50% premium to current market price, validating that the technology has value despite the stock being down 70%.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -12.8% | RGTI: -10.5% | QBTS: -28.5% | QUBT: -8.2% | QNT: -13.5% | IBM: -2.0% | HON: -1.2%", "category": "drawdown_recovery"},
    {"scenario": "QNT reports Q3 revenue of $55M, beating estimates by 40% and showing accelerating growth. The stock is down 35% but fundamentals are improving rapidly.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -15.2% | RGTI: -12.8% | QBTS: -10.5% | QUBT: -8.5% | QNT: -25.5% | IBM: -2.2% | HON: -1.8%", "category": "drawdown_recovery"},
    {"scenario": "Congress passes a $5B quantum computing funding bill with bipartisan support. The sector has been in a 3-month drawdown but this provides a fundamental catalyst for the entire industry.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -10.5% | RGTI: -12.2% | QBTS: -8.8% | QUBT: -7.5% | QNT: -11.5% | IBM: -1.8% | HON: -1.2%", "category": "drawdown_recovery"},
    {"scenario": "IonQ achieves a genuine error correction milestone (logical qubit lifetime exceeding physical qubit lifetime by 10x) during a period where the stock is down 50%. This is the breakthrough investors have been waiting for.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -25.8% | RGTI: -18.5% | QBTS: -12.2% | QUBT: -10.5% | QNT: -22.5% | IBM: -3.5% | HON: -2.0%", "category": "drawdown_recovery"},
    {"scenario": "A major hedge fund discloses a new 10% position in Rigetti, calling it 'the most undervalued quantum computing company' at current prices. The fund has a strong track record in tech investing.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -8.5% | RGTI: -32.5% | QBTS: -6.8% | QUBT: -5.5% | QNT: -9.2% | IBM: -1.2% | HON: -0.8%", "category": "drawdown_recovery"},
    {"scenario": "Quantinuum announces it has achieved cash-flow breakeven for the first time, 6 months ahead of schedule. The stock is down 30% but the business model is now proven.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -12.5% | RGTI: -10.2% | QBTS: -8.5% | QUBT: -6.8% | QNT: -22.8% | IBM: -1.8% | HON: -2.5%", "category": "drawdown_recovery"},

    # Noise during drawdown (should be zero or near-zero)
    {"scenario": "IonQ publishes a routine blog post about their intern program. No technical or business announcements.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -15.2% | RGTI: -12.5% | QBTS: -9.8% | QUBT: -8.2% | QNT: -14.5% | IBM: -2.2% | HON: -1.5%", "category": "drawdown_noise"},
    {"scenario": "A generic article discusses 'the future of quantum computing' with no new information, company-specific news, or timeline updates. Pure filler content.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -20.5% | RGTI: -18.2% | QBTS: -14.5% | QUBT: -12.8% | QNT: -19.8% | IBM: -3.0% | HON: -2.0%", "category": "drawdown_noise"},
    {"scenario": "Rigetti announces they will present at a routine investor conference next month. No details on content.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -8.5% | RGTI: -22.8% | QBTS: -6.5% | QUBT: -5.2% | QNT: -9.5% | IBM: -1.5% | HON: -0.8%", "category": "drawdown_noise"},
    {"scenario": "An incremental arXiv paper on quantum error correction theory is published by academic researchers with no company affiliation. The paper offers no new experimental results.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -12.2% | RGTI: -10.8% | QBTS: -8.5% | QUBT: -7.2% | QNT: -11.8% | IBM: -1.8% | HON: -1.2%", "category": "drawdown_noise"},
    {"scenario": "D-Wave tweets about quantum computing awareness month. No product or business announcements.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -10.5% | RGTI: -8.8% | QBTS: -15.5% | QUBT: -6.5% | QNT: -11.2% | IBM: -1.5% | HON: -1.0%", "category": "drawdown_noise"},
    {"scenario": "A podcast episode features the CEO of QUBT discussing the 'long-term vision' for quantum computing. No new milestones, contracts, or technical achievements mentioned.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -8.2% | RGTI: -7.5% | QBTS: -6.2% | QUBT: -18.5% | QNT: -9.0% | IBM: -1.2% | HON: -0.8%", "category": "drawdown_noise"},

    # Macro-driven drawdown (sector-wide, not quantum-specific)
    {"scenario": "The S&P 500 drops 5% in a single day on recession fears. Quantum stocks fall 15-25% purely on risk-off sentiment. No quantum-specific news whatsoever.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -22.5% | RGTI: -25.8% | QBTS: -18.2% | QUBT: -15.5% | QNT: -23.2% | IBM: -5.5% | HON: -4.2%", "category": "drawdown_macro"},
    {"scenario": "10-year Treasury yields spike to 6%, crushing all growth stock valuations. Quantum computing stocks fall 20% in a week with no sector-specific catalyst.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -18.8% | RGTI: -20.5% | QBTS: -15.2% | QUBT: -12.8% | QNT: -19.5% | IBM: -4.2% | HON: -3.5%", "category": "drawdown_macro"},
    {"scenario": "A banking crisis causes a broad market selloff. Quantum stocks are caught in the liquidation as hedge funds sell everything to raise cash. No quantum news.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -30.2% | RGTI: -32.5% | QBTS: -25.8% | QUBT: -22.5% | QNT: -28.8% | IBM: -8.5% | HON: -6.2%", "category": "drawdown_macro"},
    {"scenario": "Geopolitical tensions escalate and markets enter risk-off mode. All speculative technology stocks decline. A routine IonQ partnership announcement is ignored by the market.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -15.5% | RGTI: -18.2% | QBTS: -12.8% | QUBT: -10.5% | QNT: -16.5% | IBM: -3.8% | HON: -2.8%", "category": "drawdown_macro"},
    {"scenario": "The Nasdaq drops 10% in a week. Quantum stocks fall 25-35% as high-beta names get hit hardest. IonQ announces a minor product update that would normally be mildly positive.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -28.5% | RGTI: -30.2% | QBTS: -22.5% | QUBT: -18.8% | QNT: -27.5% | IBM: -6.5% | HON: -4.8%", "category": "drawdown_macro"},
    {"scenario": "Inflation data comes in hot, markets sell off broadly. Quantum stocks are down 12% this week on pure macro. A generic positive quantum computing article is published.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -12.2% | RGTI: -14.5% | QBTS: -10.8% | QUBT: -9.2% | QNT: -13.5% | IBM: -3.2% | HON: -2.5%", "category": "drawdown_macro"},
]

# ============================================================
# SIDEWAYS/CHOPPY MARKET (20)
# Stocks going nowhere. Most news is noise. Scores should be small.
# ============================================================

SIDEWAYS_SCENARIOS = [
    {"scenario": "IonQ presents at a quantum computing conference with no new announcements. Reiterates existing roadmap. Stock has been range-bound for 2 months.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: -1.2% | QBTS: +0.5% | QUBT: -0.3% | QNT: +1.1% | IBM: +0.2% | HON: -0.1%", "category": "sideways"},
    {"scenario": "Rigetti publishes a technical blog post about compiler improvements that reduce circuit depth by 5%. Incremental progress, not a breakthrough.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.5% | RGTI: +1.5% | QBTS: -0.8% | QUBT: +0.2% | QNT: -0.3% | IBM: +0.1% | HON: +0.2%", "category": "sideways"},
    {"scenario": "D-Wave announces a partnership with a small consulting firm to explore quantum optimization use cases. No revenue commitment, just a feasibility study.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: -0.5% | QBTS: +1.2% | QUBT: -0.4% | QNT: +0.5% | IBM: -0.1% | HON: +0.1%", "category": "sideways"},
    {"scenario": "An academic paper discusses theoretical improvements to quantum error correction codes. No experimental validation, no company involvement.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -1.2% | RGTI: +0.8% | QBTS: -0.3% | QUBT: +0.5% | QNT: -0.8% | IBM: +0.2% | HON: -0.2%", "category": "sideways"},
    {"scenario": "QUBT announces they have hired a new head of business development. No product or technology updates.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: -0.3% | QBTS: +0.2% | QUBT: +2.1% | QNT: +0.4% | IBM: +0.1% | HON: -0.1%", "category": "sideways"},
    {"scenario": "A listicle article ranks quantum computing stocks by market cap. Contains no new information or analysis.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.8% | RGTI: +0.5% | QBTS: -0.2% | QUBT: +0.3% | QNT: -0.5% | IBM: +0.1% | HON: +0.1%", "category": "sideways"},
    {"scenario": "IonQ files a routine 8-K for a board member appointment. The new director has a finance background with no quantum expertise.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: -0.8% | QBTS: +0.3% | QUBT: -0.2% | QNT: +0.8% | IBM: +0.1% | HON: +0.2%", "category": "sideways"},
    {"scenario": "Rigetti announces availability of their quantum computing platform in a new AWS region (Asia-Pacific). Incremental distribution expansion.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.3% | RGTI: +0.8% | QBTS: -0.5% | QUBT: +0.1% | QNT: -0.2% | IBM: +0.2% | HON: -0.1%", "category": "sideways"},
    {"scenario": "A quantum computing industry report projects the market will reach $65B by 2040. This is the same projection that has been cited for 3 years.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.2% | RGTI: -0.4% | QBTS: +0.6% | QUBT: -0.1% | QNT: +0.3% | IBM: +0.1% | HON: -0.1%", "category": "sideways"},
    {"scenario": "QNT announces they have opened a new office in Tokyo. No customer announcements or technology updates.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.5% | RGTI: +0.3% | QBTS: -0.2% | QUBT: +0.4% | QNT: +1.5% | IBM: -0.1% | HON: +0.2%", "category": "sideways"},
    {"scenario": "An analyst maintains their Hold rating on IonQ with an unchanged price target. The note contains no new analysis.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.0% | RGTI: -0.6% | QBTS: +0.4% | QUBT: -0.3% | QNT: +0.7% | IBM: +0.1% | HON: +0.1%", "category": "sideways"},
    {"scenario": "D-Wave publishes a case study about an existing customer's quantum optimization results. The customer has been using D-Wave for 2 years; this is not new.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.2% | RGTI: +0.4% | QBTS: +0.8% | QUBT: -0.5% | QNT: -0.1% | IBM: +0.1% | HON: -0.1%", "category": "sideways"},
    {"scenario": "A minor quantum computing startup raises a $10M Series A. The startup is not a direct competitor to any public company and is focused on quantum sensing, not computing.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.4% | RGTI: -0.2% | QBTS: +0.1% | QUBT: +0.3% | QNT: +0.5% | IBM: -0.1% | HON: +0.1%", "category": "sideways"},
    {"scenario": "Rigetti's CEO gives an interview reiterating that they expect to achieve quantum advantage 'within the next 2-3 years.' This is the same timeline they've stated for the past 18 months.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.8% | RGTI: +1.2% | QBTS: -0.3% | QUBT: +0.2% | QNT: -0.5% | IBM: +0.1% | HON: -0.1%", "category": "sideways"},
    {"scenario": "IonQ tweets about World Quantum Day. No product announcements, just a marketing post.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: -0.4% | QBTS: +0.2% | QUBT: -0.1% | QNT: +0.4% | IBM: +0.1% | HON: -0.1%", "category": "sideways"},
    {"scenario": "QUBT announces participation in a quantum computing hackathon. No commercial significance.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.3% | RGTI: +0.5% | QBTS: -0.1% | QUBT: +0.8% | QNT: -0.2% | IBM: +0.1% | HON: -0.1%", "category": "sideways"},
    {"scenario": "A YouTube video about 'investing in quantum computing stocks' goes mildly viral (100K views). Contains no new information, just rehashes public knowledge.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +0.8% | QBTS: +0.5% | QUBT: +0.3% | QNT: +1.0% | IBM: +0.1% | HON: +0.1%", "category": "sideways"},
    {"scenario": "QNT files a routine patent application for a quantum error correction technique. The patent is incremental and covers a minor optimization.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.5% | RGTI: +0.3% | QBTS: -0.2% | QUBT: +0.1% | QNT: +0.8% | IBM: -0.1% | HON: +0.2%", "category": "sideways"},
    {"scenario": "An industry newsletter summarizes the past month in quantum computing. All information is already public and has been priced in.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: -0.5% | QBTS: +0.4% | QUBT: -0.2% | QNT: +0.2% | IBM: +0.1% | HON: -0.1%", "category": "sideways"},
    {"scenario": "Rigetti announces that their quantum computing platform now supports a new programming language (Julia). Minor developer experience improvement.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -0.4% | RGTI: +0.6% | QBTS: -0.3% | QUBT: +0.2% | QNT: -0.2% | IBM: +0.1% | HON: -0.1%", "category": "sideways"},
]

# ============================================================
# CONFLICTING SIGNALS (20)
# Mixed news that requires nuanced, moderate scoring.
# ============================================================

CONFLICTING_SCENARIOS = [
    {"scenario": "IonQ reports strong Q3 revenue (beat by 20%) BUT simultaneously announces a $300M secondary offering at a 15% discount. Great business, bad dilution.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +5.2% | RGTI: +2.1% | QBTS: +1.2% | QUBT: +0.8% | QNT: +3.8% | IBM: +0.3% | HON: +0.2%", "category": "conflicting"},
    {"scenario": "Rigetti wins a $100M government contract BUT their CTO resigns the same day to join a competitor. Great business win, terrible talent loss.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.0% | RGTI: +3.5% | QBTS: +0.6% | QUBT: +0.3% | QNT: +1.2% | IBM: +0.2% | HON: +0.1%", "category": "conflicting"},
    {"scenario": "D-Wave demonstrates quantum advantage on a real problem BUT the problem is so niche that the total addressable market is only $50M annually.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.3% | QBTS: +4.2% | QUBT: +0.2% | QNT: +0.6% | IBM: +0.1% | HON: +0.1%", "category": "conflicting"},
    {"scenario": "QNT achieves a major error correction milestone BUT the result required a processor that costs $50M to build, making commercial deployment economically unviable for years.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +5.5% | IBM: +0.2% | HON: +0.3%", "category": "conflicting"},
    {"scenario": "IonQ's revenue grows 50% year-over-year BUT gross margins decline from 65% to 35% due to heavy discounting. Growing fast but destroying unit economics.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +3.8% | RGTI: +1.2% | QBTS: +0.6% | QUBT: +0.4% | QNT: +2.5% | IBM: +0.2% | HON: +0.2%", "category": "conflicting"},
    {"scenario": "Google announces a superconducting breakthrough that validates the approach (bullish RGTI) BUT also announces they will compete directly with Rigetti for enterprise customers (bearish RGTI).", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.8% | RGTI: +2.5% | QBTS: +0.5% | QUBT: +0.3% | QNT: +1.0% | IBM: +0.4% | HON: +0.1%", "category": "conflicting"},
    {"scenario": "Rigetti achieves 99.5% gate fidelity (great technical progress) BUT simultaneously reveals their cash runway is only 6 months without additional funding.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.7% | RGTI: -1.5% | QBTS: +0.3% | QUBT: +0.2% | QNT: +0.8% | IBM: +0.2% | HON: +0.1%", "category": "conflicting"},
    {"scenario": "IonQ acquires a promising quantum networking startup (positive for long-term strategy) BUT pays 20x revenue in an all-stock deal that dilutes shareholders by 15%.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +4.5% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +2.2% | IBM: +0.2% | HON: +0.2%", "category": "conflicting"},
    {"scenario": "The US government announces $2B in quantum computing funding (bullish for sector) BUT restricts it to companies with >50% US-based employees, which excludes QNT (UK-headquartered).", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.5% | RGTI: +1.8% | QBTS: +1.0% | QUBT: +0.6% | QNT: +3.2% | IBM: +0.3% | HON: +0.2%", "category": "conflicting"},
    {"scenario": "D-Wave's quantum annealing approach is validated by a peer-reviewed study (positive) BUT the same study shows gate-model computers will surpass annealing within 2 years (negative long-term).", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.4% | QBTS: +3.5% | QUBT: +0.2% | QNT: +0.6% | IBM: +0.2% | HON: +0.1%", "category": "conflicting"},
    {"scenario": "IonQ's trapped-ion approach is shown to have the best error rates (positive) BUT also the slowest gate speeds, meaning it will take 100x longer to run practical algorithms (negative).", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.8% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +2.2% | IBM: +0.2% | HON: +0.2%", "category": "conflicting"},
    {"scenario": "Rigetti announces a major partnership with BMW for quantum optimization (positive) BUT the partnership is non-exclusive and BMW is simultaneously working with IonQ and D-Wave.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.2% | RGTI: +2.8% | QBTS: +0.8% | QUBT: +0.3% | QNT: +1.0% | IBM: +0.2% | HON: +0.1%", "category": "conflicting"},
    {"scenario": "QNT demonstrates 100 logical qubits (massive milestone) BUT the demonstration required 10,000 physical qubits, suggesting the overhead for error correction is much higher than their roadmap assumed.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.8% | RGTI: +1.0% | QBTS: +0.5% | QUBT: +0.3% | QNT: +6.8% | IBM: +0.3% | HON: +0.4%", "category": "conflicting"},
    {"scenario": "A new quantum computing ETF launches with $500M in initial assets (positive for sector liquidity) BUT the fund's top holding is Google (30%), not any pure-play quantum company.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.0% | RGTI: +1.5% | QBTS: +1.0% | QUBT: +0.6% | QNT: +2.2% | IBM: +0.3% | HON: +0.2%", "category": "conflicting"},
    {"scenario": "IonQ's CEO buys $2M of stock on the open market (positive insider signal) BUT the purchase comes one day before a previously undisclosed secondary offering is announced.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: -2.5% | RGTI: -1.2% | QBTS: -0.5% | QUBT: -0.4% | QNT: -1.8% | IBM: +0.1% | HON: -0.1%", "category": "conflicting"},
    {"scenario": "Rigetti's new 100-qubit processor achieves record performance (positive) BUT the yield rate is only 5%, meaning 95% of chips produced are defective and must be scrapped.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.6% | RGTI: +3.2% | QBTS: +0.4% | QUBT: +0.2% | QNT: +0.8% | IBM: +0.3% | HON: +0.1%", "category": "conflicting"},
    {"scenario": "The quantum computing sector receives a major government endorsement (positive) BUT the endorsement comes with new regulations requiring security clearances for all quantum computing employees, which will slow hiring.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +1.5% | RGTI: +1.0% | QBTS: +0.6% | QUBT: +0.4% | QNT: +1.8% | IBM: +0.2% | HON: +0.2%", "category": "conflicting"},
    {"scenario": "D-Wave secures a 5-year $200M contract (very positive) BUT the contract has aggressive performance milestones that, if missed, trigger full refund clauses.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.5% | RGTI: +0.3% | QBTS: +5.8% | QUBT: +0.2% | QNT: +0.6% | IBM: +0.1% | HON: +0.1%", "category": "conflicting"},
    {"scenario": "IonQ and QNT announce a joint research collaboration (positive for the field) BUT the collaboration requires both companies to share proprietary IP, reducing each company's competitive moat.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +2.2% | RGTI: +0.8% | QBTS: +0.4% | QUBT: +0.3% | QNT: +2.5% | IBM: +0.2% | HON: +0.2%", "category": "conflicting"},
    {"scenario": "QUBT announces a breakthrough in their neutral atom approach (positive) BUT the breakthrough was achieved by a team that has since left the company to start a competitor.", "market_context": "[MARKET CONTEXT - PRIOR 5 TRADING DAYS]\nIONQ: +0.3% | RGTI: +0.2% | QBTS: +0.3% | QUBT: +4.5% | QNT: +0.4% | IBM: +0.1% | HON: -0.1%", "category": "conflicting"},
]

ALL_SCENARIOS = DRAWDOWN_SCENARIOS + SIDEWAYS_SCENARIOS + CONFLICTING_SCENARIOS
print(f"Total robustness scenarios: {len(ALL_SCENARIOS)}")

# ============================================================
# Generation (same infrastructure)
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
        except Exception:
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
        category = scenario_data.get("category", "robustness")
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
    pending = [(i, s) for i, s in enumerate(ALL_SCENARIOS) if i not in existing]
    print(f"={'='*60}")
    print(f"V5 ROBUSTNESS EXAMPLES")
    print(f"={'='*60}")
    print(f"Total: {len(ALL_SCENARIOS)} | Done: {len(existing)} | Pending: {len(pending)}")
    print(f"  Drawdown: {len(DRAWDOWN_SCENARIOS)}")
    print(f"  Sideways: {len(SIDEWAYS_SCENARIOS)}")
    print(f"  Conflicting: {len(CONFLICTING_SCENARIOS)}")
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
