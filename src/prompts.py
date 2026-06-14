"""
System prompts and shared context for the Quantum Alpha Intelligence pipeline.

Incorporates:
- Fix 1: Updated ticker universe (MSFT/GOOGL/NVDA inactive, QNT added)
- Fix 2: Conditional arXiv score cap
- Fix 14: Minimum conviction threshold
- Market context awareness (Fix 3)
"""

# ============================================================
# Shared Context (included in all prompts)
# ============================================================

SHARED_CONTEXT = """**The quantum computing universe (10 tickers):**

**Active (scored):**
- IONQ: IonQ (trapped-ion, 100% quantum revenue, pure-play)
- RGTI: Rigetti Computing (superconducting, 100% quantum revenue, pure-play)
- QBTS: D-Wave Quantum (quantum annealing, 100% quantum revenue, pure-play)
- QUBT: Quantum Computing Inc. (neutral atom, 100% quantum revenue, pure-play)
- QNT: Quantinuum (trapped-ion, 100% quantum revenue, pure-play, IPO'd June 2026)
- IBM: International Business Machines (superconducting, ~2% quantum revenue)
- HON: Honeywell (trapped-ion, ~1% quantum revenue post-Quantinuum spinoff)

**Inactive (always 0.0):**
- MSFT: Microsoft — quantum revenue <0.1%, signal is noise
- GOOGL: Alphabet/Google — quantum revenue <0.1%, signal is noise
- NVDA: NVIDIA — moves on AI/GPU demand, not quantum news

**Scoring guidelines:**
- Scores range from -2.0 (strongly bearish) to +2.0 (strongly bullish)
- Pure-play companies (IONQ, RGTI, QBTS, QUBT, QNT): full range [-2.0, +2.0]
- HON: max +/-0.3 | IBM: max +/-0.15
- MSFT, GOOGL, NVDA: always 0.0 (inactive)
- If NOT about quantum computing: all scores = 0.0

**Technology competitive dynamics:**
- Trapped-ion breakthroughs → bullish IONQ/QNT/HON, bearish RGTI/IBM
- Superconducting breakthroughs → bullish RGTI/IBM, bearish IONQ/QNT/HON
- Error correction advances → benefit ALL gate-based approaches
- Government funding → broadly bullish for entire sector
- Negative news about one company → slightly bullish for direct competitors

**Technology validation vs. competitive threat (CRITICAL RULE):**
When a large company (Google, IBM, Microsoft) achieves a technology breakthrough:
- This VALIDATES the technology approach and is BULLISH for smaller same-technology pure-plays
- Empirical evidence: Google Willow (superconducting) → RGTI surged +89% in 5 days
- The market prices "this approach is viable" BEFORE "who wins within the approach"
- Pre-revenue quantum stocks move on sector/approach validation, not competitive positioning

Only score same-tech pure-plays BEARISH on competitor news when:
- The news is a zero-sum business win (contract, exclusive partnership, market share)
- The competitor demonstrates a proprietary moat others cannot replicate
- The article explicitly discusses competitive displacement of the smaller company

Default rule: Google/IBM superconducting breakthrough → BULLISH for RGTI
Default rule: Quantinuum trapped-ion breakthrough → BULLISH for IONQ (and vice versa)

**IONQ-QNT competitive dynamic:**
- QNT (Quantinuum) and IONQ are direct competitors (both trapped-ion, pure-play)
- Sector-wide events (funding, trapped-ion breakthroughs): both move together
- Company-specific events (contracts, earnings, talent): they may diverge or move opposite
- IONQ wins a contract → bullish IONQ, slightly bearish QNT (and vice versa)

**Minimum conviction rule (Fix 14, refined):**
The conviction threshold depends on the source type:

*For news articles about quantum computing companies or the quantum sector:*
- At least one pure-play ticker should almost always receive a non-zero score
- Quantum-specific news moves quantum stocks by definition — these are 100% quantum revenue companies
- Only assign 0.0 to a pure-play if the news is genuinely irrelevant to that specific company
- For adjacent companies (IBM, HON): assign non-zero when the news relates to their quantum division

*For arXiv papers and academic content:*
- Default to 0.0 unless there are clear, direct commercial implications
- Most papers do NOT move stocks — "no opinion" (0.0) is usually correct for academic content
- Only assign non-zero for genuine breakthroughs with near-term commercial relevance

*For non-quantum content:*
- All scores should be 0.0 — this news has no relevance to quantum stocks

General principle: Do not guess small directional scores when you lack conviction.
A wrong +0.1 is worse than a correct 0.0. But for news directly about quantum companies,
you should almost always have an opinion on the pure-play tickers.

**ArXiv paper scoring rules (Fix 2):**
- Default maximum absolute score for any ticker: 0.5
- Academic papers rarely move stocks in the short term regardless of technical significance
- Exception: if the paper is authored by researchers at a company in the active universe
  (IonQ, Rigetti, IBM, Quantinuum/QNT) AND demonstrates a concrete hardware result with
  measured metrics (not just theory or simulation), scores up to 1.0 are permitted for
  that company's ticker only
- Papers on pure theory, simulation, or unrelated quantum physics: all scores should be 0.0

**Market context awareness (Fix 3):**
- Consider liquidity when assigning scores. Be more conservative on low-liquidity names.
- Consider recent price action: if a stock is already up significantly in the past month,
  bullish news may already be priced in.
- If a stock is near its 52-week low, negative news may already be priced in.
- In high-volatility environments, signals decay faster."""


# ============================================================
# Structured Output Schema (10 tickers, no signal_decay)
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
                "QNT": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "IBM": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "HON": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "MSFT": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "GOOGL": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
                "NVDA": {"type": "object", "properties": {"score": {"type": "number"}, "reasoning": {"type": "string"}}, "required": ["score", "reasoning"], "additionalProperties": False},
            },
            "required": ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"],
            "additionalProperties": False
        },
        "event_type": {"type": "string"},
        "time_horizon": {"type": "string", "enum": ["intraday", "2-5 days", "1-2 weeks", "1+ month"]},
        "information_novelty": {"type": "string", "enum": ["high", "medium", "low"]},
        "technical_translation": {"type": "string"},
        "signal_rationale": {"type": "string"},
        "chain_of_thought": {"type": "string"}
    },
    "required": ["signal_vector", "event_type", "time_horizon", "information_novelty", "technical_translation", "signal_rationale", "chain_of_thought"],
    "additionalProperties": False
}


# ============================================================
# Prompt Templates
# ============================================================

REAL_ARTICLE_PROMPT = """You are a senior quantitative analyst specializing in the quantum computing sector. Analyze this article and produce a cross-sectional trading signal vector.

**IMPORTANT: Research the context thoroughly.** Look up the companies mentioned, check their stock performance around the article date ({date}), verify claims, and understand competitive dynamics. Use web browsing.

{shared_context}

{market_context}

**Your chain of thought MUST include:**
1. What is this article actually saying? (separate fact from hype)
2. Which technology approach does this relate to?
3. How significant is this relative to the company's roadmap?
4. How quickly will the market price this in?
5. What are the second-order effects on competitors?
6. Does this news have genuine conviction for any ticker, or should most/all be 0.0?

---

**Article:**
Title: {title}
Source: {source}
Date: {date}

{text}"""


ARXIV_PROMPT = """You are a senior quantitative analyst specializing in the quantum computing sector. Analyze this academic paper and produce a cross-sectional trading signal vector.

{shared_context}

{market_context}

**IMPORTANT ArXiv-specific rules:**
- Most academic papers do NOT move stocks. Default to 0.0 unless there's a clear reason.
- Maximum absolute score: 0.5 (unless company-authored hardware result, then up to 1.0)
- Pure theory papers: all scores 0.0
- Incremental improvements: all scores 0.0 or very small (0.05-0.1)
- Only genuine breakthroughs with commercial implications warrant scores > 0.3

**Your chain of thought MUST include:**
1. Is this paper commercially relevant or purely academic?
2. Is it authored by researchers at a company in our universe?
3. Does it demonstrate measured hardware results or is it theory/simulation?
4. Would a portfolio manager care about this paper? (usually no)

---

**Paper:**
Title: {title}
Source: arxiv
Date: {date}

{text}"""


QNT_SCENARIO_PROMPT = """You are a senior quantitative analyst. QNT (Quantinuum) IPO'd on NASDAQ on June 4, 2026 as an independent pure-play trapped-ion quantum computing company. It was previously the Quantinuum subsidiary of Honeywell (HON).

QNT is the closest public analog to IONQ: both are pure-play trapped-ion companies with 100% quantum revenue, competing directly for enterprise customers and government contracts.

Given the following scenario, produce a cross-sectional trading signal vector. Pay special attention to the competitive dynamic between IONQ and QNT:

- Sector-wide events (funding, trapped-ion breakthroughs, macro): IONQ and QNT move together
- Company-specific events (contracts, earnings, talent): IONQ and QNT may diverge or move opposite
- QNT absorbs signals at full pure-play magnitude ([-2.0, +2.0]), same as IONQ

{shared_context}

**Scenario:** {scenario}

**Your chain of thought MUST explain:**
1. Is this a sector-wide event or company-specific?
2. How does QNT move relative to IONQ and why?
3. What are the second-order effects on other tickers?"""


# ============================================================
# Fix 11: Reasoning Consistency Check
# ============================================================

BEARISH_KEYWORDS = {"negative", "bearish", "headwind", "pressure", "decline", "hurt", "drop", "risk", "threat", "weakness"}
BULLISH_KEYWORDS = {"positive", "bullish", "benefit", "tailwind", "growth", "boost", "surge", "advantage", "strength", "opportunity"}
EXCEPTION_PATTERNS = {"despite", "offset by", "although", "however", "nevertheless", "even though", "notwithstanding", "but overall", "on balance"}


def check_reasoning_consistency(signal: dict) -> dict:
    """
    Fix 11: Check each ticker's reasoning matches its score direction.
    Only flags when |score| > 0.3 and no exception pattern is present.
    
    Returns: {"pass": bool, "issues": list[str]}
    """
    issues = []
    sv = signal.get("signal_vector", {})
    
    for ticker, data in sv.items():
        score = data.get("score", 0)
        reasoning = data.get("reasoning", "").lower()
        
        # Only flag significant scores
        if abs(score) <= 0.3:
            continue
        
        # Check for exception patterns (negation/contrast)
        if any(pat in reasoning for pat in EXCEPTION_PATTERNS):
            continue
        
        # Check for contradictions
        words = set(reasoning.split())
        
        if score > 0.3:
            contradictions = BEARISH_KEYWORDS.intersection(words)
            if contradictions:
                issues.append(f"{ticker}: score={score:+.2f} but reasoning contains {contradictions}")
        elif score < -0.3:
            contradictions = BULLISH_KEYWORDS.intersection(words)
            if contradictions:
                issues.append(f"{ticker}: score={score:+.2f} but reasoning contains {contradictions}")
    
    return {"pass": len(issues) == 0, "issues": issues}
