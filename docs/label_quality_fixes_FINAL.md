# Label Quality Fixes: Final Implementation Specification

This document is the authoritative implementation spec. All adversarial feedback has been incorporated. Ready to execute.

---

## Active Ticker Universe (Post-Implementation)

| Ticker | Type | Score Range | Notes |
|--------|------|-------------|-------|
| IONQ | Pure-play | [-2.0, +2.0] | Trapped-ion |
| RGTI | Pure-play | [-2.0, +2.0] | Superconducting |
| QBTS | Pure-play | [-2.0, +2.0] | Quantum annealing |
| QUBT | Pure-play | [-2.0, +2.0] | Neutral atom |
| QNT | Pure-play | [-2.0, +2.0] | Trapped-ion (IPO'd June 4, 2026) |
| IBM | Adjacent | [-0.15, +0.15] | ~2% quantum revenue |
| HON | Adjacent | [-0.3, +0.3] | Post-spinoff, reduced quantum exposure |
| MSFT | Inactive | 0.0 always | Quantum revenue <0.1%, signal is noise |
| GOOGL | Inactive | 0.0 always | Quantum revenue <0.1%, signal is noise |
| NVDA | Inactive | 0.0 always | Anti-predictive (IC=-0.175), moves on AI not quantum |

---

## Fix 1: Update Ticker Universe

### What changes
- MSFT, GOOGL, NVDA become inactive (hard-coded 0.0)
- QNT added as pure-play trapped-ion
- HON remains active but with updated reasoning (post-spinoff, reduced exposure)

### Implementation

**`src/config.py`:**
```python
PURE_PLAY_TICKERS = {
    "IONQ": {"name": "IonQ", "technology": "Trapped Ion"},
    "RGTI": {"name": "Rigetti Computing", "technology": "Superconducting"},
    "QBTS": {"name": "D-Wave Quantum", "technology": "Quantum Annealing"},
    "QUBT": {"name": "Quantum Computing Inc", "technology": "Neutral Atom"},
    "QNT":  {"name": "Quantinuum", "technology": "Trapped Ion"},
}

ADJACENT_TICKERS = {
    "IBM": {"name": "IBM", "technology": "Superconducting", "quantum_revenue_pct": 2.0, "max_score": 0.15},
    "HON": {"name": "Honeywell", "technology": "Trapped Ion", "quantum_revenue_pct": 1.0, "max_score": 0.3,
            "note": "Post-Quantinuum spinoff (June 2026). Retains minority stake. Reduced but non-zero exposure."},
}

INACTIVE_TICKERS = {
    "MSFT": {"reason": "Quantum revenue <0.1%. IC=-0.033 (p=0.53), indistinguishable from noise."},
    "GOOGL": {"reason": "Quantum revenue <0.1%. IC=-0.023 (p=0.67), indistinguishable from noise."},
    "NVDA": {"reason": "Anti-predictive (IC=-0.175, p=0.0008). Moves on AI/GPU demand, not quantum news."},
}

ACTIVE_TICKERS = list(PURE_PLAY_TICKERS.keys()) + list(ADJACENT_TICKERS.keys())
# = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"]
```

**Post-process existing training data** (addresses adversarial feedback):
- Script: `scripts/postprocess_inactive_tickers.py`
- For every example in `manus_teacher_combined.jsonl`:
  - Set `signal_vector.MSFT.score = 0.0`, reasoning = "Inactive: quantum revenue exposure too low for meaningful signal."
  - Set `signal_vector.GOOGL.score = 0.0`, reasoning = "Inactive: quantum revenue exposure too low for meaningful signal."
  - Set `signal_vector.NVDA.score = 0.0`, reasoning = "Inactive: anti-predictive, moves on AI/GPU demand not quantum news."
- Save updated file

**Structured output schema update:**
- Add QNT to the schema (10 tickers total)
- Keep MSFT/GOOGL/NVDA in schema for backward compatibility (always 0.0)

---

## Fix 2: Conditional ArXiv Score Cap

### System prompt addition
```
**ArXiv paper scoring rules:**
- Default maximum absolute score for any ticker: 0.5
- Academic papers rarely move stocks in the short term regardless of technical significance
- Exception: if the paper is authored by researchers at a company in the active universe 
  (IonQ, Rigetti, IBM, Quantinuum/QNT) AND demonstrates a concrete hardware result with 
  measured metrics (not just theory or simulation), scores up to 1.0 are permitted for 
  that company's ticker only
- Papers on pure theory, simulation, or unrelated quantum physics: all scores should be 0.0
```

### Post-processing
```python
def clip_arxiv_scores(signal: dict, source: str) -> dict:
    if source == "arxiv":
        for ticker in signal["signal_vector"]:
            score = signal["signal_vector"][ticker]["score"]
            signal["signal_vector"][ticker]["score"] = max(-1.0, min(1.0, score))
    return signal
```

---

## Fix 3: Market Context at Training AND Inference

### New file: `src/market_context.py`

```python
"""
Market context provider for teacher pipeline and inference.
Computes 5d/30d returns, 52-week position, liquidity tier, and market regime.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from src.config import LIQUIDITY_TIERS, ACTIVE_TICKERS

MARKET_DIR = Path("data/market")

def get_market_context(date: str, tickers: list = None, market_dir: Path = None) -> str:
    """
    Compute market context block for a given date.
    Returns formatted string for prompt injection, or empty string if data unavailable.
    """
    if tickers is None:
        tickers = ACTIVE_TICKERS
    if market_dir is None:
        market_dir = MARKET_DIR
    
    target_date = pd.Timestamp(date)
    rows = []
    
    for ticker in tickers:
        path = market_dir / f"{ticker}.parquet"
        if not path.exists():
            continue
        
        df = pd.read_parquet(path)
        close = df["Close"]
        
        # Find the most recent trading day on or before target_date
        available = close.loc[:target_date]
        if len(available) < 30:
            continue
        
        current_price = available.iloc[-1]
        
        # 5-day return
        if len(available) >= 6:
            ret_5d = (available.iloc[-1] / available.iloc[-6] - 1) * 100
        else:
            ret_5d = None
        
        # 30-day return
        if len(available) >= 31:
            ret_30d = (available.iloc[-1] / available.iloc[-31] - 1) * 100
        else:
            ret_30d = None
        
        # 52-week position
        if len(available) >= 252:
            high_52w = available.iloc[-252:].max()
            low_52w = available.iloc[-252:].min()
        else:
            high_52w = available.max()
            low_52w = available.min()
        
        range_52w = high_52w - low_52w
        if range_52w > 0:
            position = (current_price - low_52w) / range_52w
            if position > 0.8:
                pos_label = "Near high"
            elif position < 0.2:
                pos_label = "Near low"
            else:
                pos_label = "Mid-range"
        else:
            pos_label = "N/A"
        
        # Liquidity tier
        liq = LIQUIDITY_TIERS.get(ticker, {}).get("tier", "unknown")
        
        rows.append({
            "ticker": ticker,
            "ret_5d": f"{ret_5d:+.1f}%" if ret_5d is not None else "N/A",
            "ret_30d": f"{ret_30d:+.1f}%" if ret_30d is not None else "N/A",
            "position": pos_label,
            "liquidity": liq.replace("_", " ").title(),
        })
    
    if not rows:
        return ""  # Graceful fallback: no context available
    
    # Format as markdown table
    lines = [f"**Market Context (as of {date}):**"]
    lines.append("| Ticker | 5d Ret | 30d Ret | 52w Position | Liquidity |")
    lines.append("|--------|--------|---------|--------------|-----------|")
    for r in rows:
        lines.append(f"| {r['ticker']} | {r['ret_5d']} | {r['ret_30d']} | {r['position']} | {r['liquidity']} |")
    
    # Add regime
    regime = get_market_regime(date, market_dir)
    if regime:
        lines.append(f"\n**Market regime:** {regime}")
    
    return "\n".join(lines)


def get_market_regime(date: str, market_dir: Path = None) -> str:
    """Compute market regime for a given date."""
    if market_dir is None:
        market_dir = MARKET_DIR
    
    spy_path = market_dir / "SPY.parquet"
    if not spy_path.exists():
        return ""
    
    spy = pd.read_parquet(spy_path)["Close"]
    target_date = pd.Timestamp(date)
    available = spy.loc[:target_date]
    
    if len(available) < 31:
        return ""
    
    spy_30d = (available.iloc[-1] / available.iloc[-31] - 1)
    
    # Quantum basket volatility
    basket_tickers = ["IONQ", "RGTI", "QBTS"]
    basket_rets = []
    for t in basket_tickers:
        path = market_dir / f"{t}.parquet"
        if path.exists():
            close = pd.read_parquet(path)["Close"]
            rets = close.pct_change().loc[:target_date].tail(30)
            basket_rets.append(rets)
    
    if basket_rets:
        basket_vol = pd.concat(basket_rets, axis=1).std().mean() * (252**0.5)
    else:
        basket_vol = 0
    
    if spy_30d > 0.05:
        regime = "Bull"
    elif spy_30d < -0.05:
        regime = "Bear"
    else:
        regime = "Neutral"
    
    if basket_vol > 0.80:
        regime += " / High Volatility"
    elif basket_vol < 0.40:
        regime += " / Low Volatility"
    
    return regime
```

### System prompt addition
```
Consider market context when assigning scores:
- If a stock is already up significantly (>30%) in the past month, bullish news may already 
  be priced in. Reduce score magnitude.
- If a stock is near its 52-week low, negative news may already be priced in.
- Be more conservative (lower magnitude) on low-liquidity names unless the event is truly 
  transformative.
- Consider the market regime: in high-volatility environments, signals decay faster.
```

### Inference update (`app_v2.py`)
- In `_run_inference()`, before constructing the user message:
```python
from src.market_context import get_market_context
# Compute context from parquet data (or return "" if unavailable)
context = get_market_context(date=today_str, tickers=ACTIVE_TICKERS)
if context:
    user_message = f"{context}\n\n{user_message}"
```
- Fallback: if parquet data is stale (>7 days old), prepend "Market context: unavailable (data stale)" so the model knows context is missing.

---

## Fix 3a: Retroactive Market Context Enrichment

### Script: `scripts/enrich_training_market_context.py`

```python
"""
Post-process existing training data to add market context.
Reads each example's date, computes context from parquet files, adds to record.
Only applies to examples with real dates (real_articles category).
"""
# For each example in manus_teacher_combined.jsonl:
#   if example has 'date' field AND date is within market data range:
#     compute market_context = get_market_context(date)
#     add field: example["market_context"] = market_context
#   else:
#     example["market_context"] = ""  (empty = not available)
```

**Coverage:** All 190 real articles have dates within market data range (2024-08-02 to 2025-12-23). Synthetic/edge case/negative examples get empty context (they don't have real dates).

**Fine-tuning conversion:** When building chat-format training data, prepend `market_context` to the user message if non-empty.

---

## Fix 4: Teacher Market Accuracy Metadata

### Script: `scripts/compute_teacher_accuracy.py`

- For each real article with a date:
  - Compute 5-day forward return for each active ticker from parquet data
  - Compare sign(predicted_score) vs sign(5d_return) for each active ticker
  - `teacher_market_accuracy` = fraction with matching direction (ignoring tickers where both are ~0)
- Add field to each record
- **Analysis only — never used in training loss or filtering**

---

## Fix 5: Remove signal_decay

### Changes
1. Remove from `SIGNAL_SCHEMA["properties"]` and `SIGNAL_SCHEMA["required"]` in pipeline scripts
2. Remove from system prompt output description
3. Keep `time_horizon` (useful metadata)

### Addressing adversarial feedback: Strip from existing training data
- Script: `scripts/postprocess_remove_decay.py`
- For every example in `manus_teacher_combined.jsonl`:
  - If `signal.signal_decay` exists: delete the field
- When converting to fine-tuning chat format: ensure assistant response JSON does not include `signal_decay`

---

## Fix 6: QTUM Sector Benchmark

### Implementation
1. Download: `yfinance.download("QTUM", start="2024-01-01")` → `data/market/QTUM.parquet`
2. Update `eval/market_data.py`:
```python
SECTOR_FACTOR = "QTUM"  # Replaces SECTOR_BASKET_TICKERS = ["IONQ", "RGTI", "QBTS"]
# Two-factor model: R_stock = alpha + beta_mkt * R_SPY + beta_sector * R_QTUM
# Fallback: if QTUM data unavailable for a date, use equal-weighted basket
```
3. Recompute `data/eval/abnormal_returns.csv` with new factor model

---

## Fix 7: Liquidity Metadata

### `src/config.py` addition
```python
LIQUIDITY_TIERS = {
    "IONQ": {"avg_daily_volume_usd": 180_000_000, "tier": "high"},
    "RGTI": {"avg_daily_volume_usd": 95_000_000, "tier": "high"},
    "QBTS": {"avg_daily_volume_usd": 70_000_000, "tier": "medium"},
    "QUBT": {"avg_daily_volume_usd": 45_000_000, "tier": "medium"},
    "QNT":  {"avg_daily_volume_usd": 150_000_000, "tier": "high"},  # Early trading, will stabilize
    "IBM":  {"avg_daily_volume_usd": 800_000_000, "tier": "very_high"},
    "HON":  {"avg_daily_volume_usd": 600_000_000, "tier": "very_high"},
}
```

---

## Fix 8: Semantic Clustering for Staleness

### Implementation with validation gate (addresses adversarial feedback)

**Step 1: Cluster and validate**
```python
# scripts/cluster_articles.py
# 1. Compute embeddings with all-MiniLM-L6-v2
# 2. Sliding 3-day window, cosine threshold 0.75
# 3. Assign prior_coverage_count
# 4. OUTPUT: Print 30 random clusters for manual review
# 5. GATE: Only proceed if <20% of sampled clusters are false positives
```

**Step 2: Apply (only after validation passes)**
- Add `prior_coverage_count` to user message: "Prior coverage: This is the {N}th article about this event in the past 72 hours."
- System prompt: "If prior_coverage_count > 0, the market has likely partially priced in this information. Scale down score magnitudes by roughly 30% for each additional prior article."

**Addressing adversarial concern (over-clustering):**
- Add a secondary check: if two articles in the same cluster mention DIFFERENT primary companies (e.g., one about IonQ, one about Rigetti), they should NOT be clustered even if cosine > 0.75. Extract primary ticker mentions and require at least one overlapping ticker.

---

## Fix 9: Full-Text Extraction (Best-Effort)

### Implementation with abort condition
```python
# scripts/enrich_article_text.py
# 1. Attempt trafilatura extraction for first 50 articles
# 2. If success_rate < 30%: ABORT, skip this fix entirely
# 3. If success_rate >= 30%: continue for all articles
# 4. Store full_text field, text_quality field
```

- System prompt addition: "If input is marked summary_only, note that short summaries may be missing critical context."
- **No other fixes depend on this.** Safe to skip if it fails.

---

## Fix 10: Event Deduplication (Eval Only)

### Script: `scripts/deduplicate_for_eval.py`
- Uses event_id from Fix 8's clustering
- Groups eval predictions by event_id
- Keeps only the FIRST (earliest timestamp) prediction per event
- Outputs `data/eval/predictions_deduplicated.jsonl`
- Updates `eval/run_evaluation.py` to use deduplicated by default
- **Never applied to training data**

---

## Fix 11: Reasoning Consistency Validation

### Implementation (addresses adversarial feedback)

```python
def check_reasoning_consistency(signal: dict) -> dict:
    """Check each ticker's reasoning matches its score direction."""
    BEARISH_WORDS = {"negative", "bearish", "headwind", "pressure", "decline", "hurt", "drop", "risk", "threat"}
    BULLISH_WORDS = {"positive", "bullish", "benefit", "tailwind", "growth", "boost", "surge", "advantage"}
    EXCEPTION_PATTERNS = {"despite", "offset by", "although", "however", "nevertheless", "even though", "notwithstanding"}
    
    issues = []
    for ticker, data in signal["signal_vector"].items():
        score = data.get("score", 0)
        reasoning = data.get("reasoning", "").lower()
        
        # Only flag if |score| > 0.3 (small scores with mixed sentiment are natural)
        if abs(score) <= 0.3:
            continue
        
        # Check for exception patterns (negation/contrast)
        has_exception = any(pat in reasoning for pat in EXCEPTION_PATTERNS)
        if has_exception:
            continue
        
        # Check per-ticker reasoning only (not global rationale)
        if score > 0.3:
            contradictions = BEARISH_WORDS.intersection(reasoning.split())
            if contradictions:
                issues.append(f"{ticker}: score={score:+.1f} but reasoning contains {contradictions}")
        elif score < -0.3:
            contradictions = BULLISH_WORDS.intersection(reasoning.split())
            if contradictions:
                issues.append(f"{ticker}: score={score:+.1f} but reasoning contains {contradictions}")
    
    return {"pass": len(issues) == 0, "issues": issues}
```

- Failed examples: retry once with instruction "Ensure your reasoning text for each ticker is consistent with that ticker's score direction."

---

## Fix 12: Market Regime Tagging

Implemented within `src/market_context.py` (see Fix 3 above). The `get_market_regime()` function is called as part of `get_market_context()` and included in the context block.

---

## Fix 13: ArXiv Rebalancing (10/45/15 Split)

### 70 new examples via Manus teacher pipeline

**10 genuinely important papers (scores 0.3-0.5):**
1. "IonQ researchers demonstrate 35 algorithmic qubits with 99.7% two-qubit gate fidelity on barium qubits"
2. "IBM Quantum team achieves below-threshold error correction on 127-qubit Eagle processor using heavy-hex code"
3. "Rigetti publishes Nature paper: 99.5% CZ gate fidelity on 84-qubit Ankaa-3, enabling practical error correction"
4. "Quantinuum demonstrates 50 logical qubits with real-time error correction on H2 processor"
5. "IonQ and Duke University demonstrate distributed quantum computing across 4 networked trapped-ion nodes"
6. "IBM researchers demonstrate quantum utility for materials simulation exceeding classical methods on 127 qubits"
7. "Rigetti team publishes first demonstration of fault-tolerant quantum algorithm on superconducting hardware"
8. "Quantinuum achieves quantum volume 2^21, demonstrating exponential scaling of their trapped-ion architecture"
9. "IonQ publishes results showing barium qubit T2 coherence times exceeding 10 seconds"
10. "D-Wave publishes peer-reviewed quantum speedup on real-world logistics optimization for a Fortune 500 client"

**45 incremental papers (scores 0.0-0.1):**
- "Improved bounds on quantum circuit depth for approximate optimization algorithms"
- "Noise characterization and mitigation in superconducting transmon qubits at millikelvin temperatures"
- "Variational quantum eigensolver convergence analysis for molecular hydrogen"
- "Quantum error correction with repetition codes: a pedagogical review"
- "Benchmarking quantum volume across different qubit modalities: a comparative study"
- "Theoretical analysis of cross-talk in multi-qubit superconducting processors"
- "Machine learning approaches for quantum state tomography"
- "Quantum approximate optimization algorithm performance on random graph instances"
- (37 more similar incremental/theoretical papers)

**15 unrelated papers (scores exactly 0.0):**
- "Quantum gravity and holographic entanglement entropy in AdS/CFT"
- "Topological phases and edge states in 2D condensed matter systems"
- "Quantum information scrambling in black hole evaporation"
- "Bell inequality violations in photonic systems at room temperature"
- "Quantum key distribution over 1000km satellite links"
- (10 more quantum physics papers with no commercial computing relevance)

**Generation prompt includes:** Fix 2 (arxiv cap), Fix 14 (minimum conviction), updated ticker universe.

---

## Fix 14: Minimum Conviction Threshold

### System prompt addition (refined wording per adversarial feedback)
```
**Minimum conviction rule:**
If you have no specific reason to believe this news will move a stock's price, assign 0.0. 
Do not guess directional scores when you lack conviction. "No opinion" (0.0) is a valid 
and often correct output.

- For pure-play companies (IONQ, RGTI, QBTS, QUBT, QNT): only assign non-zero when the 
  news has clear, direct implications for that company or its technology approach.
- For adjacent companies (IBM, HON): only assign non-zero when the news specifically 
  relates to their quantum computing division or directly impacts their quantum competitive 
  position.
- Most news does not meaningfully move stocks. Incremental developments, routine updates, 
  and tangentially related stories should receive 0.0.
- When in doubt, 0.0 is better than a small guess. A wrong +0.1 is worse than a correct 0.0.
```

**Why this wording works (addresses adversarial concern about over-zeroing IBM/HON):**
- The rule is framed as "no specific reason → 0.0" rather than "expected move < 1-2% → 0.0"
- IBM at +0.15 and HON at +0.3 are still valid when there IS a specific reason (e.g., "IBM Quantum division wins $500M contract")
- The rule targets the noise case: articles tangentially related to quantum where the model was guessing +0.05

---

## Fix 15: QNT Training Examples (30-40 examples)

### Competitive dynamics framework

QNT (Quantinuum) is the second pure-play trapped-ion company alongside IONQ. The relationship is:

**Sector-wide events (QNT and IONQ move together):**
- Government quantum funding → bullish both
- Trapped-ion technology breakthroughs (third-party/academia) → bullish both
- Superconducting breakthroughs (competitors) → bearish both
- General quantum hype/sentiment → both move same direction

**Competitive/zero-sum events (QNT and IONQ diverge):**
- IONQ wins contract → bullish IONQ, bearish QNT
- QNT wins contract → bullish QNT, bearish IONQ
- IONQ earnings miss → bearish IONQ, slightly bullish QNT
- QNT demonstrates better fidelity → bullish QNT, bearish IONQ

### Implementation

**Teacher prompt for QNT examples:**
```
You are a senior quantitative analyst. QNT (Quantinuum) IPO'd on NASDAQ on June 4, 2026 
as an independent pure-play trapped-ion quantum computing company. It was previously the 
Quantinuum subsidiary of Honeywell (HON).

QNT is the closest public analog to IONQ: both are pure-play trapped-ion companies with 
100% quantum revenue, competing directly for enterprise customers and government contracts.

Given the following scenario, produce a cross-sectional trading signal vector. Pay special 
attention to the competitive dynamic between IONQ and QNT:

- Sector-wide events (funding, trapped-ion breakthroughs, macro): IONQ and QNT move together
- Company-specific events (contracts, earnings, talent): IONQ and QNT may diverge or move opposite
- QNT absorbs signals at full pure-play magnitude ([-2.0, +2.0]), same as IONQ

[SHARED CONTEXT with QNT added]

**Scenario:** {scenario}
```

### 35 QNT scenarios

**Sector-wide (QNT and IONQ move together) — 12 scenarios:**
1. "US DOE announces $3B trapped-ion quantum computing initiative"
2. "Academic paper demonstrates trapped-ion qubits maintaining coherence for 1 hour"
3. "Google announces superconducting processor with 1000 qubits and below-threshold error rates"
4. "Congress passes Quantum Computing Advancement Act with $5B funding"
5. "New theoretical result shows trapped-ion approach has fundamental advantage over superconducting for error correction"
6. "China demonstrates 100-qubit trapped-ion processor, intensifying global competition"
7. "Major enterprise survey shows 60% of quantum-interested companies prefer trapped-ion approach"
8. "Quantum computing ETF sees $500M inflows in a single week"
9. "Jensen Huang says trapped-ion quantum computers will be commercially useful within 3 years"
10. "EU announces $2B quantum computing sovereignty fund focused on trapped-ion technology"
11. "Short seller publishes report claiming all quantum computing companies are overvalued"
12. "Quantum computing stocks drop 15% sector-wide on macro fears, no quantum-specific news"

**Competitive/zero-sum (QNT and IONQ diverge) — 18 scenarios:**
13. "QNT wins $200M US Air Force contract for quantum computing services, beating IONQ in final round"
14. "IONQ wins $150M contract with JPMorgan for quantum optimization, QNT was also bidding"
15. "QNT announces 99.99% single-qubit gate fidelity, surpassing IONQ's published results"
16. "IONQ demonstrates 50 algorithmic qubits, maintaining lead over QNT's 40"
17. "QNT reports first quarter revenue of $45M, beating estimates by 20%"
18. "IONQ misses Q3 revenue estimates by 15%, cites delayed enterprise deployments"
19. "QNT's chief scientist and 3 key researchers leave to join IONQ"
20. "IONQ announces exclusive partnership with AWS for trapped-ion quantum services"
21. "QNT announces exclusive partnership with Microsoft Azure for quantum services"
22. "Benchmark study shows QNT's H3 processor outperforms IONQ's Forte on quantum chemistry"
23. "IONQ announces acquisition of a quantum networking startup, expanding beyond pure computation"
24. "QNT raises $500M secondary offering at $80/share to fund manufacturing expansion"
25. "IONQ's CEO makes controversial comments, institutional investors reduce positions"
26. "QNT announces 30% workforce reduction to extend runway"
27. "Major customer publicly switches from IONQ to QNT, citing better error rates"
28. "IONQ patents a novel ion-shuttling technique that QNT's architecture cannot replicate"
29. "QNT announces a breakthrough in photonic interconnects for modular trapped-ion systems"
30. "Analyst initiates QNT at Overweight, IONQ at Underweight, citing valuation gap"

**Mixed/ambiguous — 5 scenarios:**
31. "IONQ and QNT announce a joint venture to develop quantum networking standards"
32. "A major trapped-ion patent held by QNT expires, allowing IONQ to use the technology freely"
33. "QNT's IPO lockup period expires, insiders begin selling shares"
34. "Both IONQ and QNT miss earnings in the same quarter, raising sector concerns"
35. "A new trapped-ion startup raises $1B, competing with both IONQ and QNT"

---

## Post-Processing Pipeline (Run After All Fixes)

### Script: `scripts/postprocess_all_fixes.py`

This single script applies all post-processing to existing training data:

```python
"""
Apply all post-processing fixes to existing training data.
Run once after implementing all fixes.
"""

def main():
    # 1. Zero out MSFT/GOOGL/NVDA scores (Fix 1)
    # 2. Remove signal_decay field (Fix 5)
    # 3. Add market_context field for dated examples (Fix 3a)
    # 4. Compute teacher_market_accuracy for dated examples (Fix 4)
    # 5. Add market_regime tag for dated examples (Fix 12)
    # 6. Clip any arxiv scores to [-1.0, 1.0] (Fix 2)
    # 7. Save updated manus_teacher_combined.jsonl
    pass
```

---

## Fine-Tuning Data Conversion Update

When converting `manus_teacher_combined.jsonl` → chat-format JSONL for fine-tuning:

1. **System prompt:** Use updated prompt with all new rules (inactive tickers, arxiv cap, minimum conviction, market context awareness)
2. **User message:** Prepend `market_context` if available, then source instruction, then article text
3. **Assistant response:** JSON output with:
   - All 10 tickers (IONQ, RGTI, QBTS, QUBT, QNT, IBM, HON, MSFT, GOOGL, NVDA)
   - MSFT/GOOGL/NVDA always 0.0
   - QNT scored for post-June-2026 examples, 0.0 for pre-June-2026 (or omitted)
   - No `signal_decay` field
   - All other fields: event_type, time_horizon, information_novelty, technical_translation, signal_rationale, chain_of_thought

---

## Implementation Order

| Phase | Fixes | Effort | Dependencies |
|-------|-------|--------|--------------|
| A | 5, 7, 12, 1 (config + post-process) | 2 hours | None |
| B | 3, 3a, 6, 4 | 3 hours | Fix 7 |
| C | 2, 14, 11 | 1 hour | None |
| D | 13 (70 arXiv examples), 15 (35 QNT examples) | 8-10 hours (Manus API) | Fixes 2, 14 |
| E | 8, 10 (with validation) | 2 hours | sentence-transformers |
| F | 9 (best-effort) | 1 hour | trafilatura |

**Total estimated time:** ~18-20 hours (mostly waiting for Manus API in Phase D)

---

## Success Criteria

After implementation, re-run evaluation and expect:
1. **Overall IC improvement:** From 0.055 → target 0.08+ (by removing anti-predictive NVDA noise)
2. **Direction accuracy improvement:** From 53.2% → target 56%+ (by reducing false small-score guesses)
3. **ArXiv IC improvement:** From -0.018 → target 0.0+ (by teaching correct arxiv behavior)
4. **No regression on best tickers:** HON IC should remain >0.15, RGTI should remain >0.10
