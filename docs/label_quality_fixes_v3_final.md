# Label Quality Fixes v3 (Final): Implementation Proposal & Adversarial Analysis

## Scope Changes from v2

- **Fix 1 revised:** Remove MSFT, GOOGL, NVDA only. Leave HON and QNT for a separate future effort.
- **Fix 3a (new):** Retroactively enrich existing 1,000 training examples with market context (post-processing, no regeneration).
- **Fix 13 revised:** ArXiv rebalancing with 10/45/15 split (important/incremental/unrelated).
- **Fix 14 (new):** Minimum conviction threshold — teach the model that "no opinion" (0.0) is valid and often correct.

---

## Part 1: Implementation Proposal (14 Fixes)

### Fix 1 (Revised): Update Ticker Universe — Remove MSFT/GOOGL/NVDA Only

**Active universe:** IONQ, RGTI, QBTS, QUBT, IBM, HON (6 tickers — unchanged from current except removing 3)
**Inactive (hard-coded 0.0):** MSFT, GOOGL, NVDA

**Implementation:**
1. `src/config.py`: Add `INACTIVE_TICKERS` dict:
```python
INACTIVE_TICKERS = {
    "MSFT": {"reason": "Quantum revenue exposure <0.1%, signal is noise (IC=-0.033, p=0.53)"},
    "GOOGL": {"reason": "Quantum revenue exposure <0.1%, signal is noise (IC=-0.023, p=0.67)"},
    "NVDA": {"reason": "Anti-predictive (IC=-0.175, p=0.0008). Moves on AI/GPU demand, not quantum news."},
}
```
2. Pipeline prompts: For MSFT/GOOGL/NVDA, hard-code score=0.0 with reasoning "Quantum revenue exposure too low for meaningful signal."
3. Schema: Keep all 9 tickers in output for backward compatibility. Active scoring only on IONQ, RGTI, QBTS, QUBT, IBM, HON.
4. Update `app_v2.py` COMPANY_INFO to mark MSFT/GOOGL/NVDA as inactive.

**HON/QNT:** Left unchanged for now. HON remains active (IC=0.1664, best predictor). QNT will be addressed separately when sufficient trading history exists.

---

### Fix 2 (Revised): Conditional ArXiv Score Cap

**System prompt addition:**
> For arXiv papers, the default maximum absolute score for any ticker is 0.5. Academic papers rarely move stocks in the short term. Exception: if the paper is authored by researchers at a company in the active universe (IonQ, Rigetti, IBM, Quantinuum) AND demonstrates a concrete hardware result with measured metrics (not just theory or simulation), scores up to 1.0 are permitted for that company's ticker only.

**Post-processing:** Clip all arxiv-sourced scores to [-1.0, 1.0] as a safety net. The prompt instruction handles the 0.5 default; the 1.0 hard limit catches edge cases.

**Implementation:**
- Add rule to system prompt in `scripts/manus_teacher_concurrent.py`
- Add post-processing clip in the pipeline's `validate_signal_format()` function
- Apply to both new training generation and evaluation predictions

---

### Fix 3 (Revised): Market Context at Training AND Inference

**Implementation:**
1. Create `src/market_context.py`:
```python
def get_market_context(date: str, tickers: list, market_dir: Path) -> str:
    """Compute market context block for a given date."""
    # Read parquet files, compute:
    # - 5-day return for each ticker
    # - 30-day return for each ticker  
    # - 52-week high/low position
    # - Liquidity tier from config
    # Return formatted markdown table
```

2. Update teacher pipeline: prepend market context to user message for all dated articles.

3. Update inference apps (`app.py`, `app_v2.py`): compute market context from parquet files at inference time, prepend to user message before model call.

4. System prompt addition:
> Consider liquidity when assigning scores. Be more conservative (lower magnitude) on low-liquidity names unless the event is truly transformative. Consider recent price action: if a stock is already up 40% in the past month, bullish news may already be priced in. If a stock has already dropped significantly, bearish news may be priced in.

**Market context format:**
```
**Market Context (as of {date}):**
| Ticker | 5d Ret | 30d Ret | 52w Position | Liquidity |
|--------|--------|---------|--------------|-----------|
| IONQ   | +3.2%  | +15.4%  | Near high    | High      |
| RGTI   | -1.1%  | -8.2%   | Mid-range    | High      |
| QBTS   | +0.5%  | +22.1%  | Near high    | Medium    |
| QUBT   | -2.3%  | -12.0%  | Near low     | Medium    |
| IBM    | +0.8%  | +3.1%   | Mid-range    | Very High |
| HON    | +0.2%  | +1.5%   | Mid-range    | Very High |
```

---

### Fix 3a (New): Retroactively Enrich Existing Training Data with Market Context

**Implementation:**
1. Create `scripts/enrich_training_market_context.py`:
   - Load `data/training/manus_teacher_combined.jsonl`
   - For each example with a `date` field (real articles: 190 examples)
   - Compute market context using `src/market_context.py`
   - Add `market_context` field to the record
   - Save updated file
2. For examples WITHOUT dates (synthetic, edge cases, negatives): add a placeholder or skip. These categories don't have real dates, so market context doesn't apply.
3. For the fine-tuning data conversion script (which converts raw JSONL to chat format): prepend market context to the user message when available.

**Coverage:** 190 real articles have dates within market data range (2024-08-02 to 2025-12-23, all covered by parquet data 2024-01-02 to 2026-06-05). All 190 can be enriched.

---

### Fix 4 (Revised): Teacher Market Accuracy Metadata (No Filtering)

**Implementation:**
1. Create `scripts/compute_teacher_accuracy.py`:
   - Load `manus_teacher_combined.jsonl` + market parquet data
   - For each real article with a date:
     - Compute 5-day forward return for each active ticker
     - Compare sign of predicted score vs sign of actual 5d return
     - Compute `teacher_market_accuracy`: fraction of active tickers with matching direction
   - Add field to each record
   - Save updated file
2. Output summary statistics: mean accuracy, distribution, worst examples
3. This field is for **analysis only** — never used in training loss or filtering

---

### Fix 5: Remove signal_decay from Training

**Implementation:**
- Remove `"signal_decay"` from `SIGNAL_SCHEMA["required"]` list in pipeline scripts
- Remove from `SIGNAL_SCHEMA["properties"]`
- Remove signal_decay instructions from system prompt
- Keep `time_horizon` (useful metadata)
- Add comment in `eval/run_evaluation.py`: `# signal_decay removed from training — discover empirically from IC-by-horizon curve`

---

### Fix 6: Add QTUM as Sector Benchmark

**Implementation:**
1. Download QTUM price data: `yfinance.download("QTUM", start="2024-01-01")`, save to `data/market/QTUM.parquet`
2. Update `eval/market_data.py`:
   - Change `SECTOR_BASKET_TICKERS = ["IONQ", "RGTI", "QBTS"]` to use QTUM
   - Two-factor model: `R_stock = alpha + beta_mkt * R_SPY + beta_sector * R_QTUM`
   - Fallback to old basket for dates before QTUM history begins (QTUM launched 2018, so this is just a safety check)
3. Recompute abnormal returns with the new factor model

---

### Fix 7: Liquidity Metadata

**Implementation:**
- Add to `src/config.py`:
```python
LIQUIDITY_TIERS = {
    "IONQ": {"avg_daily_volume_usd": 180_000_000, "tier": "high"},
    "RGTI": {"avg_daily_volume_usd": 95_000_000, "tier": "high"},
    "QBTS": {"avg_daily_volume_usd": 70_000_000, "tier": "medium"},
    "QUBT": {"avg_daily_volume_usd": 45_000_000, "tier": "medium"},
    "IBM":  {"avg_daily_volume_usd": 800_000_000, "tier": "very_high"},
    "HON":  {"avg_daily_volume_usd": 600_000_000, "tier": "very_high"},
}
```
- Verify against recent Yahoo Finance volume × price data
- Used by `src/market_context.py` for the context block

---

### Fix 8: Semantic Clustering for Staleness

**Implementation:**
1. Create `scripts/cluster_articles.py`:
   - Install `sentence-transformers`, use `all-MiniLM-L6-v2`
   - Compute embeddings for all articles in `data/raw/articles.jsonl`
   - Sliding 3-day window, cosine similarity threshold 0.75
   - Assign `prior_coverage_count` (0 = first, 1+ = follow-up)
   - Save enriched articles
2. Update teacher pipeline user message: "Prior coverage: This is the {N}th article about this event in the past 72 hours."
3. System prompt addition: "If prior_coverage_count > 0, reduce information_novelty accordingly. The market has likely already partially priced in this information. Scale down score magnitudes by roughly 30% for each additional prior article."

**Note:** Changed from 50% reduction to 30% per the adversarial feedback that 50% was too aggressive.

---

### Fix 9: Full-Text Extraction (Best-Effort)

**Implementation:**
1. Create `scripts/enrich_article_text.py`:
   - Install `trafilatura`
   - For articles with `source == "news"` and URL: attempt extraction
   - Store in `full_text` field, add `text_quality` ("full" or "summary_only")
   - Save enriched dataset
2. Update pipeline: use `full_text` when available, fall back to `text`
3. System prompt: "If input is marked summary_only, note that short summaries may be missing critical context that could change your analysis."
4. **Abort condition:** If success rate < 30% after first 50 attempts, skip this fix entirely.

---

### Fix 10: Event Deduplication (Eval Only)

**Implementation:**
1. Extend Fix 8 clustering to assign `event_id` per cluster
2. Create `scripts/deduplicate_for_eval.py`:
   - Group eval predictions by `event_id`
   - Keep only first (earliest) prediction per event
   - Output `data/eval/predictions_deduplicated.jsonl`
3. Update `eval/run_evaluation.py`: compute IC on deduplicated by default
4. **Never** apply to training data

---

### Fix 11: Reasoning Consistency Validation

**Implementation:**
- Post-generation check in pipeline for each ticker:
  - Only flag if `|score| > 0.3` (ignore small scores where mixed sentiment is natural)
  - Bearish keywords: "negative", "bearish", "headwind", "pressure", "decline", "hurt", "drop"
  - Bullish keywords: "positive", "bullish", "benefit", "tailwind", "growth", "boost", "surge"
  - Exception patterns: skip if reasoning contains "despite", "offset by", "although", "however.*but", "nevertheless"
- Add `reasoning_consistency` field: "pass" or "fail" with details
- Failed examples: retry once with instruction "Ensure your reasoning text is consistent with the score direction"

---

### Fix 12: Market Regime Tagging

**Implementation:**
- Add to `src/market_context.py`:
```python
def get_market_regime(date: str, spy_data: pd.Series, basket_data: pd.DataFrame) -> str:
    spy_30d = spy_data.loc[:date].tail(30).sum()
    basket_vol = basket_data.loc[:date].tail(30).std().mean() * (252**0.5)
    
    if spy_30d > 0.05:
        regime = "bull"
    elif spy_30d < -0.05:
        regime = "bear"
    else:
        regime = "neutral"
    
    if basket_vol > 0.80:
        regime += "_high_vol"
    
    return regime
```
- Add regime tag to market context block
- Enables analysis: "IC in bull vs bear markets"

---

### Fix 13 (Revised): Rebalance Training Data for ArXiv (10/45/15 Split)

**Implementation:**
1. Use Manus teacher pipeline to generate 70 new arXiv training examples:
   - **10 genuinely important papers** (scores 0.3-0.5): Company-authored hardware results
     - "IonQ researchers demonstrate 35 algorithmic qubits with 99.7% fidelity"
     - "IBM team achieves below-threshold error correction on 72-qubit processor"
     - "Rigetti publishes 99.5% two-qubit gate fidelity on Ankaa-3"
     - etc.
   - **45 incremental papers** (scores 0.0-0.1): Typical academic output
     - "Improved bounds on quantum circuit depth for approximate optimization"
     - "Noise characterization in superconducting transmon qubits"
     - "Variational quantum eigensolver convergence analysis"
     - etc.
   - **15 unrelated papers** (scores exactly 0.0): Not commercially relevant
     - "Quantum gravity and holographic entanglement entropy"
     - "Topological phases in 2D condensed matter systems"
     - "Quantum information scrambling in black holes"
     - etc.
2. Use revised system prompt with arxiv cap (Fix 2) and minimum conviction threshold (Fix 14)
3. Save to `data/training/manus_arxiv_rebalance.jsonl`
4. Add to combined training file
5. Target distribution after rebalancing: ~60% news, ~25% arXiv, ~15% other

---

### Fix 14 (New): Minimum Conviction Threshold

**System prompt addition:**
> **Minimum conviction rule:** If the expected stock impact of this news is less than 1-2% over 5 days, assign a score of 0.0. Only assign non-zero scores when you have genuine conviction about a meaningful directional move. "No opinion" (0.0) is a valid and often correct output. Most news does not meaningfully move stocks — especially for diversified companies or incremental developments. Err on the side of 0.0 rather than guessing a small directional score.

**Implementation:**
1. Add the above to the system prompt in all pipeline scripts
2. Add to the inference prompt in `app_v2.py`
3. For training data: this will naturally be reflected in Fix 13's arXiv examples (45 incremental papers scoring 0.0) and in any future regenerated data
4. For existing training data: no change needed (the model will learn the threshold from the prompt + new examples)

**Rationale:** The eval analysis shows the model is random on small moves (<2%) but accurate on large moves (59% at >20%). Teaching it to output 0.0 on low-conviction cases eliminates noise and improves overall IC by removing false signals.

---

## Part 2: Adversarial Analysis (v3)

### Fix 1 (Remove MSFT/GOOGL/NVDA) — LOW RISK

**Assessment:** Clean and well-justified.

- MSFT IC=-0.033 (p=0.53): pure noise
- GOOGL IC=-0.023 (p=0.67): pure noise  
- NVDA IC=-0.175 (p=0.0008): actively harmful

**Remaining concern:** The existing 1,000 training examples all have MSFT/GOOGL/NVDA scores. If you hard-code them to 0.0 in the prompt but the training data shows non-zero scores, the model receives contradictory signals. Two options:
1. Post-process existing training data to set MSFT/GOOGL/NVDA scores to 0.0 (recommended)
2. Accept the inconsistency and rely on the new prompt to override (less clean)

**Severity:** Low. Option 1 is a simple post-processing step.

**Verdict:** ✅ Implement. Add a post-processing step to zero out MSFT/GOOGL/NVDA in existing training data.

---

### Fix 2 (Conditional ArXiv Cap) — LOW RISK

**Assessment:** Well-designed. The conditional exception is reasonable.

**Remaining concern:** The 1.0 hard limit for company-authored papers may still be too generous. The eval data shows arxiv IC is essentially zero across the board. Even company-authored papers may not move stocks in 5 days (the market often ignores technical papers regardless of authorship).

**Counter-argument:** The cap is for the TEACHER's labels, not the student's predictions. If the teacher assigns 0.8 to a genuinely important IonQ paper, and the stock does move, that's a correct high-quality label. The issue was the teacher assigning +2.0 — the cap prevents that.

**Severity:** Low.

**Verdict:** ✅ Implement as designed.

---

### Fix 3 + 3a (Market Context, Both Sides + Retroactive) — LOW RISK

**Assessment:** The retroactive enrichment eliminates the distribution shift concern from v2. All 190 real articles have dates within market data range.

**Remaining concerns:**

1. **Synthetic/edge case/negative examples don't have real dates.** These 560 examples won't have market context. The model will see some training examples with market context and some without. This is actually fine — at inference, some articles may not have market data available (e.g., if parquet data is stale), so the model should handle both cases.

2. **Market context at inference requires fresh data.** The parquet files end at 2026-06-05. For articles after that date, the context will be stale. Need either:
   - A live Yahoo Finance fetch at inference time (adds latency, requires internet)
   - A graceful fallback: "Market context: unavailable for this date"

3. **Prompt length budget.** Market context adds ~150-200 tokens. With 4096 max_seq_length, this is ~4% of the budget. Acceptable, but worth noting for very long articles.

**Severity:** Low. The retroactive enrichment is the key improvement. The edge cases (no date, stale data) are handled gracefully by making context optional.

**Verdict:** ✅ Implement. Add fallback for missing dates/data.

---

### Fix 4 (Metadata Only) — NO RISK

**Assessment:** Purely additive. Cannot harm training.

**Verdict:** ✅ Implement.

---

### Fix 5 (Remove signal_decay) — NO RISK

**Assessment:** Removes a field that adds no value and wastes model capacity.

**One consideration:** The existing training data has `signal_decay` in the assistant responses. When converting to fine-tuning format, you'll need to strip it from the target output. Otherwise the model still learns to produce it.

**Verdict:** ✅ Implement. Ensure the fine-tuning data conversion strips `signal_decay` from assistant messages.

---

### Fix 6 (QTUM Benchmark) — LOW RISK

**Assessment:** Strictly better than the 3-stock equal-weighted basket.

**Verified:** QTUM (Defiance Quantum ETF) launched September 2018. Full coverage of eval period (Dec 2024 - June 2026).

**Verdict:** ✅ Implement.

---

### Fix 7 (Liquidity Metadata) — NO RISK

**Assessment:** Informational only. Used by Fix 3's context block.

**Verdict:** ✅ Implement.

---

### Fix 8 (Semantic Clustering) — MEDIUM RISK

**Assessment:** Conceptually sound but parameter-sensitive.

**Remaining concerns:**

1. **Over-clustering within quantum sector.** Two articles — "IonQ announces trapped-ion breakthrough" and "Rigetti announces superconducting breakthrough" — might cluster together (both mention "quantum computing breakthrough") even though they're about different companies with opposite signal implications. This would incorrectly mark the second as "follow-up coverage" and reduce its scores.

2. **The 30% reduction rule is still somewhat arbitrary.** What if the 2nd article contains genuinely new information? Example: Article 1: "Company X announces partnership" → Article 2: "Partnership valued at $500M, details emerge" — the second article has new material information.

3. **Embedding model quality.** `all-MiniLM-L6-v2` is general-purpose. It may not distinguish between "same event, different article" and "same sector, different event" in financial news.

**Mitigation:** Add a manual validation step: after clustering, sample 30 clusters and verify they're genuinely about the same event. If >20% are false positives, raise the threshold to 0.85.

**Severity:** Medium. The concept improves data quality but the parameters need empirical validation.

**Verdict:** ⚠️ Implement with validation gate. Don't apply score reduction until clustering quality is verified.

---

### Fix 9 (Full-Text Extraction) — MEDIUM RISK

**Assessment:** Best-effort with abort condition. The 30% threshold is a good safety valve.

**Remaining concerns:**

1. **Google News URLs.** Tested: the articles use `https://news.google.com/rss/articles/CBMi...` URLs. Trafilatura's ability to resolve these is uncertain. It may follow the redirect, or it may fail.

2. **Even if extraction works, the value is questionable.** The model was trained on summaries and performs at IC=0.055. Adding full text might improve quality, but it also means the model learns to expect longer inputs. At inference, will users paste full articles or just summaries/headlines?

3. **Copyright/ToS risk.** Storing full article text from news sites may violate terms of service.

**Severity:** Medium. The abort condition limits downside. If it works, it's a nice-to-have. If it doesn't, nothing is lost.

**Verdict:** ⚠️ Implement as best-effort. Don't depend on it for other fixes.

---

### Fix 10 (Event Deduplication) — NO RISK (Eval Only)

**Assessment:** Correctly scoped. Improves eval rigor without touching training.

**Verdict:** ✅ Implement.

---

### Fix 11 (Reasoning Consistency) — LOW RISK

**Assessment:** The |score| > 0.3 threshold and exception patterns address v2 concerns.

**One edge case:** "The trapped-ion breakthrough is positive for IONQ but creates competitive pressure on RGTI." If RGTI's score is -0.5 and the reasoning mentions "positive" (referring to IONQ), the keyword check might false-positive. Solution: check keywords only in the SPECIFIC ticker's reasoning string, not the overall rationale.

**Severity:** Low. The per-ticker scoping prevents cross-contamination.

**Verdict:** ✅ Implement. Ensure keyword check is scoped to each ticker's individual reasoning field.

---

### Fix 12 (Market Regime) — NO RISK

**Assessment:** Purely additive metadata.

**Verdict:** ✅ Implement.

---

### Fix 13 (ArXiv Rebalance, 10/45/15) — LOW RISK

**Assessment:** Addresses the most critical gap. The 10/45/15 split matches real-world base rates.

**Remaining concerns:**

1. **The 70 new examples use the REVISED prompt (with Fixes 2, 14).** This means they'll have the arxiv cap and minimum conviction threshold baked in. The existing 1,000 examples don't have these rules. This creates a mild inconsistency: old examples may have higher scores for similar content than new examples.

   **Mitigation:** This is actually desirable. The new examples teach the model the CORRECT behavior (lower scores for arxiv). The old examples with higher scores will be outvoted by the 70 new examples that demonstrate the correct pattern. Fine-tuning naturally handles this — the model learns the majority pattern.

2. **Scenario generation quality.** The 10 "important" papers need to be realistic. If the scenarios are too generic ("IonQ publishes breakthrough"), the teacher might generate unrealistic articles. Provide specific, grounded scenarios with concrete numbers.

**Severity:** Low. Well-designed fix.

**Verdict:** ✅ Implement.

---

### Fix 14 (Minimum Conviction Threshold) — LOW-MEDIUM RISK

**Assessment:** Addresses a real problem (random on small moves) with an elegant solution.

**Remaining concerns:**

1. **Risk of over-zeroing.** If the model learns to output 0.0 too aggressively, it might miss genuine small signals. The 1-2% threshold in the prompt is a suggestion, not a hard rule — the model may interpret it differently. Some articles genuinely warrant a +0.2 score (small but real).

2. **Interaction with Fix 1.** After removing MSFT/GOOGL/NVDA (which were already near-zero), the remaining tickers (IONQ, RGTI, QBTS, QUBT, IBM, HON) are all ones where the model SHOULD have opinions. The minimum conviction threshold might cause the model to zero-out IBM and HON too aggressively (since their caps are already low: IBM ±0.15, HON ±0.3).

3. **Training data inconsistency.** The existing 1,000 examples were generated WITHOUT this threshold. Many have small non-zero scores (e.g., +0.05 for IBM). The new arXiv examples (Fix 13) will have more zeros. The model sees both patterns during fine-tuning. This is manageable — the model will learn from the prompt instruction — but it's not perfectly clean.

**Mitigation for concern #2:** Clarify in the prompt that the threshold applies to the EXPECTED STOCK MOVE, not the score itself. A score of +0.15 for IBM (which is the maximum) represents a meaningful conviction about IBM's quantum division. The threshold is about "don't guess" — not "don't score small."

**Severity:** Low-Medium. The concept is correct. The wording needs to be precise to avoid over-zeroing adjacent tickers.

**Revised prompt wording suggestion:**
> "If you have no specific reason to believe this news will move a stock's price, assign 0.0. Do not guess directional scores when you lack conviction. For pure-play quantum companies (IONQ, RGTI, QBTS, QUBT), only assign non-zero when the news has clear, direct implications. For adjacent companies (IBM, HON), only assign non-zero when the news specifically relates to their quantum division."

**Verdict:** ✅ Implement with refined wording.

---

## Summary: Final Risk Matrix

| Fix | Risk Level | Verdict | Dependencies |
|-----|-----------|---------|--------------|
| 1 (remove MSFT/GOOGL/NVDA) | Low | ✅ Implement | Post-process existing data |
| 2 (arxiv cap, conditional) | Low | ✅ Implement | None |
| 3 (market context, both) | Low | ✅ Implement | Fix 7 (liquidity data) |
| 3a (retroactive enrichment) | Low | ✅ Implement | Fix 3, Fix 7 |
| 4 (metadata only) | None | ✅ Implement | Market data |
| 5 (remove signal_decay) | None | ✅ Implement | None |
| 6 (QTUM benchmark) | Low | ✅ Implement | None |
| 7 (liquidity metadata) | None | ✅ Implement | None |
| 8 (semantic clustering) | Medium | ⚠️ Validate first | sentence-transformers |
| 9 (full-text extraction) | Medium | ⚠️ Best-effort | trafilatura |
| 10 (event dedup, eval only) | None | ✅ Implement | Fix 8 |
| 11 (reasoning consistency) | Low | ✅ Implement | None |
| 12 (market regime) | None | ✅ Implement | Market data |
| 13 (arxiv rebalance) | Low | ✅ Implement | Fixes 2, 14 |
| 14 (minimum conviction) | Low-Medium | ✅ Implement (refine wording) | None |

---

## Recommended Implementation Order

**Phase A (no dependencies, immediate):**
1. Fix 5 (remove signal_decay)
2. Fix 7 (liquidity metadata)
3. Fix 12 (market regime tagging)
4. Fix 1 (remove MSFT/GOOGL/NVDA + post-process existing data)

**Phase B (market data infrastructure):**
5. Fix 3 (market context module)
6. Fix 3a (retroactive enrichment)
7. Fix 6 (QTUM benchmark)
8. Fix 4 (teacher accuracy metadata)

**Phase C (prompt improvements for new data):**
9. Fix 2 (arxiv cap)
10. Fix 14 (minimum conviction)
11. Fix 11 (reasoning consistency)

**Phase D (new training data generation):**
12. Fix 13 (arXiv rebalance — 70 new examples via Manus API)

**Phase E (advanced, validate first):**
13. Fix 8 (semantic clustering)
14. Fix 10 (event deduplication)
15. Fix 9 (full-text extraction — best effort)

---

## Post-Implementation Checklist

After all fixes are applied:
1. [ ] Regenerate fine-tuning JSONL with updated schema (no signal_decay, MSFT/GOOGL/NVDA=0.0, market context prepended)
2. [ ] Re-run evaluation with QTUM factor model and deduplicated predictions
3. [ ] Compare new IC vs old IC (expect improvement from removing anti-predictive tickers)
4. [ ] Verify the 70 new arXiv examples pass format validation
5. [ ] Commit to `fix/label-quality` branch and push
6. [ ] Fine-tune new model version and compare to baseline
