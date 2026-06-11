# Label Quality Fixes v2: Revised Implementation Proposal & Adversarial Analysis

## Revisions from v1

The following changes were made based on feedback:

1. **Fix 1 (HON):** Accepted — HON is removed because the stock fundamentally changed after the Quantinuum spinoff (June 4, 2026). The historical IC=0.1664 reflects pre-spinoff behavior that is no longer relevant.
2. **Fix 3 (revised):** Market context will be applied at BOTH training AND inference time, eliminating distribution shift concerns.
3. **Fix 4 (revised):** No filtering. Add `teacher_market_accuracy` as metadata only — not used for weighting or filtering during fine-tuning.
4. **Fix 2 (revised):** Conditional cap — company-authored hardware papers can score up to 1.0.
5. **Fix 13 (new):** Rebalance training data to include 60-80 arXiv examples matching eval distribution.

---

## Part 1: Revised Implementation Proposal

### Fix 1: Update Ticker Universe

**Active universe:** IONQ, RGTI, QBTS, QUBT, QNT, IBM (6 tickers)
**Inactive (hard-coded 0.0):** MSFT, GOOGL, NVDA, HON

**Implementation:**
- `src/config.py`: Add `INACTIVE_TICKERS` dict with removal reasons. Add QNT to `PURE_PLAY_TICKERS`.
- Pipeline prompts: Output 0.0 for inactive tickers with standard reasoning strings.
- Schema: Keep all original 9 tickers + add QNT (10 total) for backward compatibility.
- Inference apps (`app.py`, `app_v2.py`): Update `QUANTUM_UNIVERSE` to include QNT, mark inactive tickers.

**Notes:**
- QNT IPO'd June 4, 2026 at $60/share on NASDAQ. No historical price data before that date.
- For training examples dated before June 4, 2026: HON should still be scored (it was Quantinuum's parent). For examples after: HON=0.0, QNT is active.
- Need to download QNT price data for the ~7 days of trading history available.

---

### Fix 2 (Revised): Conditional ArXiv Score Cap

**Rule in system prompt:**
> For arXiv papers, default maximum absolute score is 0.5. Exception: if the paper is authored by researchers at a company in the active universe (IonQ, Rigetti, IBM, Quantinuum/QNT) AND demonstrates a concrete hardware result with measured metrics (not just theory or simulation), scores up to 1.0 are permitted for that company's ticker only.

**Implementation:**
- Add rule to system prompt in pipeline scripts
- Post-processing validation: clip arxiv scores to [-0.5, 0.5] by default, allow [-1.0, 1.0] only if the signal's `chain_of_thought` explicitly mentions company authorship and hardware metrics
- In practice, the post-processing will just clip to [-1.0, 1.0] for arxiv (the prompt handles the 0.5 default)

---

### Fix 3 (Revised): Market Context at Training AND Inference

**Implementation:**
- Create `src/market_context.py`:
  - `get_market_context(date: str, tickers: list) -> str` — reads parquet files, computes 5d/30d returns, 52-week position, liquidity tier
  - Returns formatted text block for prompt injection
- Update teacher pipeline (`scripts/manus_teacher_concurrent.py`): prepend market context to user message for all real articles
- Update inference apps (`app.py`, `app_v2.py`): compute market context from parquet files at inference time, prepend to user message before model call
- Add to system prompt: "Consider liquidity when assigning scores. Be more conservative on low-liquidity names. Consider recent price action: if a stock is already up 40% in the past month, bullish news may already be priced in."

**Market context format:**
```
**Market Context (as of {date}):**
| Ticker | 5d Return | 30d Return | 52w Position | Liquidity |
|--------|-----------|------------|--------------|-----------|
| IONQ   | +3.2%     | +15.4%     | Near high    | $180M/day |
| RGTI   | -1.1%     | -8.2%      | Mid-range    | $95M/day  |
...
```

---

### Fix 4 (Revised): Teacher Market Accuracy Metadata (No Filtering)

**Implementation:**
- Create `scripts/compute_teacher_accuracy.py`:
  - Load `manus_teacher_combined.jsonl` + market data
  - For each real article with a date: compute 5d forward return per active ticker
  - Compare direction of predicted score vs actual return
  - Add `teacher_market_accuracy` field: fraction of active tickers where direction matched
- Output: updated `manus_teacher_combined.jsonl` with the new field
- This field is for analysis ONLY — not used in training loss or filtering

---

### Fix 5: Remove signal_decay from Training

**Implementation:**
- Remove `signal_decay` from `SIGNAL_SCHEMA` in pipeline scripts
- Remove from system prompt required output description
- Keep `time_horizon` (useful metadata for downstream)
- Add comment in eval pipeline: "signal_decay should be discovered empirically from IC-by-horizon curve"

---

### Fix 6: Add QTUM as Sector Benchmark

**Implementation:**
- Download QTUM price data via yfinance, save to `data/market/QTUM.parquet`
- Update `eval/market_data.py`: replace `SECTOR_BASKET_TICKERS = ["IONQ", "RGTI", "QBTS"]` with QTUM
- Two-factor model: `R_stock = alpha + beta_mkt * R_SPY + beta_sector * R_QTUM`
- Keep old basket as fallback for dates before QTUM had sufficient history

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
    "QNT":  {"avg_daily_volume_usd": 200_000_000, "tier": "high"},  # IPO week volume
    "IBM":  {"avg_daily_volume_usd": 800_000_000, "tier": "very_high"},
}
```
- Verify against recent Yahoo Finance data
- Used by `src/market_context.py` for the context block

---

### Fix 8: Semantic Clustering for Staleness

**Implementation:**
- Create `scripts/cluster_articles.py`:
  - Install `sentence-transformers`, use `all-MiniLM-L6-v2`
  - Compute embeddings for all articles in `data/raw/articles.jsonl`
  - Sliding 3-day window, cosine similarity threshold 0.75
  - Assign `prior_coverage_count` (0 = first article, 1+ = follow-up)
  - Save enriched articles
- Update teacher pipeline: include `prior_coverage_count` in user message
- System prompt addition: "If prior_coverage_count > 0, reduce information_novelty accordingly. Scale down score magnitudes by roughly 50% for each additional prior article."

---

### Fix 9: Full-Text Extraction

**Implementation:**
- Create `scripts/enrich_article_text.py`:
  - Install `trafilatura`
  - For articles with `source == "news"` and URL field: attempt full-text extraction
  - Store in `full_text` field, add `text_quality` field ("full" or "summary_only")
  - Save enriched dataset
- Update pipeline: use `full_text` when available, fall back to `text`
- System prompt: "If input is marked summary_only, reduce confidence in your signal."
- Add `trafilatura` to requirements.txt

---

### Fix 10: Event Deduplication (Eval Only)

**Implementation:**
- Extend Fix 8's clustering to assign `event_id` per cluster
- Create `scripts/deduplicate_for_eval.py`:
  - Group eval predictions by `event_id`
  - Keep only first (earliest) prediction per event
  - Output `data/eval/predictions_deduplicated.jsonl`
- Update `eval/run_evaluation.py`: compute IC on deduplicated by default
- Do NOT apply to training data

---

### Fix 11: Reasoning Consistency Validation

**Implementation:**
- Add post-generation check in pipeline:
  - For each ticker: if score > 0.1 and reasoning contains bearish keywords → flag
  - If score < -0.1 and reasoning contains bullish keywords → flag
- Bearish keywords: "negative", "bearish", "headwind", "pressure", "decline", "hurt", "risk"
- Bullish keywords: "positive", "bullish", "benefit", "tailwind", "growth", "boost", "advantage"
- Exception patterns: "despite [bearish_word]" or "offset by" should not trigger
- Add `reasoning_consistency` field: "pass" or "fail" with details
- Failed examples: retry once with explicit instruction to align reasoning with score

---

### Fix 12: Market Regime Tagging

**Implementation:**
- Add to `src/market_context.py`:
```python
def get_market_regime(date, spy_returns, basket_returns):
    spy_30d = spy_returns.loc[:date].tail(30).sum()
    basket_vol = basket_returns.loc[:date].tail(30).std() * (252**0.5)
    
    if spy_30d > 0.05:
        regime = "bull"
    elif spy_30d < -0.05:
        regime = "bear"
    else:
        regime = "neutral"
    
    if basket_vol > 0.8:
        regime += "_high_vol"
    
    return regime
```
- Add regime tag to each training example and eval prediction
- Enables future analysis: "IC in bull markets vs bear markets"

---

### Fix 13 (New): Rebalance Training Data for ArXiv

**Implementation:**
- Use the Manus teacher pipeline to generate 70 new arXiv training examples:
  - 20 genuinely important papers (scores 0.3-0.5): e.g., "IonQ researchers demonstrate 99.9% gate fidelity", "IBM team achieves below-threshold error correction"
  - 30 incremental papers (scores near 0.0): e.g., "Theoretical analysis of noise in quantum circuits", "Improved classical simulation of shallow circuits"
  - 20 papers unrelated to commercial quantum (scores exactly 0.0): e.g., "Quantum gravity entanglement entropy", "Topological phases in condensed matter"
- Target training distribution: ~60% news, ~30% arXiv, ~10% other (synthetic/edge cases)
- Use the revised system prompt (with arxiv cap rules from Fix 2) for generation
- Add these to `data/training/manus_arxiv_rebalance.jsonl`

---

## Part 2: Adversarial Analysis (v2)

### Fix 1: Remove HON, Add QNT — ACCEPTED WITH CAVEATS

**Remaining concerns:**

1. **QNT has only 5 trading days of history.** You cannot compute meaningful 5d/30d returns, 52-week position, or beta for QNT. The market context block (Fix 3) will have empty/NA values for QNT for months. The model will see QNT with no context while other tickers have rich context — this asymmetry could confuse the student.

2. **No QNT training data exists.** The entire training set (1,000 examples) was generated with the old universe. None of the examples score QNT. You'll need to regenerate or augment training data to include QNT examples. Fix 13 partially addresses this for arXiv, but you also need QNT examples in real articles, synthetic scenarios, and edge cases.

3. **Date-conditional logic adds complexity.** "Score HON before June 4, score QNT after" means the model needs to understand temporal context about corporate actions. This is learnable but requires explicit training examples showing the transition.

**Severity:** Medium. The fix is directionally correct but incomplete without QNT training data.

**Recommendation:** Add a "Fix 14" that generates 30-50 QNT-specific training examples (post-IPO scenarios, Quantinuum news scored to QNT instead of HON).

---

### Fix 2 (Revised): Conditional ArXiv Cap — IMPROVED, MINOR CONCERN

**Remaining concerns:**

1. **The exception is hard to validate programmatically.** "Authored by researchers at a company AND demonstrates concrete hardware results" requires semantic understanding. The post-processing can't reliably detect this — it would need to parse author affiliations and distinguish theory from hardware results. In practice, you'll likely just clip to [-1.0, 1.0] for all arxiv and rely on the prompt instruction.

2. **The 0.5 cap may still be too generous for most papers.** The eval data shows arxiv IC=-0.0176 — essentially zero predictive power. Even 0.5 might be too high for the vast majority of papers. Consider whether 0.3 would be more appropriate as the default cap.

**Severity:** Low. The conditional exception is reasonable but unenforceable in post-processing. The prompt instruction alone should suffice.

---

### Fix 3 (Revised): Market Context at Training AND Inference — MUCH IMPROVED

**Remaining concerns:**

1. **Parquet data ends at 2026-06-05.** At inference time, if the user submits an article dated after June 5, 2026, the market context will be stale or unavailable. Need a fallback: "Market context unavailable for this date" or fetch live data from Yahoo Finance.

2. **Market context increases prompt length significantly.** A 10-ticker table with 5d/30d returns, 52-week position, and liquidity adds ~200-300 tokens. For a 4096 max_seq_length model, this reduces the space available for the article text. Not critical but worth noting.

3. **Training data regeneration required.** The existing 1,000 training examples were generated WITHOUT market context. If you add market context to inference but not to the existing training data, there's still a distribution shift. You'd need to either:
   - Regenerate all training data with market context (expensive, ~12 hours)
   - Only add market context to NEW training examples (Fix 13's arXiv examples)
   - Accept the mild distribution shift for existing examples

**Severity:** Low-Medium. The approach is correct. The main risk is the training data was generated without market context, creating a mild mismatch. For the fine-tuned model, the system prompt is the same regardless — the market context is in the user message. The model should generalize.

**Recommendation:** For the existing training data, you could retroactively ADD market context to the user messages in the JSONL files (since you have the dates and parquet data). This is a post-processing step, not a regeneration.

---

### Fix 4 (Revised): Metadata Only — NO CONCERNS

This is the safest possible version. Adding metadata for analysis without affecting training is purely beneficial.

**Severity:** None. Clean improvement.

---

### Fix 5: Remove signal_decay — NO CONCERNS

Same as v1. Clean improvement.

**Severity:** None.

---

### Fix 6: QTUM Benchmark — MINOR CONCERN

**Remaining concerns:**

1. **QTUM constituent weights may not match your universe.** QTUM holds ~70 stocks including many non-quantum companies (semiconductor equipment, etc.). It's a better proxy than your 3-stock basket, but it's not a perfect sector factor. The residual alpha may be contaminated by non-quantum factors in QTUM.

2. **QTUM inception date:** Verify QTUM has sufficient history covering your eval period (Dec 2024 - June 2026). QTUM launched in 2018, so this should be fine.

**Severity:** Low. Strictly better than the current approach.

---

### Fix 7: Liquidity Metadata — NO CONCERNS

**One note:** QNT's liquidity is inflated by IPO-week volume. After the first few weeks, it will likely settle to a lower level. Mark QNT's liquidity as "estimated" or "IPO_week" to flag this uncertainty.

**Severity:** None.

---

### Fix 8: Semantic Clustering — MODERATE CONCERN

**Remaining concerns:**

1. **The "reduce by 50% per prior article" rule is too aggressive.** If an article is the 3rd about an event, the rule says reduce by 75% (0.5^2). But the 3rd article might contain genuinely new information (e.g., first article: "Company announces breakthrough", second: "Analysts react", third: "Company provides technical details"). Each adds new signal.

2. **Cosine similarity of 0.75 on general embeddings may over-cluster.** Two articles about different quantum companies could cluster together if they both mention "quantum computing breakthrough" — even if one is about IonQ and the other about Rigetti. Financial news has high lexical overlap within a sector.

3. **The 3-day window may miss slow-developing stories.** A government funding announcement might be covered over 2 weeks as details emerge. A 3-day window would treat week-2 coverage as "new" when it's really follow-up.

**Severity:** Medium. The concept is sound but the parameters need tuning. Recommend implementing with configurable thresholds and validating on 30+ examples before applying to training.

---

### Fix 9: Full-Text Extraction — STILL PROBLEMATIC

**Remaining concerns (unchanged from v1):**

1. **Google News redirect URLs.** Checked the data: article URLs are `https://news.google.com/rss/articles/CBMi...` format. Trafilatura may or may not resolve these — it depends on whether Google's redirect is followed. Success rate is unpredictable.

2. **Stale content.** Articles from 2024 may be paywalled, moved, or deleted. Expect significant failure rate.

3. **Train/inference asymmetry.** At inference time, will the user provide full text or just a summary? If the model is trained on full text but receives summaries at inference, performance degrades. If it receives full text at inference, why does the user need the model? They already have the article.

4. **The `text_quality` flag creates a bimodal distribution.** Some training examples have 500-word full text, others have 50-word summaries. The model may learn to rely on details only present in full-text examples.

**Severity:** Medium. Not harmful if implemented as optional enrichment, but the effort-to-benefit ratio remains poor. The model should be robust to summary-only input since that's the realistic inference scenario.

**Recommendation:** Implement but don't block on it. If trafilatura gets <30% success rate, skip this fix.

---

### Fix 10: Event Deduplication — NO CONCERNS (Eval Only)

Correctly scoped to evaluation only. No risk to training.

**Severity:** None.

---

### Fix 11: Reasoning Consistency — MINOR CONCERN

**Remaining concerns:**

1. **Exception patterns need care.** "Despite bearish sentiment, the long-term outlook remains positive" should NOT flag a bullish score. The exception list needs to handle negation and contrast patterns:
   - "despite [bearish_word]" → don't flag
   - "offset by [bearish_word]" → don't flag
   - "although [bearish_word]" → don't flag
   - "however" followed by opposite sentiment → use the sentiment after "however"

2. **Magnitude matters.** A score of +0.15 with reasoning mentioning "slight headwind but overall positive" is fine. Only flag when the contradiction is stark (e.g., score +1.5 with reasoning saying "significant bearish pressure").

**Severity:** Low. The check is useful but should have a magnitude threshold (e.g., only flag if |score| > 0.3 AND contradictory keywords found).

---

### Fix 12: Market Regime Tagging — NO CONCERNS

Same as v1. Purely additive metadata.

**Severity:** None.

---

### Fix 13 (New): ArXiv Rebalancing — IMPORTANT, WELL-DESIGNED

**Remaining concerns:**

1. **70 examples may not be enough.** The eval set has 159 arXiv articles. With 70 training examples, the model sees each "type" of arXiv paper (important/incremental/unrelated) only 20-30 times. This might be sufficient for learning the score cap but may not be enough for nuanced differentiation.

2. **The distribution of 20/30/20 is a design choice.** In reality, >90% of arXiv papers are incremental or unrelated. A more realistic distribution might be 5/50/15 (important/incremental/unrelated). Training with 20 "important" papers out of 70 (29%) may teach the model that important papers are more common than they actually are.

3. **Timing matters.** The arXiv training examples should span the same date range as the existing training data (Aug 2024 - Dec 2025) to maintain temporal consistency.

**Severity:** Low-Medium. The fix addresses the most critical gap (train/eval distribution mismatch). The exact proportions can be tuned.

**Recommendation:** Consider 10/45/15 split instead of 20/30/20 to better reflect real-world base rates. Most papers are incremental.

---

## Summary: Risk Assessment v2

| Fix | Benefit | Risk | Verdict |
|-----|---------|------|---------|
| 1 (ticker universe) | High | Medium (needs QNT training data) | Implement + add QNT examples |
| 2 (arxiv cap, conditional) | Medium | Low | Implement |
| 3 (market context, both) | High | Low-Medium (existing data mismatch) | Implement + retroactively add context to existing data |
| 4 (metadata only) | Low | None | Implement |
| 5 (remove decay) | Low | None | Implement |
| 6 (QTUM) | Medium | Low | Implement |
| 7 (liquidity) | Low | None | Implement |
| 8 (clustering) | Medium | Medium (parameter sensitivity) | Implement with validation |
| 9 (full-text) | Low | Medium (unreliable URLs) | Implement as best-effort |
| 10 (dedup, eval only) | Medium | None | Implement |
| 11 (reasoning consistency) | Low | Low | Implement with magnitude threshold |
| 12 (regime) | Low | None | Implement |
| 13 (arxiv rebalance) | High | Low | Implement (consider 10/45/15 split) |

---

## Remaining Gaps Not Addressed

1. **No QNT-specific training examples.** Fix 1 adds QNT to the universe but no fix generates QNT training data. Need 30-50 examples of Quantinuum/QNT news scored appropriately.

2. **Existing training data lacks market context.** Fix 3 adds context to inference and new training, but the 1,000 existing examples don't have it. Consider retroactively enriching them.

3. **The model's poor IONQ performance (IC=0.0437) is not addressed.** IONQ is the most-covered ticker but has weak predictive power. This might indicate the model over-scores IONQ news (assigns high magnitude when the market doesn't react proportionally).

4. **No fix addresses the "accuracy improves on large moves" finding.** The model is better at predicting 20%+ moves but random on <2% moves. This suggests the model should output 0.0 more often (when the expected move is small) rather than always assigning non-zero scores. A "minimum conviction threshold" in the prompt could help.
