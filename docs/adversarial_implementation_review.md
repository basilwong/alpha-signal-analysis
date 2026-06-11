# Adversarial Implementation Review: Failure Modes, Checks, and Rollback

This review stress-tests the implementation plan for things that could go wrong during execution, identifies verification checks for each step, and provides rollback procedures.

---

## CRITICAL FINDING: 53% of Training Data Has Broken chain_of_thought

**This was not in the original fix list but is the single biggest quality issue in the dataset.**

The Manus API's structured output extraction produced placeholder `chain_of_thought` values for 411 out of 776 successful training examples (53%):

| Category | Placeholder Rate |
|----------|-----------------|
| Synthetic | 68% |
| Negatives | 62% |
| Real Articles | 51% |
| Paraphrased | 46% |
| Edge Cases | 26% |

The placeholders are: "Not disclosed", "REDACTED", "[redacted]", "" (empty).

**Why this matters:** The original spec says "The `chain_of_thought` field is critical. Research shows training on reasoning traces significantly improves student model quality." If 53% of examples have no reasoning trace, the student model learns that "REDACTED" is an acceptable chain_of_thought — which defeats the purpose.

**Good news:** The `signal_rationale` field is intact in ALL examples (307 have bad cot but good rationale, 0 have both bad). The rationale contains the actual reasoning.

**Recommended fix (add to implementation):**
- For examples with placeholder `chain_of_thought`: copy `signal_rationale` into `chain_of_thought` (it's the same content, just in a different field)
- Or: concatenate `signal_rationale` + per-ticker reasonings into a synthetic chain_of_thought
- Or: remove `chain_of_thought` from the schema entirely and rely on `signal_rationale` + per-ticker `reasoning` fields

**Verification check:** After post-processing, assert that 0 examples have chain_of_thought < 50 chars.

---

## Per-Fix Implementation Risks and Checks

### Fix 1: Update Ticker Universe

**What could go wrong:**
1. **Schema mismatch between old and new data.** Old examples have 9 tickers, new examples will have 10. If the fine-tuning conversion doesn't handle this consistently, the model sees different output structures.
2. **Forgetting to update a file.** 6 files reference the ticker list. Missing one creates inconsistency.
3. **QNT in pre-June-2026 examples.** If QNT is added with score=0.0 and reasoning "not yet public", the model might learn that QNT always scores 0.0.

**Checks:**
- [ ] After post-processing: verify ALL examples in combined JSONL have exactly 10 tickers in signal_vector
- [ ] Verify MSFT/GOOGL/NVDA scores are exactly 0.0 in all examples
- [ ] Verify QNT score is 0.0 in pre-June-2026 examples, non-zero in Fix 15 examples
- [ ] Run `grep -r "QUANTUM_TICKERS\|QUANTUM_UNIVERSE" --include="*.py"` and verify all instances updated

**Rollback:** Keep original `manus_teacher_combined.jsonl` as `manus_teacher_combined_v1_backup.jsonl` before any post-processing.

---

### Fix 2: Conditional ArXiv Cap

**What could go wrong:**
1. **Post-processing clip applied to non-arxiv articles.** If the source field is missing or inconsistent, the clip might be applied incorrectly.
2. **The conditional exception (company-authored, hardware results) is unenforceable in post-processing.** The clip will be [-1.0, 1.0] for all arxiv regardless of authorship.

**Checks:**
- [ ] Verify: no arxiv-sourced example has |score| > 1.0 after processing
- [ ] Verify: non-arxiv examples are NOT clipped (their scores remain unchanged)
- [ ] Count how many arxiv examples were actually clipped (should be very few given the 0.5 prompt rule)

**Rollback:** The clip is applied during post-processing. Keep backup of pre-clip data.

---

### Fix 3 + 3a: Market Context

**What could go wrong:**
1. **Date parsing errors.** Article dates might be in different formats ("2024-08-02" vs "2024/08/02" vs "August 2, 2024").
2. **Market data gaps.** Weekends/holidays have no data. If an article is dated on a Saturday, the lookup needs to find the previous Friday.
3. **Parquet read errors.** The index is DatetimeIndex but article dates are strings. Need proper conversion.
4. **Context for synthetic/edge case examples.** These don't have dates. The function must return "" gracefully.
5. **Inference fallback when parquet data is stale.** After June 5, 2026, the context will be outdated.

**Checks:**
- [ ] Unit test: `get_market_context("2024-08-02", ["IONQ"])` returns non-empty string
- [ ] Unit test: `get_market_context("2024-08-03", ["IONQ"])` (Saturday) returns Friday's data
- [ ] Unit test: `get_market_context("2020-01-01", ["IONQ"])` returns "" (before data starts)
- [ ] Unit test: `get_market_context("2026-12-01", ["IONQ"])` returns "" or stale warning
- [ ] Verify: exactly 190 real article examples get market context added
- [ ] Verify: synthetic/edge case/negative examples have empty market_context field

**Rollback:** Market context is additive (new field). If it causes issues, simply don't prepend it to the user message during fine-tuning conversion.

---

### Fix 3a: Retroactive Enrichment

**What could go wrong:**
1. **Overwriting the file incorrectly.** If the script crashes mid-write, the JSONL file could be corrupted (partial write).
2. **Memory issues.** Loading all 1,000 examples + computing context for each.

**Checks:**
- [ ] Write to a NEW file (`manus_teacher_combined_enriched.jsonl`), then rename. Never overwrite in-place.
- [ ] After enrichment: verify line count matches original (1,000 lines)
- [ ] Verify: `json.loads()` succeeds on every line of the output file
- [ ] Spot-check 5 random examples: market context values match manual calculation

**Rollback:** Original file preserved. Delete enriched file and revert.

---

### Fix 4: Teacher Market Accuracy

**What could go wrong:**
1. **Forward return computation on the last few days of data.** If an article is dated 2025-12-20 and market data ends 2025-12-23, there's no 5-day forward return available.
2. **Division by zero.** If all scores are 0.0 (negative examples), accuracy is undefined.

**Checks:**
- [ ] Skip examples where 5-day forward return is unavailable (article date + 5 > market data end)
- [ ] For negative examples (all scores 0.0): set accuracy to None/null, not 0.0 or 1.0
- [ ] Verify: accuracy values are between 0.0 and 1.0 for all computed examples
- [ ] Print distribution: mean, median, min, max

**Rollback:** This is metadata-only. If wrong, just delete the field.

---

### Fix 5: Remove signal_decay

**What could go wrong:**
1. **Fine-tuning conversion still includes signal_decay.** If the conversion script reads from the raw JSONL and doesn't strip it, the model still learns to produce it.
2. **Evaluation scripts expect signal_decay.** If eval code tries to read this field, it'll KeyError.

**Checks:**
- [ ] After post-processing: verify NO example has `signal_decay` key in its signal dict
- [ ] Search all eval scripts for "signal_decay" references and remove/guard them
- [ ] Verify fine-tuning output format: parse the assistant message JSON and confirm no signal_decay

**Rollback:** Re-add the field from backup if needed. Low risk.

---

### Fix 6: QTUM Benchmark

**What could go wrong:**
1. **yfinance download fails or returns incomplete data.** API rate limits, network issues.
2. **QTUM has different trading hours or missing days compared to individual stocks.** Alignment issues in the two-factor model.
3. **The new factor model produces worse results.** QTUM might be a worse factor than the 3-stock basket for this specific universe.

**Checks:**
- [ ] Verify QTUM data covers full eval period (2024-12-09 to 2026-05-22)
- [ ] Compare R² of old model (SPY + basket) vs new model (SPY + QTUM) — new should be higher
- [ ] Recompute IC with new abnormal returns — should not decrease significantly
- [ ] If IC decreases >20%, revert to old basket

**Rollback:** Keep old `abnormal_returns.csv` as backup. Revert if new model is worse.

---

### Fix 7: Liquidity Metadata

**What could go wrong:**
1. **Stale numbers.** Volume changes over time. Numbers from today may not reflect 2024 conditions.
2. **QNT liquidity is inflated by IPO week.**

**Checks:**
- [ ] Verify numbers against Yahoo Finance (spot check 3 tickers)
- [ ] Add `as_of_date` field to the config so it's clear when numbers were last updated
- [ ] Mark QNT as "estimated_ipo_week" in the tier

**Rollback:** Config change only. Trivial to revert.

---

### Fix 8: Semantic Clustering

**What could go wrong:**
1. **sentence-transformers installation fails or model download times out.** The model is ~90MB.
2. **Over-clustering:** "IonQ breakthrough" and "Rigetti breakthrough" cluster together despite being different events with opposite signals.
3. **Under-clustering:** Same event reported by Reuters and Bloomberg doesn't cluster because of different writing styles.
4. **Memory issues:** Computing pairwise cosine similarity for 620 articles × 384-dim embeddings.
5. **The 30% score reduction rule is applied before validation, corrupting training data.**

**Checks:**
- [ ] **VALIDATION GATE:** After clustering, print 30 random multi-article clusters. Manually verify ≥80% are genuinely the same event.
- [ ] Secondary check: within each cluster, verify at least one overlapping ticker mention
- [ ] Memory: 620 × 384 × 4 bytes = ~1MB. No issue.
- [ ] **DO NOT apply score reduction to existing training data.** Only use for NEW data generation and eval deduplication.

**Rollback:** Clustering is a separate enrichment step. If validation fails, don't apply it. Other fixes don't depend on it.

---

### Fix 9: Full-Text Extraction

**What could go wrong:**
1. **Google News redirect URLs don't resolve.** trafilatura may not follow the redirect chain.
2. **Rate limiting by news sites.** Rapid requests get blocked.
3. **Extracted text is garbage** (cookie banners, navigation menus, ads).
4. **Copyright/ToS violations.**

**Checks:**
- [ ] Test on first 10 URLs manually before running full batch
- [ ] **ABORT CONDITION:** If <30% success rate after 50 attempts, skip entirely
- [ ] For successful extractions: verify text length > 100 chars and < 10,000 chars
- [ ] Spot-check 5 extractions: does the text match the article title?
- [ ] Add 2-second delay between requests to avoid rate limiting

**Rollback:** This fix is fully optional. No other fix depends on it. Simply don't use the `full_text` field.

---

### Fix 10: Event Deduplication

**What could go wrong:**
1. **Depends on Fix 8.** If clustering is bad, deduplication is bad.
2. **Keeping only the "first" article might keep the worst one.** The first report is often a brief headline; later articles have more detail.

**Checks:**
- [ ] Only run after Fix 8 passes validation gate
- [ ] Compare IC on deduplicated vs full predictions — if dedup IC is LOWER, the clustering is wrong
- [ ] Print summary: "Original: N predictions, M events. Deduplicated: M predictions."
- [ ] Verify no eval predictions are lost (all should map to an event_id)

**Rollback:** Keep original `predictions_manus_teacher.jsonl` unchanged. Deduplicated file is separate.

---

### Fix 11: Reasoning Consistency

**What could go wrong:**
1. **False positives from cross-ticker language.** "Positive for IONQ but creates pressure on RGTI" — the word "pressure" in RGTI's context is correct (bearish), but if it appears in IONQ's reasoning it's a false flag.
2. **The retry burns Manus API credits.** If many examples fail, retries are expensive.

**Checks:**
- [ ] Verify keyword check is scoped to EACH TICKER'S individual reasoning string, not the global rationale
- [ ] Count false positive rate: manually review 10 flagged examples. If >50% are false positives, raise threshold to |score| > 0.5
- [ ] Cap retries: maximum 20 retries total across all examples (not unlimited)

**Rollback:** The consistency check is informational. If it flags too many, just ignore the field.

---

### Fix 12: Market Regime

**What could go wrong:**
1. **SPY data gaps.** Same date alignment issues as Fix 3.
2. **Basket volatility computation with missing data.** If one of IONQ/RGTI/QBTS has no data for a date, the vol estimate is biased.

**Checks:**
- [ ] Use same date-handling logic as Fix 3 (find previous trading day)
- [ ] If fewer than 2 basket tickers have data for a date, return "unknown" regime
- [ ] Verify: regime distribution is roughly 30% bull / 30% bear / 40% neutral (not all one category)

**Rollback:** Metadata-only. Remove field if wrong.

---

### Fix 13: ArXiv Rebalancing

**What could go wrong:**
1. **Manus API rate limits.** 70 tasks at 10 concurrent = ~35 minutes. Should be fine.
2. **The teacher assigns scores > 0.5 to "incremental" papers despite the prompt cap.** The structured output extraction might not respect the prompt instruction.
3. **Quality inconsistency.** New examples use the revised prompt (Fixes 2, 14) while old examples don't. The model sees two different "styles" during fine-tuning.
4. **The 10 "important" papers are unrealistically important.** If scenarios are too dramatic, the teacher generates unrealistic articles.

**Checks:**
- [ ] After generation: verify ALL 45 incremental papers have max |score| ≤ 0.1
- [ ] Verify ALL 15 unrelated papers have ALL scores exactly 0.0
- [ ] Verify the 10 important papers have max |score| between 0.3 and 0.5 (not higher)
- [ ] If any fail: post-process clip to enforce bounds
- [ ] Verify chain_of_thought is NOT placeholder (use signal_rationale as fallback if needed)

**Rollback:** These are new examples in a separate file. Simply don't include them in the combined JSONL.

---

### Fix 14: Minimum Conviction Threshold

**What could go wrong:**
1. **Over-zeroing.** The model learns to output 0.0 too aggressively, missing genuine small signals.
2. **Interaction with IBM/HON caps.** IBM max is ±0.15. If the model interprets "minimum conviction" as "don't score small", it might never score IBM.
3. **Inconsistency with existing data.** Old examples have many small non-zero scores. New examples (Fix 13) will have more zeros. The model sees conflicting patterns.

**Checks:**
- [ ] After fine-tuning with new data: check that IBM and HON still receive non-zero scores on relevant news (run 5 test articles about IBM Quantum / Quantinuum)
- [ ] Compare score distributions before/after: the mean |score| for IBM should not drop to 0.0
- [ ] If IBM/HON over-zero: revise prompt wording to explicitly say "IBM and HON should still be scored when news directly relates to their quantum divisions"

**Rollback:** This is a prompt change. If it causes over-zeroing, remove the paragraph from the system prompt and re-fine-tune.

---

### Fix 15: QNT Training Examples

**What could go wrong:**
1. **The teacher doesn't understand QNT.** Since QNT only IPO'd 7 days ago, the Manus model may have limited knowledge about it.
2. **Competitive dynamics are too simplistic.** "IONQ wins contract → bearish QNT" ignores the possibility that the contract validates the trapped-ion market (bullish both).
3. **35 examples may not be enough.** The model sees QNT in only 35/1070 examples (~3%). It might not generalize well.
4. **No QNT market data for validation.** Can't compute teacher_market_accuracy for QNT examples.

**Checks:**
- [ ] Verify the teacher's QNT scores are in [-2.0, +2.0] range
- [ ] Verify competitive dynamics: for IONQ-specific positive news, QNT should be slightly negative (not zero, not strongly negative)
- [ ] Verify sector-wide events: IONQ and QNT should have same sign
- [ ] Spot-check 5 examples: does the reasoning correctly explain the IONQ-QNT relationship?
- [ ] If teacher produces poor QNT reasoning: add more explicit instructions about the competitive dynamic

**Rollback:** QNT examples are in a separate file. Don't include if quality is poor.

---

## Global Implementation Safeguards

### Backup Strategy
```bash
# Before ANY post-processing:
cp data/training/manus_teacher_combined.jsonl data/training/manus_teacher_combined_BACKUP.jsonl
cp data/eval/predictions_manus_teacher.jsonl data/eval/predictions_manus_teacher_BACKUP.jsonl
```

### Atomic File Writes
All post-processing scripts should:
1. Write to a TEMP file (`*.tmp`)
2. Validate the temp file (line count, JSON parse, schema check)
3. Only then rename to final filename
4. Never overwrite in-place

### Validation Script (Run After ALL Fixes)
```python
# scripts/validate_all_fixes.py
def validate():
    # 1. Line count: combined JSONL should have 1000 + 70 (arxiv) + 35 (QNT) = 1105 lines
    # 2. Schema: every example has exactly 10 tickers in signal_vector
    # 3. Inactive tickers: MSFT/GOOGL/NVDA score == 0.0 in ALL examples
    # 4. No signal_decay field in any example
    # 5. No placeholder chain_of_thought (all > 50 chars)
    # 6. ArXiv examples: max |score| <= 1.0
    # 7. Market context: present for all 190 real articles
    # 8. QNT: score == 0.0 for pre-June-2026, non-zero in Fix 15 examples
    # 9. Score ranges: pure-play [-2,2], IBM [-0.15,0.15], HON [-0.3,0.3]
    # 10. JSON parseable: every line in every output file
    pass
```

### Git Branch Strategy
```bash
git checkout -b fix/label-quality
# Implement fixes in phases
# After each phase: run validate_all_fixes.py
# Only merge to main after ALL checks pass
```

### Rollback Procedure
If fine-tuning with the new data produces WORSE results than baseline:
1. Check which fix introduced the regression (A/B test by removing one fix at a time)
2. Most likely culprits: Fix 14 (over-zeroing) or Fix 1 (removing tickers that were actually useful)
3. Revert the problematic fix using backup files
4. Re-fine-tune without that fix

---

## Implementation Dependency Graph

```
Fix 7 (liquidity) ─────┐
                        ├──→ Fix 3 (market context) ──→ Fix 3a (retroactive) ──→ Fix 4 (accuracy metadata)
Fix 12 (regime) ────────┘                                                         │
                                                                                   ├──→ Fix 13 (arxiv, 70 examples)
Fix 5 (remove decay) ──→ Post-processing script ──→ Fix 1 (tickers) ─────────────┤
                                                                                   ├──→ Fix 15 (QNT, 35 examples)
Fix 2 (arxiv cap) ──────────────────────────────────────────────────────────────────┘
Fix 14 (min conviction) ────────────────────────────────────────────────────────────┘

Fix 11 (reasoning) ──→ Applied during Fix 13 and Fix 15 generation

Fix 8 (clustering) ──→ Fix 10 (dedup, eval only)

Fix 9 (full-text) ──→ Independent, best-effort

Fix 6 (QTUM) ──→ Independent, eval only
```

**Critical path:** Fix 7 → Fix 3 → Fix 3a → Post-processing → Fix 13/15 generation

---

## Summary: Top 5 Things Most Likely to Go Wrong

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| 1 | Placeholder chain_of_thought not fixed | High (if forgotten) | High (53% of data is low quality) | Add to post-processing: copy signal_rationale → chain_of_thought where placeholder detected |
| 2 | Fix 14 causes over-zeroing of IBM/HON | Medium | Medium (loses signal on 2 tickers) | Test with 5 IBM/HON articles after fine-tuning. Revert prompt if scores collapse to 0 |
| 3 | Fix 8 over-clusters different events | Medium | Medium (incorrect staleness tags) | Validation gate: manual review of 30 clusters before applying |
| 4 | QNT examples have poor quality | Medium | Low (only 35 examples) | Spot-check 5 examples. Exclude if reasoning is poor |
| 5 | Post-processing corrupts JSONL file | Low | High (lose all training data) | Atomic writes + backup before any modification |
