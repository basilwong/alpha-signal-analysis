# Label Quality Fixes: Implementation Proposal & Adversarial Analysis

## Part 1: Implementation Proposal

### Context

The label quality analysis identified an overall IC of 0.055 (p=0.031) at 5-day horizon, with direction accuracy of 53.2%. The following per-ticker ICs were measured from `data/eval/results.json`:

| Ticker | IC | p-value | Status |
|--------|-----|---------|--------|
| HON | +0.1664 | 0.0015 | Best predictor |
| RGTI | +0.1299 | 0.0135 | Strong |
| QBTS | +0.0850 | 0.1070 | Moderate |
| QUBT | +0.0839 | 0.1116 | Moderate |
| IBM | +0.0570 | 0.2797 | Weak |
| IONQ | +0.0437 | 0.4080 | Weak |
| GOOGL | -0.0228 | 0.6665 | Noise |
| MSFT | -0.0334 | 0.5267 | Noise/anti |
| NVDA | -0.1749 | 0.0008 | Strongly anti-predictive |

ArXiv source IC is -0.0176 (p=0.84), essentially random noise.

---

### Fix 1: Update Ticker Universe

**What:** Remove MSFT, GOOGL, NVDA, HON from active scoring. Add QNT (Quantinuum, IPO'd June 4, 2026 on NASDAQ).

**Implementation:**
- Update `src/config.py`: move HON/MSFT/GOOGL/NVDA to `INACTIVE_TICKERS`, add QNT to `PURE_PLAY_TICKERS`
- Update pipeline prompts: hard-code 0.0 for inactive tickers with explanatory reasoning
- Update structured output schema: keep all 9 original tickers + add QNT for backward compatibility
- Active universe: IONQ, RGTI, QBTS, QUBT, QNT, IBM

**Effort:** Low (config + prompt changes)

---

### Fix 2: Cap ArXiv Scores at ±0.5

**What:** Add prompt rule and post-processing clip for arxiv-sourced articles.

**Implementation:**
- Add to system prompt: academic paper cap rule
- Add post-processing in pipeline: `if source == "arxiv": clip scores to [-0.5, 0.5]`
- Apply to both training generation and evaluation prediction pipelines

**Effort:** Low

---

### Fix 3: Market Context in Teacher Prompt

**What:** Provide 5d/30d returns, 52-week position, and liquidity context from parquet files.

**Implementation:**
- Create `src/market_context.py` with function `get_market_context(date, tickers)`
- Read from `data/market/*.parquet` (available 2024-01-02 to 2026-06-05)
- Format as prompt prefix with returns table and liquidity notes
- Add liquidity-awareness instruction to system prompt

**Effort:** Medium (need to handle date alignment, missing data, QNT having no history)

---

### Fix 4: Label Validation Filter

**What:** Compare teacher predictions vs actual 5d forward returns, keep only correct-majority examples.

**Implementation:**
- Load `manus_teacher_combined.jsonl` + market data
- For each real article with a date: compute 5d forward return per ticker, compare direction to predicted score
- Keep examples where teacher was correct on majority of active tickers
- Synthetic/negative examples pass through automatically (no market validation possible)

**Effort:** Medium

---

### Fix 5: Remove signal_decay from Training

**What:** Remove `signal_decay` from the required output schema.

**Implementation:**
- Remove from `SIGNAL_SCHEMA` in pipeline scripts
- Remove from system prompt
- Keep `time_horizon` (useful metadata)
- Add comment in eval pipeline about empirical decay discovery

**Effort:** Low

---

### Fix 6: Add QTUM as Sector Benchmark

**What:** Replace synthetic equal-weighted basket with QTUM ETF in the two-factor model.

**Implementation:**
- Download QTUM price data via yfinance
- Update `eval/market_data.py` to use QTUM as sector factor
- Keep old basket as fallback for early dates
- Update abnormal return computation

**Effort:** Medium

---

### Fix 7: Liquidity Metadata

**What:** Add `LIQUIDITY_TIERS` dict to config.

**Implementation:**
- Add dict with avg daily dollar volume per ticker
- Verify via Yahoo Finance
- Used by Fix 3's market context function

**Effort:** Low

---

### Fix 8: Semantic Clustering for Staleness

**What:** Cluster articles by cosine similarity within 3-day windows, assign `prior_coverage_count`.

**Implementation:**
- Install `sentence-transformers`, use `all-MiniLM-L6-v2`
- Compute embeddings for all articles
- Sliding window clustering with cosine threshold 0.75
- Add `prior_coverage_count` field
- Update teacher prompt with coverage context

**Effort:** Medium-High (model download, embedding computation, clustering logic)

---

### Fix 9: Full-Text Extraction

**What:** Use `trafilatura` to fetch full article text from URLs.

**Implementation:**
- Install `trafilatura`
- For each article with `source == "news"` and a URL, attempt extraction
- Store in `full_text` field, add `text_quality` indicator
- Update pipeline to prefer `full_text` when available

**Effort:** Medium (web scraping is inherently unreliable)

---

### Fix 10: Event Deduplication

**What:** Group related articles by `event_id`, keep only first prediction per event for eval.

**Implementation:**
- Extends Fix 8's clustering to assign `event_id`
- Create `scripts/deduplicate_for_eval.py`
- Output `predictions_deduplicated.jsonl`
- Update eval to use deduplicated by default

**Effort:** Medium (depends on Fix 8)

---

### Fix 11: Reasoning Consistency Validation

**What:** Keyword-based check that score direction matches reasoning sentiment.

**Implementation:**
- After signal generation, check each ticker's reasoning for contradictory keywords
- Flag inconsistent examples for retry
- Add `reasoning_consistency` field to output

**Effort:** Low

---

### Fix 12: Market Regime Tagging

**What:** Tag each example with bull/bear/neutral regime based on SPY 30d return and quantum basket volatility.

**Implementation:**
- Add function to `src/market_context.py`
- Compute SPY 30d return and basket 30d realized vol
- Classify into regimes
- Add to training examples and eval predictions

**Effort:** Low-Medium

---

## Part 2: Adversarial Analysis

### Critical Issue: Fix 1 (Remove HON) — DATA CONTRADICTS THE PREMISE

**Severity: HIGH — This fix would DESTROY the best-performing signal.**

The prompt states HON should be removed because "Quantinuum spun off as QNT on June 4, 2026; quantum news no longer moves HON." However, the actual evaluation data tells a completely different story:

- **HON has the highest IC of any ticker: +0.1664 (p=0.0015)** — the most statistically significant predictor in the entire model
- HON's non-zero predictions have IC=0.2065 with 61.5% direction accuracy
- HON is the only ticker where the model's signal is both economically and statistically significant

**The Quantinuum spinoff happened on June 4, 2026** — only 7 days ago. The evaluation data spans Dec 2024 to May 2026, meaning **99% of the eval period is pre-spinoff when HON absolutely should be scored.** Removing HON from training data retroactively poisons the model's ability to score HON during the period when it was most predictive.

**Recommendation:** Do NOT remove HON from training. Instead:
- Keep HON active for articles dated before June 4, 2026
- Add QNT for articles dated after June 4, 2026
- Add a date-conditional rule: "After June 4, 2026, score QNT instead of HON for Quantinuum news"

---

### Critical Issue: Fix 1 (Remove GOOGL/MSFT) — Low Risk but Wasteful

GOOGL (IC=-0.0228) and MSFT (IC=-0.0334) are essentially noise — not statistically significant. Removing them is fine from a "do no harm" perspective, but they're already capped at ±0.05 in the current pipeline. The model already outputs near-zero for these. Removing them adds complexity (schema changes, backward compatibility) for minimal gain.

**Risk:** Low. But the effort-to-benefit ratio is poor.

---

### Critical Issue: Fix 1 (Remove NVDA) — Correct Direction, Wrong Solution

NVDA is genuinely anti-predictive (IC=-0.1749, p=0.0008). The model consistently gets NVDA wrong. However, simply removing it doesn't fix the underlying problem — the model may still learn incorrect NVDA-related reasoning that bleeds into other tickers.

**Better alternative:** Instead of removing NVDA, investigate WHY it's anti-predictive. Possible causes:
1. NVDA moves on AI news, not quantum news — quantum signals are noise for NVDA
2. The model assigns positive NVDA scores to quantum breakthroughs, but NVDA actually drops when quantum advances (because quantum threatens GPU simulation revenue)

**Recommendation:** Remove NVDA from active scoring (correct), but also add explicit reasoning: "NVDA moves primarily on AI/GPU demand, not quantum computing news. Quantum breakthroughs may actually be slightly bearish for NVDA as they reduce demand for quantum simulation on GPUs."

---

### Moderate Issue: Fix 2 (Cap ArXiv at ±0.5) — Correct but Insufficient

The eval data shows arxiv IC=-0.0176 (essentially zero). Capping at ±0.5 helps, but the real problem is that **only 1 out of 194 training articles is from arxiv**, while **159 out of 426 eval articles are arxiv** (37%). The model was barely trained on arxiv content but is heavily evaluated on it.

**Risk:** The cap helps prevent extreme scores, but doesn't address the fundamental train/eval distribution mismatch. The model needs MORE arxiv training examples, not just capped scores.

**Recommendation:** In addition to the cap, regenerate training data with more arxiv articles (the `collect_historical_articles.py` script already collects them — they were just filtered out of training by the temporal split).

---

### Moderate Issue: Fix 3 (Market Context) — Potential Look-Ahead Bias

Adding market context (5d/30d returns) to the teacher prompt is powerful but dangerous:

1. **For training data generation (retrospective):** The teacher can see what happened to the stock before AND after the article date. If the teacher sees "IONQ is up 40% in the past month" and the article is bullish, it might assign a lower score because it knows the move already happened. This is CORRECT behavior for a real-time system.

2. **For evaluation predictions:** The prompt already forbids looking up future information. But providing past returns is fine.

3. **For the fine-tuned student model at inference time:** Will the student model HAVE access to market context? If not, training with market context creates a feature the student can't use, potentially hurting performance.

**Risk:** Medium. If the student model won't have real-time market data at inference, training the teacher with it creates a distribution shift.

**Recommendation:** Only add market context if the inference pipeline will also provide it. Otherwise, the student learns to rely on information it won't have.

---

### Moderate Issue: Fix 4 (Label Validation Filter) — Survivorship Bias

Filtering training examples to keep only those where the teacher was "correct" introduces survivorship bias:

1. **The market is noisy.** A correct signal can still see the stock move against it in any 5-day window due to unrelated factors.
2. **You're training the model to predict what happened, not what should happen.** This is subtly different from training it to produce good signals.
3. **You lose edge cases.** The most interesting training examples are often ones where the signal was correct but the market moved against it (e.g., due to a macro shock). These teach the model about signal decay and noise.
4. **Sample size reduction.** If you reject 30-40% of examples, you're back to ~600 training examples — potentially below the threshold for effective fine-tuning.

**Risk:** Medium-High. Could reduce training set size significantly and introduce bias toward "easy" predictions.

**Recommendation:** Instead of hard filtering, add a `label_confidence` field (0-1) based on whether the market agreed. Use this for weighted training loss rather than binary inclusion/exclusion.

---

### Low Issue: Fix 5 (Remove signal_decay) — Correct, No Concerns

The analysis clearly shows signal_decay doesn't correlate with empirical decay. Removing it simplifies the output and prevents the model from wasting capacity on a useless field.

**Risk:** None. Clean improvement.

---

### Low Issue: Fix 6 (QTUM Benchmark) — Correct, Minor Implementation Risk

Using QTUM as sector factor is better than the synthetic basket. The only risk is that QTUM may have different constituent weights than our universe, potentially introducing tracking error.

**Risk:** Low. Strictly better than the current approach.

---

### Low Issue: Fix 7 (Liquidity Metadata) — Correct, No Concerns

Adding liquidity tiers is purely informational and helps the model make better decisions.

**Risk:** None.

---

### Moderate Issue: Fix 8 (Semantic Clustering) — Threshold Sensitivity

The 0.75 cosine similarity threshold is arbitrary. Too low → unrelated articles clustered together. Too high → related articles missed.

**Risks:**
1. `all-MiniLM-L6-v2` may not capture financial news similarity well (it's a general-purpose model)
2. The 3-day window may be too short for slow-developing stories or too long for fast-moving markets
3. The "reduce by 50% per prior article" rule is a blunt instrument — some follow-up articles contain genuinely new information

**Recommendation:** Implement but validate the clustering quality manually on 20-30 examples before using it to modify scores. Consider using a financial-domain embedding model instead.

---

### High Issue: Fix 9 (Full-Text Extraction) — Unreliable and Potentially Harmful

**Risks:**
1. **Most URLs are Google News redirect URLs** (the articles in `articles_train.jsonl` have URLs like `https://news.google.com/rss/articles/CBMi...`). These are not direct article URLs — trafilatura will likely fail on most of them.
2. **Stale URLs:** Articles from 2024 may have moved, been paywalled, or deleted. Expect <30% success rate.
3. **Inconsistent training data:** If only some articles have full text, the model sees a bimodal distribution — some with rich context, some with just summaries. This can confuse fine-tuning.
4. **Copyright concerns:** Scraping and storing full article text may violate terms of service.

**Recommendation:** This fix has the worst effort-to-reliability ratio. Skip it or make it optional. The model should learn to work with summaries since that's what it'll get at inference time (RSS feeds provide summaries, not full text).

---

### Low Issue: Fix 10 (Event Deduplication) — Correct for Eval, Risky for Training

For evaluation: deduplication is correct. Multiple predictions for the same event inflate the sample size artificially.

For training: do NOT deduplicate training data. Multiple articles about the same event with different framing are valuable training examples (they teach the model that the same event can be described differently).

**Risk:** Low if applied only to evaluation. Medium if accidentally applied to training.

---

### Low Issue: Fix 11 (Reasoning Consistency) — Correct but Limited

Keyword-based sentiment detection is crude. "The stock faces headwinds but the long-term outlook is positive" would trigger a false positive for a bullish score. However, it's fast, free, and catches obvious errors.

**Risk:** Low. May flag some false positives, but the retry mechanism handles that.

**Recommendation:** Use a small allowlist of exception patterns (e.g., "despite headwinds" followed by "overall bullish" should not flag).

---

### Low Issue: Fix 12 (Market Regime Tagging) — Correct, No Concerns

Tagging regime is purely additive metadata. It enables future analysis without changing current behavior.

**Risk:** None.

---

## Summary: Recommended Implementation Order

### Implement As-Is (Low Risk):
- Fix 5 (Remove signal_decay) ✅
- Fix 7 (Liquidity metadata) ✅
- Fix 11 (Reasoning consistency) ✅
- Fix 12 (Market regime tagging) ✅
- Fix 6 (QTUM benchmark) ✅
- Fix 2 (Cap arxiv at ±0.5) ✅

### Implement with Modifications:
- **Fix 1:** Keep HON active for pre-June 2026 data. Add QNT for post-June 2026. Remove MSFT/GOOGL/NVDA from active scoring.
- **Fix 3:** Only add market context if the inference pipeline will also provide it. Otherwise skip.
- **Fix 4:** Use weighted confidence instead of hard filtering. Or filter but set a minimum retained count (e.g., keep at least 800 examples).
- **Fix 8:** Implement but validate clustering quality before using it to modify scores.
- **Fix 10:** Apply only to evaluation, never to training.

### Skip or Defer:
- **Fix 9 (Full-text extraction):** Most URLs are Google News redirects. Low success rate expected. The model should learn from summaries since that's what inference provides.

---

## Risk Matrix

| Fix | Benefit | Risk of Harm | Confidence |
|-----|---------|-------------|------------|
| 1 (ticker universe) | High | **HIGH if HON removed** | Modify |
| 2 (arxiv cap) | Medium | Low | Implement |
| 3 (market context) | High | Medium (distribution shift) | Conditional |
| 4 (label filter) | Medium | Medium-High (survivorship bias) | Modify |
| 5 (remove decay) | Low | None | Implement |
| 6 (QTUM) | Medium | Low | Implement |
| 7 (liquidity) | Low | None | Implement |
| 8 (clustering) | Medium | Medium (threshold sensitivity) | Validate first |
| 9 (full-text) | Low | Medium (unreliable) | Skip |
| 10 (dedup) | Medium | Low (eval only) | Implement |
| 11 (reasoning) | Low | Low | Implement |
| 12 (regime) | Low | None | Implement |
