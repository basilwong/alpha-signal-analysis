# 6-Way Evaluation Diagnostic Report

## Summary of Findings

The 6-way model comparison produced counterintuitive results where the smaller 8B model outperformed the larger 14B model at the 5-day prediction horizon. This report documents the full investigation, identifies the root causes, and provides actionable recommendations.

| Config | IC @5d | Dir Acc @5d |
|--------|--------|-------------|
| 1. 8B Base (no memory) | +0.047 | 53.8% |
| **2. 8B + Memory** | **+0.107** | **56.8%** |
| 3. 14B Base (no memory) | +0.006 | 49.9% |
| 4. 14B + Memory | -0.007 | 53.2% |
| 5. 14B Fine-tuned | +0.010 | 51.9% |
| 6. 14B FT + Memory | -0.004 | 52.9% |

## 1. Is This a Bug?

**No.** The evaluation pipeline is functioning correctly. We verified:
- The `signal_vector_clean` fields are correctly formatted across all configs.
- Dates align properly with market data.
- The forward return calculation is accurate.
- The IC and direction accuracy math is correct.

We also ruled out **mode collapse** via a temperature experiment. Running the 14B model at temperatures 0.3, 0.5, 0.7, and 1.0 produced nearly identical score distributions (mean |score| of 0.77 to 0.79, % above 0.3 threshold of 72% to 74% across all settings). Higher temperature made things slightly worse, not better. The 14B model genuinely lacks predictive power at the 5-day horizon.

## 2. Root Cause: The "Rising Tide" Fallacy

The primary reason the 14B model underperforms is a systematic misinterpretation of competitive dynamics in the quantum computing sector. We ran a case-by-case comparison of 20 articles through both models and found:

**When the two models disagreed on direction (40 instances), the pattern was nearly always the same: the 8B model predicted bearish while the 14B model predicted bullish.** In these disagreements, the 8B model was correct 76% of the time.

The 14B model operates under a "rising tide lifts all boats" heuristic. When it reads about a breakthrough by Google, Microsoft, or Quantinuum, it assigns bullish scores to all pure-play quantum stocks, reasoning that the news validates the sector. The 8B model correctly interprets these events as competitive threats, assigning bearish scores to the smaller companies that now face a stronger competitor.

### Illustrative Case: Quantinuum Helios (Nov 5, 2025)

The article announced Quantinuum's Helios as "the most accurate quantum computer in the world."

**14B reasoning:** "This development is likely to benefit companies like IonQ (IONQ)... QBTS and QUBT may also see positive momentum."
- 14B scores: IONQ +1.5, QUBT +1.2, QBTS +0.8
- 14B accuracy: 33%

**8B reasoning:** "IONQ, RGTI, QBTS, QUBT, and QNT are all quantum computing companies that may face increased competition or market uncertainty due to this breakthrough."
- 8B scores: IONQ -1.2, QUBT -0.7, QBTS -0.5
- 8B accuracy: 100%

**Actual 5-day returns:** IONQ -8.5%, RGTI -24.1%, QBTS -14.9%, QUBT -21.5%

This pattern repeated across multiple articles about Microsoft, Google, and other large-company breakthroughs.

## 3. Systematic Bullish Bias in the 14B Model

Across our 20-article diagnostic sample, the 14B model exhibited an extreme bullish bias on pure-play quantum tickers:

| Ticker | 14B Bullish % | 8B Bullish % | Actual % Positive |
|--------|--------------|-------------|-------------------|
| IONQ | 95% | 45% | 45% |
| QBTS | 95% | 45% | 25% |
| QUBT | 95% | 50% | 30% |
| QNT | 95% | 45% | 0% |
| IBM | 90% | 55% | 60% |

The 14B model predicted bullish on IONQ in 19 out of 20 articles. In reality, IONQ only went up in 9 of those 20 periods. The 8B model was much more balanced (9 bullish, 11 bearish), closely matching the actual distribution.

## 4. Reasoning-to-Score Disconnect

A linguistic analysis of the `chain_of_thought` outputs revealed that the 14B model actually uses more cautious and bearish vocabulary than the 8B model (26 bearish terms vs 16 for the 8B). However, it assigns far fewer bearish numerical scores (only 20% of 14B scores are negative vs 49% for 8B).

The most likely explanation is that the 14B model's bearish language functions as hedging within an argument that ultimately concludes bullish. The model writes balanced prose that acknowledges risks, but its final numerical "decision" is anchored to a bullish default. The 8B model does not hedge; when it identifies a threat, it directly translates that into a negative score.

## 5. Why Memory Helps 8B but Hurts 14B

The memory system generates procedural rules based on the model's historical accuracy. Because the two models have different baseline behaviors, they learned opposite rules:

**8B Memory Rules (11 rules):** Predominantly instruct the model to be bullish with high conviction. Examples: "predict bullish for all quantum-related tickers," "predict strongly bullish (>=+1.8) for quantum computing tickers." These rules amplified the 8B model's existing (weak but positive) signal.

**14B Memory Rules (7 rules):** Predominantly instruct the model to be conservative and reduce conviction. Examples: "Be CONSERVATIVE and cap conviction at 0.5 when analyzing news content," "Predictions for IONQ have been only 22% accurate. Reduce conviction." These rules pushed the 14B model's scores below the 0.3 evaluation threshold, reducing the number of qualifying predictions and degrading IC.

The 14B memory rules were generated after the first batch of 50 articles (Jan-Mar 2026), during which quantum stocks were in a severe bear market. Because the 14B model was always bullish during this crash, the feedback loop concluded it was unreliable and generated conservative rules that persisted for the rest of the evaluation.

## 6. Why the 14B Model Behaves This Way

Three factors likely contribute to the 14B model's bullish bias:

**Stronger priors from pre-training.** The 14B model has absorbed more financial media during pre-training. The dominant narrative in financial coverage of emerging technology sectors frames competitor milestones as "sector validation." Sell-side analysts routinely use competitor breakthroughs to reiterate buy ratings on their entire coverage universe. The 14B model has internalized this framing more deeply than the 8B model.

**Alignment toward safe outputs.** RLHF alignment tends to penalize strong negative claims. Predicting that a stock will decline is inherently riskier from an alignment perspective. The 14B model, having undergone more extensive alignment, may be more reluctant to assign high-conviction bearish scores.

**Training data label bias.** If the fine-tuning examples for this task predominantly showed positive scores for quantum stocks (collected during a bull market), the model learned that positive scores are the "correct" output pattern for quantum computing articles, regardless of the specific content.

## 7. Implications and Recommendations

**The 14B model is not broken.** It produces well-formatted outputs, sophisticated reasoning, and reasonable-looking scores. It simply has a systematic bias that happens to be wrong for this specific task at this specific horizon.

**Recommendations:**

1. **Prompt engineering for the 14B model.** Add explicit instructions about competitive dynamics: "When a large company (Google, Microsoft, Quantinuum) announces a breakthrough, this is typically BEARISH for smaller pure-play competitors in the short term, as it signals increased competition and potential market share loss."

2. **Two-step generation.** Force the model to first output a direction (bullish/bearish) for each ticker with a brief justification, then assign magnitude. This may tighten the coupling between reasoning and scores.

3. **Evaluate at longer horizons.** The 14B model's "sector validation" thesis may actually be correct at longer time horizons (30-60 days), where fundamental improvements in quantum technology do lift all boats. Test IC at 20d and 60d horizons.

4. **Cross-sectional evaluation.** Instead of predicting absolute direction, evaluate the model on its ability to rank stocks relative to each other. A model that always predicts bullish but correctly identifies which stock will go up *most* still has alpha.

5. **Fix the memory feedback loop.** The current system generates rules after the first batch and never revises them. Implement rule decay or periodic re-evaluation so that rules generated during a bear market don't permanently suppress the model's conviction.

6. **Use the 8B model for production signals.** Given the current evidence, the 8B model is the better signal generator for 5-day directional predictions in this sector. The 14B model may be better suited for longer-horizon or cross-sectional tasks.
