# Codex Prompt: Generate Quantum Alpha Training Data (V8)

## Repository Setup

```bash
# Clone the repo
git clone https://github.com/basilwong/quantum-alpha-intelligence.git
cd quantum-alpha-intelligence

# Switch to the working branch
git checkout fix/label-quality

# Key file locations:
# Input articles (training): data/raw/articles_train.jsonl
# Input articles (eval):     data/raw/articles_eval.jsonl
# Market data (parquet):     data/market/IONQ.parquet, RGTI.parquet, etc.
# Output training data:      data/training/quantum_alpha_train_v8.jsonl
# Output eval predictions:   data/eval/predictions_codex_teacher.jsonl
# Previous V4 data (reference): data/training/quantum_alpha_train_v4.jsonl
# Config/prompts:            src/config.py, src/prompts.py, src/market_context.py
```

---

## Context

You are generating training data for a fine-tuned reasoning model that produces cross-sectional trading signals for quantum computing stocks. The model will run locally via vLLM with no internet access at inference time.

**Critical learning from previous iterations:**

1. **V4 (no thinking traces, 881 examples) is the best model so far.** IC = +0.075 at 5d. It works because it is SELECTIVE — it only produces non-zero scores when it has genuine conviction.

2. **V5/V7 (with thinking traces, 1121 examples) degraded performance.** IC = -0.010 at 5d. The thinking traces made the model LESS selective — it reasoned its way into making predictions it shouldn't have made. It produced 2x more non-zero predictions (6,688 vs 3,265), diluting the signal.

3. **Bearish examples hurt.** The synthetic bearish scenarios (earnings misses, setbacks) had lower label quality than the real-article examples. Removing them improved performance.

**The core problem to solve:** How do we get the benefits of thinking traces (better calibration, consistency, interpretability) WITHOUT losing selectivity? The answer is: the thinking must frequently conclude "this doesn't warrant a non-zero score" and output zeros.

---

## Your Task

Generate training data where the assistant response includes a `<think>...</think>` block followed by JSON. The critical difference from previous attempts:

**At least 40-50% of examples should have the thinking conclude with mostly zero scores.** The model must learn that "I thought about it carefully and decided there's no signal here" is the most common correct output.

### Output Format

```
assistant: <think>
[100-200 tokens of genuine reasoning]
</think>
{"signal_vector": {"IONQ": {"score": 0.0, "reasoning": "..."}, ...}, ...}
```

### The Three Types of Examples (Target Distribution)

**Type A: Genuine Signal (30-35% of examples)**
- Article has clear, direct implications for specific quantum companies
- Thinking identifies the technology, affected companies, and competitive dynamics
- Produces non-zero scores with conviction (|score| > 0.5 for at least one ticker)
- Examples: major contract wins, hardware breakthroughs, earnings surprises

**Type B: Considered Zero (40-50% of examples)**
- Article is about quantum computing but doesn't warrant trading action
- Thinking explicitly reasons through WHY there's no signal
- Concludes with all or nearly all zeros
- Examples: incremental papers, routine updates, old news, tangentially related content
- THE THINKING MUST SHOW THE REASONING FOR ZERO: "This is incremental, already priced in, too speculative, or affects companies outside our universe"

**Type C: Moderate Signal (20-25% of examples)**
- Article has some implications but conviction is low
- Thinking shows uncertainty and produces small scores (|score| 0.1-0.5)
- Examples: partnerships with unclear value, mixed earnings, ambiguous competitive moves

---

## Ticker Universe

**Active (assign scores):**
- IONQ: IonQ (trapped-ion, pure-play) — range [-2.0, +2.0]
- RGTI: Rigetti Computing (superconducting, pure-play) — range [-2.0, +2.0]
- QBTS: D-Wave Quantum (quantum annealing, pure-play) — range [-2.0, +2.0]
- QUBT: Quantum Computing Inc. (neutral atom, pure-play) — range [-2.0, +2.0]
- QNT: Quantinuum (trapped-ion, pure-play) — range [-2.0, +2.0]
- IBM: International Business Machines (superconducting, ~2% quantum revenue) — range [-0.15, +0.15]
- HON: Honeywell (trapped-ion, ~1% quantum revenue) — range [-0.3, +0.3]

**Inactive (always 0.0):**
- MSFT, GOOGL, NVDA — always 0.0 (but their news affects active tickers)

---

## Scoring Rules

1. **Selectivity is paramount.** Only assign non-zero scores when you have genuine conviction that this news will move the stock over 5 trading days. When in doubt, output 0.0.

2. **Technology validation rule:** When Google/IBM/Microsoft achieves a superconducting breakthrough → BULLISH for RGTI (validates the approach). When Quantinuum achieves a trapped-ion breakthrough → BULLISH for IONQ. Market evidence: RGTI surged +89% when Google announced Willow.

3. **IONQ-QNT competitive dynamic:** Both are trapped-ion pure-plays. Sector-wide events → both move together. Company-specific wins → they diverge (zero-sum).

4. **ArXiv papers:** Almost always 0.0. Only non-zero for company-authored hardware results with measured metrics.

5. **Market context awareness:** If stocks are already up 30%+ in the prior week, bullish news may be priced in. If stocks are down 20%+, consider whether bad news is already reflected.

---

## System Prompt (Use This Exactly)

```
You are a quantitative NLP signal generator for the quantum computing sector. You MUST think step-by-step before producing scores.

OUTPUT FORMAT:
<think>
[Your reasoning, 100-200 tokens]
</think>
{"signal_vector": {...}, "event_type": "...", ...}

CRITICAL: Most news does NOT warrant a trading signal. Your default output should be all zeros. Only assign non-zero scores when you have HIGH CONVICTION that this specific news will move a specific stock's price over the next 5 trading days.

Ask yourself: "Would a portfolio manager actually trade on this?" If the answer is "probably not," output zeros.

[Include full ticker universe, score ranges, and technology dynamics here]
```

---

## User Message Format

```
**Market Context (as of {date}):**
| Ticker | 5d Ret | 30d Ret | 52w Position | Liquidity |
|--------|--------|---------|--------------|-----------|
| IONQ | +X.X% | +X.X% | Position | Tier |
| ... | ... | ... | ... | ... |

**Market regime:** Bull/Bear/Neutral / Volatility

[ARTICLE]
Title: {title}
Date: {date}
Source: {source}

{article_text}
```

---

## Example: Type B (Considered Zero — Most Common)

```
assistant: <think>
This article discusses a minor software update to IonQ's cloud platform that improves job scheduling. This is routine engineering work, not a hardware breakthrough or commercial milestone. No portfolio manager would trade on this. The improvement is incremental and already expected as part of normal platform maintenance. All scores should be 0.0.
</think>
{"signal_vector": {"IONQ": {"score": 0.0, "reasoning": "Routine software update, no trading signal."}, "RGTI": {"score": 0.0, "reasoning": "Unrelated to Rigetti."}, "QBTS": {"score": 0.0, "reasoning": "Unrelated."}, "QUBT": {"score": 0.0, "reasoning": "Unrelated."}, "QNT": {"score": 0.0, "reasoning": "Unrelated."}, "IBM": {"score": 0.0, "reasoning": "Unrelated."}, "HON": {"score": 0.0, "reasoning": "Unrelated."}, "MSFT": {"score": 0.0, "reasoning": "Inactive."}, "GOOGL": {"score": 0.0, "reasoning": "Inactive."}, "NVDA": {"score": 0.0, "reasoning": "Inactive."}}, "event_type": "routine_update", "time_horizon": "intraday", "information_novelty": "low", "technical_translation": "Minor cloud platform improvement with no commercial significance.", "signal_rationale": "No trading signal. Incremental engineering work does not move stock prices."}
```

## Example: Type A (Genuine Signal — Less Common)

```
assistant: <think>
IonQ just won a $200M contract with the US Air Force for quantum computing services. This is their largest contract ever and represents ~4x their current annual revenue. It validates their trapped-ion approach for defense applications and provides multi-year revenue visibility.

Direct impact: IONQ strongly bullish (+1.8). QNT mildly bearish (-0.5) as they likely competed for this contract. RGTI neutral — different technology, different customer segment. IBM slightly positive (+0.1) as it validates enterprise quantum demand broadly.

Conviction: HIGH. This is a material, unexpected contract win with clear revenue implications.
</think>
{"signal_vector": {"IONQ": {"score": 1.8, "reasoning": "Largest contract in company history, 4x annual revenue, validates defense use case."}, "RGTI": {"score": 0.0, "reasoning": "Different technology and customer segment."}, "QBTS": {"score": 0.0, "reasoning": "Annealing approach not relevant to this contract type."}, "QUBT": {"score": 0.0, "reasoning": "Not competing in this space."}, "QNT": {"score": -0.5, "reasoning": "Direct competitor for trapped-ion defense contracts, likely lost this bid."}, "IBM": {"score": 0.1, "reasoning": "Validates enterprise quantum demand broadly."}, "HON": {"score": 0.15, "reasoning": "Trapped-ion validation, but limited direct exposure post-spinoff."}, "MSFT": {"score": 0.0, "reasoning": "Inactive."}, "GOOGL": {"score": 0.0, "reasoning": "Inactive."}, "NVDA": {"score": 0.0, "reasoning": "Inactive."}}, "event_type": "contract_win", "time_horizon": "2-5 days", "information_novelty": "high", "technical_translation": "IonQ secures transformative defense contract that provides multi-year revenue visibility and validates trapped-ion for government applications.", "signal_rationale": "Material contract win with high conviction. IONQ direct beneficiary, QNT loses competitive bid, sector broadly validated."}
```

---

## Implementation Instructions for Codex

1. **Source articles from `data/raw/articles_train.jsonl`** — these are real articles with dates within the training period (before 2026-01-07).

2. **For each article, generate the full training example** (system + user + assistant messages) with the thinking trace format.

3. **Enforce the distribution:** Track how many Type A / B / C examples you've generated. Aim for 40-50% Type B (zeros), 30-35% Type A (genuine signal), 20-25% Type C (moderate).

4. **Market context:** Compute from parquet files in `data/market/` using the article's date. Only use data UP TO the article date (no future leakage).

5. **Validation after generation:**
   - All 10 tickers present
   - MSFT/GOOGL/NVDA = 0.0
   - Score ranges respected
   - Thinking block present and 100-200 tokens
   - JSON parses correctly
   - Total tokens < 4096

6. **Save to `data/training/quantum_alpha_train_v8.jsonl`** in OpenAI messages format.

---

## For Eval Predictions (Teacher Baseline)

In addition to training data, generate predictions for ALL eval articles to benchmark Codex as a teacher model.

### Input

- File: `data/raw/articles_eval.jsonl`
- 426 articles from 2026-01-07 to 2026-06-09
- Each article has: `title`, `date`, `source`, `text`

### Output

- File: `data/eval/predictions_codex_teacher.jsonl`
- One line per article, same thinking + JSON format

### Output Format (per line)

```json
{"article_idx": 0, "date": "2026-01-07", "title": "...", "source": "news", "status": "success", "thinking": "...", "signal": {"signal_vector": {...}, "event_type": "...", ...}}
```

### Anti-Cheating Rules (CRITICAL)

1. **Market context must only use data UP TO the article's publication date.** Compute 5d/30d returns from the parquet files ending at or before the article date. Never include future price data.

2. **Do NOT look up what happened to stocks after the article date.** You are predicting, not backtesting. Treat each article as if today is the article's publication date.

3. **Do NOT reference any events that occurred after the article date.** If the article is from January 2026, you cannot reference anything from February 2026 or later.

4. **Do NOT use web browsing to check current stock prices.** Base your analysis solely on the article text, your pre-existing knowledge, and the historical market context provided.

### Selectivity Applies to Eval Too

Apply the same selectivity standard as training. If an eval article doesn't warrant a trading signal, output zeros with reasoning. The benchmark comparison will measure:
- IC (Information Coefficient) at 1, 2, 5, 10, 20 day horizons
- Direction accuracy on non-zero predictions
- Number of non-zero predictions (selectivity)

A model that outputs fewer, higher-conviction predictions will score better than one that scores everything.

### Processing All 426 Articles

Process every article in `articles_eval.jsonl` sequentially. For each:
1. Read the article's date
2. Compute market context from parquet files (up to that date only)
3. Apply the same system prompt and thinking format
4. Save the result

If an article fails for any reason, log it with `"status": "error"` and continue to the next one. Do not skip articles.

### Comparison Baseline

The Codex predictions will be compared against:
- **Manus teacher v1** (old prompt): IC = 0.063 at 5d
- **Manus teacher v2** (refined prompt): IC = 0.093 at 5d
- **V4 fine-tuned model**: IC = 0.075 at 5d

To run the comparison after generating predictions:
```bash
python scripts/compare_eval_ic.py
# This script automatically picks up predictions_codex_teacher.jsonl
# You may need to update the script to include the new file
```

---

## Key Insight to Internalize

The V4 model worked because it was SELECTIVE. It only scored ~3,265 ticker-article pairs out of a possible ~6,000+. The thinking traces in V5/V7 made the model score ~6,688 pairs — it reasoned its way into having an opinion on everything.

**The fix:** Train the model that thinking often leads to "no signal." The `<think>` block should frequently conclude with "this doesn't warrant a trade" and output zeros. This preserves the interpretability benefits of thinking traces while maintaining the selectivity that made V4 successful.

---

## Pushing Results to GitHub

After generation is complete:

```bash
cd quantum-alpha-intelligence

# Verify the output files exist and have expected line counts
wc -l data/training/quantum_alpha_train_v8.jsonl
wc -l data/eval/predictions_codex_teacher.jsonl

# Stage, commit, and push
git add data/training/quantum_alpha_train_v8.jsonl
git add data/eval/predictions_codex_teacher.jsonl
git commit -m "V8 training data: thinking traces with selectivity (Codex teacher)

Generated by Codex agent. Key properties:
- X examples total (report actual count)
- ~40-50% Type B (thinking leads to zeros)
- ~30-35% Type A (genuine signal)
- ~20-25% Type C (moderate signal)
- All with <think> reasoning traces
- Eval predictions: 426 articles as Codex teacher baseline"

git push origin fix/label-quality
```

### Validation Before Pushing

Run this validation script before committing:

```python
import json

def validate_v8(filepath):
    with open(filepath) as f:
        records = [json.loads(l) for l in f if l.strip()]
    
    errors = 0
    type_a = 0  # genuine signal
    type_b = 0  # considered zero
    type_c = 0  # moderate
    
    for i, r in enumerate(records):
        assistant = r['messages'][2]['content']
        
        # Check format
        if '<think>' not in assistant or '</think>' not in assistant:
            print(f'ERROR {i}: missing think tags')
            errors += 1
            continue
        
        # Extract JSON
        think_end = assistant.find('</think>')
        json_str = assistant[think_end + len('</think>'):].strip()
        try:
            parsed = json.loads(json_str)
        except:
            print(f'ERROR {i}: JSON parse failed')
            errors += 1
            continue
        
        # Check tickers
        sv = parsed.get('signal_vector', {})
        if len(sv) < 10:
            print(f'ERROR {i}: only {len(sv)} tickers')
            errors += 1
        
        # Check inactive
        for t in ['MSFT', 'GOOGL', 'NVDA']:
            if sv.get(t, {}).get('score', 0) != 0.0:
                print(f'ERROR {i}: {t} != 0.0')
                errors += 1
        
        # Classify type
        active_scores = [abs(sv.get(t, {}).get('score', 0)) for t in ['IONQ','RGTI','QBTS','QUBT','QNT','IBM','HON']]
        max_score = max(active_scores)
        if max_score >= 0.5:
            type_a += 1
        elif max_score < 0.05:
            type_b += 1
        else:
            type_c += 1
    
    total = len(records)
    print(f'Total: {total} | Errors: {errors}')
    print(f'Type A (signal): {type_a} ({type_a/total*100:.0f}%)')
    print(f'Type B (zeros):  {type_b} ({type_b/total*100:.0f}%)')
    print(f'Type C (moderate): {type_c} ({type_c/total*100:.0f}%)')
    print(f'Target: A=30-35%, B=40-50%, C=20-25%')

validate_v8('data/training/quantum_alpha_train_v8.jsonl')
```
