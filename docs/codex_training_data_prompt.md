# Codex Prompt: Generate Quantum Alpha Training Data (V8) + Eval Predictions

## Repository Setup

```bash
# Clone the repo
git clone https://github.com/basilwong/quantum-alpha-intelligence.git
cd quantum-alpha-intelligence

# Switch to the working branch
git checkout fix/label-quality

# Install dependencies for market context computation
pip install pandas pyarrow numpy
```

### Key File Locations

| File | Description |
|------|-------------|
| `data/raw/articles_train.jsonl` | 194 training articles (2024-08-02 to 2025-12-31). Sources: news, arxiv. Fields: `text`, `source`, `title`, `date`, `url`, `outcome_cleaned` |
| `data/raw/articles_eval.jsonl` | 426 eval articles (2026-01-07 to 2026-06-09). Same fields. |
| `data/market/*.parquet` | Daily OHLCV data for: IONQ, RGTI, QBTS, QUBT, QNT, IBM, HON, MSFT, GOOGL, NVDA, SPY, QTUM |
| `src/market_context.py` | Helper module: `get_market_context(date)` computes the full market context table |
| `src/config.py` | Ticker universe, liquidity tiers, score ranges |
| `data/training/quantum_alpha_train_v4.jsonl` | Previous best training data (881 examples, reference for format) |

### Output Files to Generate

| File | Description |
|------|-------------|
| `data/training/quantum_alpha_train_v8.jsonl` | New training data (~194 examples with thinking traces) |
| `data/eval/predictions_codex_teacher.jsonl` | Eval predictions (426 articles, Codex teacher baseline) |

---

## Context

You are generating training data for a fine-tuned reasoning model that produces cross-sectional trading signals for quantum computing stocks. The model will run locally via vLLM with no internet access at inference time.

**Critical learning from previous iterations:**

1. **V4 (no thinking traces, 881 examples) is the best model so far.** IC = +0.075 at 5d. It works because it is SELECTIVE — it only produces non-zero scores when it has genuine conviction.

2. **V5/V7 (with thinking traces, 1121 examples) degraded performance.** IC = -0.010 at 5d. The thinking traces made the model LESS selective — it reasoned its way into making predictions it shouldn't have made. It produced 2x more non-zero predictions (6,688 vs 3,265), diluting the signal.

3. **Bearish examples hurt.** The synthetic bearish scenarios (earnings misses, setbacks) had lower label quality than the real-article examples. Removing them improved performance.

**The core problem to solve:** How do we get the benefits of thinking traces (better calibration, consistency, interpretability) WITHOUT losing selectivity? The answer is: the thinking must frequently conclude "this doesn't warrant a non-zero score" and output zeros.

---

## Task 1: Generate Training Data

Generate training data from the 194 articles in `data/raw/articles_train.jsonl`. Each article becomes one training example.

### Output Format (OpenAI messages format)

Each line of `quantum_alpha_train_v8.jsonl` is a JSON object:

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "<think>\n...\n</think>\n{...}"}]}
```

The assistant response MUST include a `<think>...</think>` block followed immediately by JSON:

```
<think>
[100-200 tokens of genuine reasoning]
</think>
{"signal_vector": {"IONQ": {"score": 0.0, "reasoning": "..."}, ...}, "event_type": "...", ...}
```

### The Three Types of Examples (Target Distribution)

**Type A: Genuine Signal (30-35% of examples, ~60-70 articles)**
- Article has clear, direct implications for specific quantum companies
- Thinking identifies the technology, affected companies, and competitive dynamics
- Produces non-zero scores with conviction (|score| > 0.5 for at least one ticker)
- Examples: major contract wins, hardware breakthroughs, earnings surprises, significant competitive moves

**Type B: Considered Zero (40-50% of examples, ~80-95 articles)**
- Article is about quantum computing but doesn't warrant trading action
- Thinking explicitly reasons through WHY there's no signal
- Concludes with all or nearly all zeros
- Examples: incremental papers, routine updates, old news, tangentially related content, generic outlook pieces
- THE THINKING MUST SHOW THE REASONING FOR ZERO: "This is incremental, already priced in, too speculative, or affects companies outside our universe"

**Type C: Moderate Signal (20-25% of examples, ~40-50 articles)**
- Article has some implications but conviction is low
- Thinking shows uncertainty and produces small scores (|score| 0.1-0.5)
- Examples: partnerships with unclear value, mixed news, ambiguous competitive moves

### How to Classify Each Article

Read the article and ask: "Would a portfolio manager at a quantum-focused fund actually trade on this within the next 5 days?"
- If YES with high conviction → Type A
- If NO → Type B
- If MAYBE / PARTIALLY → Type C

---

## Ticker Universe

**Active (assign scores):**
- IONQ: IonQ (trapped-ion, pure-play) — range [-2.0, +2.0]
- RGTI: Rigetti Computing (superconducting, pure-play) — range [-2.0, +2.0]
- QBTS: D-Wave Quantum (quantum annealing, pure-play) — range [-2.0, +2.0]
- QUBT: Quantum Computing Inc. (neutral atom, pure-play) — range [-2.0, +2.0]
- QNT: Quantinuum (trapped-ion, pure-play, IPO'd June 2026) — range [-2.0, +2.0]
- IBM: International Business Machines (superconducting, ~2% quantum revenue) — range [-0.15, +0.15]
- HON: Honeywell (trapped-ion, ~1% quantum revenue post-Quantinuum spinoff) — range [-0.3, +0.3]

**Inactive (always 0.0, but reason about their impact on active tickers):**
- MSFT: Microsoft (topological approach) — always 0.0
- GOOGL: Alphabet/Google (superconducting approach) — always 0.0
- NVDA: NVIDIA (quantum hardware enabler) — always 0.0

News ABOUT Google/Microsoft/NVIDIA is still relevant — it affects active tickers. For example, "Google achieves superconducting breakthrough" → GOOGL stays 0.0 but RGTI gets bullish (technology validation).

---

## Scoring Rules

1. **Selectivity is paramount.** Only assign non-zero scores when you have genuine conviction that this news will move the stock over 5 trading days. When in doubt, output 0.0.

2. **Technology validation rule:** When Google/IBM/Microsoft achieves a superconducting breakthrough → BULLISH for RGTI (validates the approach). When Quantinuum achieves a trapped-ion breakthrough → BULLISH for IONQ. Market evidence: RGTI surged +89% when Google announced Willow (Dec 2024).

3. **IONQ-QNT competitive dynamic:** Both are trapped-ion pure-plays. Sector-wide events → both move together. Company-specific wins → they diverge (zero-sum).

4. **ArXiv papers:** Almost always 0.0. Only non-zero for company-authored hardware results with measured metrics demonstrating a genuine milestone.

5. **Market context awareness:** If stocks are already up 30%+ in the prior week, bullish news may be priced in. If stocks are down 20%+, consider whether bad news is already reflected.

6. **Score magnitude guide:**
   - |0.0|: No signal / no conviction
   - |0.1-0.3|: Mild signal, low conviction
   - |0.5-1.0|: Moderate signal, clear implications
   - |1.0-1.5|: Strong signal, high conviction
   - |1.5-2.0|: Maximum conviction, transformative event

---

## System Prompt (Use This Exactly for All Examples)

```
You are a quantitative NLP signal generator for the quantum computing sector. You MUST think step-by-step before producing scores.

OUTPUT FORMAT:
<think>
[Your reasoning, 100-200 tokens]
</think>
{"signal_vector": {...}, "event_type": "...", ...}

CRITICAL: Most news does NOT warrant a trading signal. Your default output should be all zeros. Only assign non-zero scores when you have HIGH CONVICTION that this specific news will move a specific stock's price over the next 5 trading days.

Ask yourself: "Would a portfolio manager actually trade on this?" If the answer is "probably not," output zeros.

ACTIVE TICKERS (assign scores):
- IONQ: IonQ (trapped-ion, pure-play) — range [-2.0, +2.0]
- RGTI: Rigetti Computing (superconducting, pure-play) — range [-2.0, +2.0]
- QBTS: D-Wave Quantum (quantum annealing, pure-play) — range [-2.0, +2.0]
- QUBT: Quantum Computing Inc. (neutral atom, pure-play) — range [-2.0, +2.0]
- QNT: Quantinuum (trapped-ion, pure-play) — range [-2.0, +2.0]
- IBM: IBM (superconducting, ~2% quantum revenue) — range [-0.15, +0.15]
- HON: Honeywell (trapped-ion, ~1% quantum revenue) — range [-0.3, +0.3]

INACTIVE (always 0.0): MSFT, GOOGL, NVDA

TECHNOLOGY DYNAMICS:
- Trapped-ion breakthroughs → bullish IONQ/QNT/HON, may pressure RGTI/IBM
- Superconducting breakthroughs → bullish RGTI/IBM, may pressure IONQ/QNT
- Technology validation by big companies (Google SC breakthrough) → BULLISH for same-tech pure-plays (RGTI)
- Company-specific wins are zero-sum between direct competitors (IONQ vs QNT)
- Error correction advances → benefit all gate-based approaches
- Government funding → broadly bullish for sector

JSON STRUCTURE:
{"signal_vector": {"IONQ": {"score": float, "reasoning": "..."}, "RGTI": {...}, "QBTS": {...}, "QUBT": {...}, "QNT": {...}, "IBM": {...}, "HON": {...}, "MSFT": {"score": 0.0, "reasoning": "Inactive."}, "GOOGL": {"score": 0.0, "reasoning": "Inactive."}, "NVDA": {"score": 0.0, "reasoning": "Inactive."}}, "event_type": "...", "time_horizon": "intraday|2-5 days|1-2 weeks|1+ month", "information_novelty": "high|medium|low", "technical_translation": "2-3 sentences for a portfolio manager.", "signal_rationale": "Why these scores?"}
```

---

## User Message Format

For each article, construct the user message as follows:

```
**Market Context (as of {date}):**
| Ticker | 5d Ret | 30d Ret | 52w Position | Liquidity |
|--------|--------|---------|--------------|-----------|
| IONQ | +X.X% | +X.X% | Near high/Mid-range/Near low | High |
| RGTI | ... | ... | ... | High |
| QBTS | ... | ... | ... | Medium |
| QUBT | ... | ... | ... | Medium |
| QNT | ... | ... | ... | High |
| IBM | ... | ... | ... | Very High |
| HON | ... | ... | ... | Very High |

**Market regime:** Bull/Bear/Neutral / High/Normal/Low Volatility

[ARTICLE]
Title: {title}
Date: {date}
Source: {source}

{article_text}
```

**Computing market context:** Use `src/market_context.py`:
```python
import sys
sys.path.insert(0, '.')
from src.market_context import get_market_context
context = get_market_context("2025-03-15")  # returns formatted table or "" if no data
```

If `get_market_context()` returns empty string (no data for that date), omit the market context section.

---

## Examples

### Type B: Considered Zero (Most Common — 40-50% of examples)

```
assistant: <think>
This article discusses a minor software update to IonQ's cloud platform that improves job scheduling. This is routine engineering work, not a hardware breakthrough or commercial milestone. No portfolio manager would trade on this. The improvement is incremental and already expected as part of normal platform maintenance. All scores should be 0.0.
</think>
{"signal_vector": {"IONQ": {"score": 0.0, "reasoning": "Routine software update, no trading signal."}, "RGTI": {"score": 0.0, "reasoning": "Unrelated to Rigetti."}, "QBTS": {"score": 0.0, "reasoning": "Unrelated."}, "QUBT": {"score": 0.0, "reasoning": "Unrelated."}, "QNT": {"score": 0.0, "reasoning": "Unrelated."}, "IBM": {"score": 0.0, "reasoning": "Unrelated."}, "HON": {"score": 0.0, "reasoning": "Unrelated."}, "MSFT": {"score": 0.0, "reasoning": "Inactive."}, "GOOGL": {"score": 0.0, "reasoning": "Inactive."}, "NVDA": {"score": 0.0, "reasoning": "Inactive."}}, "event_type": "routine_update", "time_horizon": "intraday", "information_novelty": "low", "technical_translation": "Minor cloud platform improvement with no commercial significance.", "signal_rationale": "No trading signal. Incremental engineering work does not move stock prices."}
```

### Type A: Genuine Signal (Less Common — 30-35% of examples)

```
assistant: <think>
IonQ just won a $200M contract with the US Air Force for quantum computing services. This is their largest contract ever and represents ~4x their current annual revenue. It validates their trapped-ion approach for defense applications and provides multi-year revenue visibility.

Direct impact: IONQ strongly bullish (+1.8). QNT mildly bearish (-0.5) as they likely competed for this contract. RGTI neutral — different technology, different customer segment. IBM slightly positive (+0.1) as it validates enterprise quantum demand broadly.

Conviction: HIGH. This is a material, unexpected contract win with clear revenue implications.
</think>
{"signal_vector": {"IONQ": {"score": 1.8, "reasoning": "Largest contract in company history, 4x annual revenue, validates defense use case."}, "RGTI": {"score": 0.0, "reasoning": "Different technology and customer segment."}, "QBTS": {"score": 0.0, "reasoning": "Annealing approach not relevant to this contract type."}, "QUBT": {"score": 0.0, "reasoning": "Not competing in this space."}, "QNT": {"score": -0.5, "reasoning": "Direct competitor for trapped-ion defense contracts, likely lost this bid."}, "IBM": {"score": 0.1, "reasoning": "Validates enterprise quantum demand broadly."}, "HON": {"score": 0.15, "reasoning": "Trapped-ion validation, but limited direct exposure post-spinoff."}, "MSFT": {"score": 0.0, "reasoning": "Inactive."}, "GOOGL": {"score": 0.0, "reasoning": "Inactive."}, "NVDA": {"score": 0.0, "reasoning": "Inactive."}}, "event_type": "contract_win", "time_horizon": "2-5 days", "information_novelty": "high", "technical_translation": "IonQ secures transformative defense contract that provides multi-year revenue visibility and validates trapped-ion for government applications.", "signal_rationale": "Material contract win with high conviction. IONQ direct beneficiary, QNT loses competitive bid, sector broadly validated."}
```

### Type C: Moderate Signal (20-25% of examples)

```
assistant: <think>
This article reports that Rigetti has partnered with a European bank for a quantum optimization pilot. The partnership is exploratory — no committed revenue, no exclusivity, and the pilot may not lead to a production contract. It shows interest in Rigetti's platform but is not material enough for a strong signal. Mild positive for RGTI (+0.3) as it validates commercial interest, but low conviction.
</think>
{"signal_vector": {"IONQ": {"score": 0.0, "reasoning": "Not directly affected by this partnership."}, "RGTI": {"score": 0.3, "reasoning": "Exploratory partnership shows commercial interest but no committed revenue."}, "QBTS": {"score": 0.0, "reasoning": "Different technology approach."}, "QUBT": {"score": 0.0, "reasoning": "Not relevant."}, "QNT": {"score": 0.0, "reasoning": "Not competing for this type of engagement."}, "IBM": {"score": 0.0, "reasoning": "Not materially affected."}, "HON": {"score": 0.0, "reasoning": "Not relevant."}, "MSFT": {"score": 0.0, "reasoning": "Inactive."}, "GOOGL": {"score": 0.0, "reasoning": "Inactive."}, "NVDA": {"score": 0.0, "reasoning": "Inactive."}}, "event_type": "partnership_exploratory", "time_horizon": "1-2 weeks", "information_novelty": "medium", "technical_translation": "Rigetti secures an exploratory pilot with a European bank. Non-binding, no revenue commitment, but shows enterprise interest in superconducting quantum optimization.", "signal_rationale": "Low conviction positive for RGTI. Exploratory partnership validates interest but is not material enough for strong signal."}
```

---

## Implementation Steps

1. **Load articles:** Read all 194 articles from `data/raw/articles_train.jsonl`

2. **For each article:**
   a. Compute market context using `src/market_context.py` (only data up to article date)
   b. Construct the user message (market context + article)
   c. Determine the appropriate type (A/B/C) based on article content
   d. Generate the assistant response (thinking + JSON)
   e. Assemble into messages format

3. **Track distribution:** Keep a running count of Type A/B/C. Adjust as needed to hit targets.

4. **Validate each example:** All 10 tickers present, inactive = 0.0, ranges respected, thinking present, JSON parses.

5. **Save to `data/training/quantum_alpha_train_v8.jsonl`**

---

## Task 2: Generate Eval Predictions (Teacher Baseline)

Generate predictions for ALL 426 eval articles to benchmark Codex as a teacher model.

### Input

- File: `data/raw/articles_eval.jsonl`
- 426 articles from 2026-01-07 to 2026-06-09
- Each article has: `text`, `source`, `title`, `date`, `url`, `outcome_cleaned`

### Output

- File: `data/eval/predictions_codex_teacher.jsonl`
- One JSON object per line, one per article

### Output Format (per line)

```json
{"article_idx": 0, "date": "2026-01-07", "title": "...", "source": "news", "status": "success", "thinking": "...", "signal": {"signal_vector": {...}, "event_type": "...", "time_horizon": "...", "information_novelty": "...", "technical_translation": "...", "signal_rationale": "..."}}
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

**A model that outputs fewer, higher-conviction predictions will score better than one that scores everything.**

### Processing All 426 Articles

Process every article in `articles_eval.jsonl` sequentially (index 0 to 425). For each:
1. Read the article's date
2. Compute market context from parquet files (up to that date only)
3. Apply the same system prompt and thinking format
4. Save the result with `article_idx` matching the line number (0-indexed)

If an article fails for any reason, log it with `"status": "error"` and continue to the next one. Do not skip articles.

### Comparison Baseline

The Codex predictions will be compared against:
- **Manus teacher v1** (old prompt): IC = 0.063 at 5d
- **Manus teacher v2** (refined prompt): IC = 0.093 at 5d
- **V4 fine-tuned model**: IC = 0.075 at 5d

To run the comparison after generating predictions:
```bash
python scripts/compare_eval_ic.py
# You may need to update the script to include predictions_codex_teacher.jsonl
```

---

## Key Insight to Internalize

The V4 model worked because it was SELECTIVE. It only scored ~3,265 ticker-article pairs out of a possible ~6,000+. The thinking traces in V5/V7 made the model score ~6,688 pairs — it reasoned its way into having an opinion on everything.

**The fix:** Train the model that thinking often leads to "no signal." The `<think>` block should frequently conclude with "this doesn't warrant a trade" and output zeros. This preserves the interpretability benefits of thinking traces while maintaining the selectivity that made V4 successful.

---

## Pushing Results to GitHub

After BOTH tasks are complete:

```bash
cd quantum-alpha-intelligence

# Verify the output files exist and have expected line counts
wc -l data/training/quantum_alpha_train_v8.jsonl   # Should be ~194
wc -l data/eval/predictions_codex_teacher.jsonl     # Should be 426

# Run validation
python -c "
import json

def validate(filepath, name):
    with open(filepath) as f:
        records = [json.loads(l) for l in f if l.strip()]
    errors = 0
    type_a, type_b, type_c = 0, 0, 0
    for i, r in enumerate(records):
        # Handle both training format and eval format
        if 'messages' in r:
            assistant = r['messages'][2]['content']
        elif 'thinking' in r and 'signal' in r:
            assistant = '<think>' + r['thinking'] + '</think>' + json.dumps(r['signal'])
        else:
            errors += 1
            continue
        if '<think>' not in assistant or '</think>' not in assistant:
            errors += 1
            continue
        think_end = assistant.find('</think>')
        json_str = assistant[think_end + len('</think>'):].strip()
        try:
            parsed = json.loads(json_str)
        except:
            errors += 1
            continue
        sv = parsed.get('signal_vector', {})
        if len(sv) < 10:
            errors += 1
        for t in ['MSFT', 'GOOGL', 'NVDA']:
            if sv.get(t, {}).get('score', 0) != 0.0:
                errors += 1
        active_scores = [abs(sv.get(t, {}).get('score', 0)) for t in ['IONQ','RGTI','QBTS','QUBT','QNT','IBM','HON']]
        max_score = max(active_scores) if active_scores else 0
        if max_score >= 0.5: type_a += 1
        elif max_score < 0.05: type_b += 1
        else: type_c += 1
    total = len(records)
    print(f'{name}: {total} examples | Errors: {errors}')
    print(f'  Type A (signal): {type_a} ({type_a/total*100:.0f}%)')
    print(f'  Type B (zeros):  {type_b} ({type_b/total*100:.0f}%)')
    print(f'  Type C (moderate): {type_c} ({type_c/total*100:.0f}%)')
    print(f'  Target: A=30-35%, B=40-50%, C=20-25%')
    print()

validate('data/training/quantum_alpha_train_v8.jsonl', 'Training V8')
validate('data/eval/predictions_codex_teacher.jsonl', 'Eval Predictions')
"

# Stage, commit, and push
git add data/training/quantum_alpha_train_v8.jsonl
git add data/eval/predictions_codex_teacher.jsonl
git commit -m "V8 training data + Codex teacher eval predictions

Training (quantum_alpha_train_v8.jsonl):
- X examples from real articles (report actual count)
- ~40-50% Type B (thinking leads to zeros)
- ~30-35% Type A (genuine signal)
- ~20-25% Type C (moderate signal)
- All with <think> reasoning traces
- Selectivity-first approach

Eval (predictions_codex_teacher.jsonl):
- 426 articles predicted
- Anti-cheating: market context only uses data up to article date
- Codex teacher baseline for IC comparison

Generated by Codex agent."

git push origin fix/label-quality
```
