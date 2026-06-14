# Codex Follow-Up: Generate Synthetic Training Components

## Execution Report

Completed on branch `fix/label-quality` and pushed in commit `c58ee22`:

```text
c58ee22 V8 synthetic training components (550 examples)
```

### Final Outputs

| File | Rows | Type A | Type B | Type C | Purpose |
|------|------|--------|--------|--------|---------|
| `data/training/v8_synthetic.jsonl` | 200 | 70 | 80 | 50 | Broader realistic quantum-news coverage |
| `data/training/v8_negatives.jsonl` | 150 | 0 | 150 | 0 | Hard selectivity training, all-zero labels |
| `data/training/v8_edge_cases.jsonl` | 100 | 0 | 60 | 40 | Ambiguous, stale, priced-in, and indirect cases |
| `data/training/v8_paraphrased.jsonl` | 100 | 40 | 50 | 10 | Style invariance with preserved source scores |
| **Total** | **550** | **110** | **340** | **100** | V8 synthetic add-on set |

Combined with `quantum_alpha_train_v8.jsonl`, the V8 training set now has 740 examples:

| Component | Rows | Type A | Type B | Type C | Active Non-Zero Pairs |
|-----------|------|--------|--------|--------|------------------------|
| `quantum_alpha_train_v8.jsonl` | 190 | 52 | 128 | 10 | 204 / 1330, 15.3% |
| `v8_synthetic.jsonl` | 200 | 70 | 80 | 50 | 468 / 1400, 33.4% |
| `v8_negatives.jsonl` | 150 | 0 | 150 | 0 | 0 / 1050, 0.0% |
| `v8_edge_cases.jsonl` | 100 | 0 | 60 | 40 | 160 / 700, 22.9% |
| `v8_paraphrased.jsonl` | 100 | 40 | 50 | 10 | 162 / 700, 23.1% |
| **Combined** | **740** | **162** | **468** | **110** | **994 / 5180, 19.2%** |

### Step-By-Step Breakdown

1. Verified repository state.
   - Confirmed branch `fix/label-quality` was clean and synced with `origin/fix/label-quality`.
   - Confirmed the existing V8 real-article files from the prior run were present.
   - Confirmed `quantum_alpha_train_v8.jsonl` had 190 examples and `predictions_codex_teacher.jsonl` had 426 eval rows.

2. Reviewed the follow-up prompt and tightened the execution strategy.
   - Kept the four requested output files and target counts.
   - Added a manifest-first workflow so category, type distribution, ticker coverage, and style coverage were controlled before generation.
   - Preserved the exact system prompt from the first row of `quantum_alpha_train_v8.jsonl`.
   - Required GPT-5.5 with `xhigh` reasoning through fresh Codex worker sessions.
   - Avoided OpenAI API usage entirely.

3. Added quality constraints beyond the original prompt.
   - Added hard negatives that mention mega-cap tickers or active universe names but are not quantum-specific, such as IBM consulting, Microsoft Azure AI, Google search advertising, and NVIDIA AI GPUs.
   - Kept synthetic bearish scenarios concrete: confirmed misses, guidance cuts, formal investigations, actual dilution, or credible technical setbacks.
   - Treated rumors, generic lawsuits, vague short reports, and unsupported competitor claims as zero or mild signals rather than strong bearish labels.
   - Made QNT/Quantinuum IPO and lockup examples use 2026 dates instead of 2025 dates.
   - Made paraphrases preserve the exact source ticker scores from V8, not just the same broad direction.

4. Built a controlled manifest.
   - Temporary helper scripts generated JSONL manifests under `logs/v8_components/manifests/`.
   - The manifest was chunked into 11 independent 50-row chunks:
   - `synthetic_00` through `synthetic_03`
   - `negatives_00` through `negatives_02`
   - `edge_cases_00` and `edge_cases_01`
   - `paraphrased_00` and `paraphrased_01`
   - `logs/` is ignored by git, so these intermediate files were not committed.

5. Generated each chunk with fresh GPT-5.5/xhigh Codex workers.
   - Each chunk used a fresh subagent session.
   - Up to three workers were kept active concurrently.
   - Each worker read only its assigned manifest and wrote only its assigned chunk output under `logs/v8_components/chunks/`.
   - Workers were instructed not to edit final training files and not to revert other work.

6. Assembled final files from worker chunks.
   - A temporary assembler read every chunk, validated it against the manifest, and wrote the four final files.
   - The assembler enforced row counts, ticker coverage, inactive ticker zeros, score ranges, target type, required signal fields, and thinking trace presence.
   - For `v8_negatives.jsonl`, the assembler required every ticker score to be exactly `0.0`.
   - For `v8_paraphrased.jsonl`, the assembler required every ticker score to exactly match the source V8 example at the selected `source_idx`.

7. Ran the requested validation script.
   - `v8_synthetic.jsonl`: 200 examples, 0 errors, A/B/C = 70/80/50.
   - `v8_negatives.jsonl`: 150 examples, 0 errors, A/B/C = 0/150/0.
   - `v8_edge_cases.jsonl`: 100 examples, 0 errors, A/B/C = 0/60/40.
   - `v8_paraphrased.jsonl`: 100 examples, 0 errors, A/B/C = 40/50/10.

8. Ran additional strict checks.
   - Verified all four files use the exact V8 system prompt.
   - Verified all 10 tickers are present in every assistant JSON.
   - Verified `MSFT`, `GOOGL`, and `NVDA` are always exactly `0.0`.
   - Verified active ticker ranges: pure-plays in `[-2.0, +2.0]`, IBM in `[-0.15, +0.15]`, HON in `[-0.3, +0.3]`.
   - Verified `v8_negatives.jsonl` is 100% all-zero.
   - Verified `git diff --check` passed.

9. Cleaned up temporary generation files.
   - Removed temporary helper scripts before committing.
   - Did not commit manifests, worker chunk outputs, or logs.
   - Only the four requested JSONL training files were staged.

10. Committed and pushed.
    - Commit: `c58ee22 V8 synthetic training components (550 examples)`.
    - Pushed to `origin/fix/label-quality`.
    - Final branch state: `fix/label-quality` synced with `origin/fix/label-quality`.

### Verification Commands Run

```bash
wc -l data/training/v8_synthetic.jsonl \
      data/training/v8_negatives.jsonl \
      data/training/v8_edge_cases.jsonl \
      data/training/v8_paraphrased.jsonl
```

Output:

```text
     200 data/training/v8_synthetic.jsonl
     150 data/training/v8_negatives.jsonl
     100 data/training/v8_edge_cases.jsonl
     100 data/training/v8_paraphrased.jsonl
     550 total
```

The validation script from this prompt was run with `/usr/bin/python3` and returned 0 errors for all four files. Additional strict validation also returned `strict_errors 0`.

### Notes For Future Runs

- The generated set is intentionally more selective than V4. V4 had a high active non-zero pair rate; the combined V8 set is 19.2%, preserving the V8 selectivity objective.
- The negative examples are stricter than generic unrelated news only: several mention IBM, Honeywell, Microsoft, Google, or NVIDIA in non-quantum contexts to teach that ticker mention alone is not a quantum alpha signal.
- Paraphrases are true style-invariance examples because scores are exactly copied from source V8 rows.
- The temporary manifest-and-assembler workflow is worth reusing for future data expansions because it prevents valid JSON from drifting into poor label distribution.

---

## Context

Your previous run (commit `d702986`) successfully generated V8 training data from 190 real articles and 421 eval predictions. However, the best-performing model (V4, IC=0.075) was trained on **881 examples that were 79% synthetic**. The V8 data with only 190 real articles is missing the synthetic components that made V4 effective.

**V4's composition that worked:**
| Category | Count | % | What It Taught |
|----------|-------|---|----------------|
| Real articles | 184 | 21% | Ground-truth signal from actual news |
| Synthetic scenarios | 196 | 22% | Broader coverage of events the model might encounter |
| Paraphrased articles | 178 | 20% | Style invariance (same content, different writing) |
| Negative examples | 130 | 15% | **SELECTIVITY** — non-quantum content → all zeros |
| Edge cases | 88 | 10% | Ambiguous situations requiring nuanced judgment |
| ArXiv papers | 70 | 8% | Academic content → mostly zeros |
| QNT dynamics | 35 | 4% | IONQ-QNT competitive relationship |

**The key insight:** The 130 "negative" examples (non-quantum content that should score all zeros) were likely the biggest contributor to V4's selectivity. They taught the model that "most things don't deserve a score."

---

## Repository Setup

```bash
cd quantum-alpha-intelligence
git checkout fix/label-quality
git pull origin fix/label-quality

# The raw article files are gitignored. They exist in the V8 training data
# you already generated, so you don't need them again for this task.

# Install dependencies
pip install pandas pyarrow numpy
```

---

## Your Task

Generate 4 additional training data files that complement the V8 real-article data:

| Output File | Examples | Description |
|-------------|----------|-------------|
| `data/training/v8_synthetic.jsonl` | ~200 | Synthetic quantum computing scenarios |
| `data/training/v8_negatives.jsonl` | ~150 | Non-quantum content → all zeros |
| `data/training/v8_edge_cases.jsonl` | ~100 | Ambiguous scenarios requiring nuanced judgment |
| `data/training/v8_paraphrased.jsonl` | ~100 | Real articles rewritten in different styles |

**Total: ~550 additional examples**, bringing V8 from 190 to ~740 examples.

All files use the same format as `quantum_alpha_train_v8.jsonl` (OpenAI messages with `<think>` traces).

---

## Format (Same as V8)

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "<think>\n...\n</think>\n{...}"}]}
```

Use the same system prompt from `quantum_alpha_train_v8.jsonl` (read the first example's system message).

---

## Category 1: Synthetic Scenarios (~200 examples)

Generate realistic quantum computing news scenarios that COULD happen. These expand the model's coverage beyond the 194 real articles.

### Types of Synthetic Scenarios

**Hardware milestones (40):**
- "IonQ achieves 50 algorithmic qubits on their Forte Enterprise processor"
- "Rigetti demonstrates 99.5% two-qubit gate fidelity on 84-qubit Ankaa-3"
- "D-Wave announces 7000-qubit system solving logistics 100x faster than classical"
- "QNT demonstrates fault-tolerant logical qubit with 10x lifetime improvement"

**Business events (40):**
- "IonQ wins $150M Air Force contract for quantum-secured communications"
- "Rigetti partners with BMW for production quantum optimization"
- "D-Wave secures $200M strategic investment from Samsung"
- "QUBT announces first commercial customer for reservoir computing product"

**Competitive moves (30):**
- "Google demonstrates 1000-qubit superconducting processor with below-threshold errors"
- "Microsoft topological qubit achieves 99.99% fidelity"
- "Amazon announces proprietary quantum hardware program"
- "Chinese quantum startup matches IonQ performance at 1/5th cost"

**Earnings/financials (30):**
- "IonQ reports Q3 revenue of $15M, beating estimates by 25%"
- "Rigetti misses revenue estimates, cuts guidance"
- "D-Wave reports first quarter of positive free cash flow"
- "QNT IPO lockup expires, Honeywell sells 20% stake"

**Sector-wide events (30):**
- "Congress passes $5B quantum computing funding bill"
- "Quantum winter narrative gains traction after VC funding drops 60%"
- "QTUM ETF sees record inflows of $500M in single week"
- "Fed rate hike crushes all speculative growth stocks"

**Talent/leadership (15):**
- "IonQ co-founder leaves for Google Quantum AI"
- "Rigetti's entire QEC team resigns to start competitor"
- "QNT hires former IBM quantum VP as CTO"

**Regulatory (15):**
- "SEC investigates IonQ revenue recognition"
- "Export controls restrict quantum hardware sales internationally"
- "Patent lawsuit threatens Rigetti's multi-chip architecture"

### Distribution Within Synthetic

Maintain the Type A/B/C distribution:
- ~35% should be Type A (clear signal, |score| > 0.5)
- ~40% should be Type B (thinking concludes with zeros — "this is routine/incremental/not material")
- ~25% should be Type C (moderate, |score| 0.1-0.5)

**Important:** Many synthetic scenarios should be MILD versions that don't warrant trading. For example:
- "IonQ announces minor partnership with consulting firm" → Type B (zeros)
- "IonQ wins $200M defense contract" → Type A (strong signal)

---

## Category 2: Negative Examples (~150 examples)

These are articles about topics COMPLETELY UNRELATED to quantum computing. The correct output is ALL ZEROS for every ticker. This teaches selectivity.

### Types of Negative Content

Generate realistic article text (2-3 paragraphs) about:

**Unrelated tech (30):** AI/ML advances, cloud computing, cybersecurity, semiconductors (non-quantum), social media
**Finance/macro (25):** Interest rate decisions, bank earnings, crypto markets, real estate
**Science (non-quantum) (25):** Biology, chemistry, materials science, astronomy, climate
**Industry (25):** Automotive, healthcare, energy, agriculture, manufacturing
**General news (25):** Politics, sports, entertainment, weather, geopolitics
**Tangentially related but no signal (20):** General "future of computing" pieces, quantum computing history articles, educational content about quantum mechanics with no commercial implications

### Negative Example Format

The thinking block MUST explain why there's no quantum trading signal:

```
assistant: <think>
This article is about the Federal Reserve's interest rate decision. While rate changes affect all stocks including quantum computing companies, this is a macro event with no quantum-specific implications. The impact on quantum stocks would be through general market beta, not quantum-specific alpha. No signal.
</think>
{"signal_vector": {"IONQ": {"score": 0.0, "reasoning": "Macro event, no quantum-specific signal."}, "RGTI": {"score": 0.0, "reasoning": "Macro event."}, "QBTS": {"score": 0.0, "reasoning": "Macro event."}, "QUBT": {"score": 0.0, "reasoning": "Macro event."}, "QNT": {"score": 0.0, "reasoning": "Macro event."}, "IBM": {"score": 0.0, "reasoning": "Macro event."}, "HON": {"score": 0.0, "reasoning": "Macro event."}, "MSFT": {"score": 0.0, "reasoning": "Inactive."}, "GOOGL": {"score": 0.0, "reasoning": "Inactive."}, "NVDA": {"score": 0.0, "reasoning": "Inactive."}}, "event_type": "macro_unrelated", "time_horizon": "intraday", "information_novelty": "low", "technical_translation": "Federal Reserve rate decision. No quantum computing implications.", "signal_rationale": "No quantum-specific signal. Macro events affect all stocks through beta, not alpha."}
```

---

## Category 3: Edge Cases (~100 examples)

Ambiguous scenarios that test the model's judgment. These should produce MODERATE or ZERO scores with thoughtful reasoning.

### Types of Edge Cases

**Ambiguous relevance (20):**
- "A Nobel Prize is awarded for theoretical quantum physics work with unclear commercial timeline"
- "Quantum computing mentioned positively in State of the Union but no funding announced"
- "A quantum computing company's stock surges 30% on Reddit hype with no fundamental news"

**Conflicting signals (20):**
- "IonQ beats revenue estimates but announces massive dilutive offering"
- "Rigetti wins major contract but their CTO resigns same day"
- "D-Wave proves quantum advantage but only on a niche problem worth $10M/year"

**Priced-in / overextended (20):**
- Stocks up 40-60% in prior week + trivial positive news → should be zero or mildly bearish
- Use high market context numbers to signal overextension

**Stale/repeated news (20):**
- "This is the 4th article this week about Google Willow" → diminishing signal
- "Analyst reiterates existing rating with no new analysis" → zero

**Indirect/second-order effects (20):**
- "Nvidia announces new GPU that could accelerate quantum simulation" → maybe mild positive for sector?
- "University publishes theoretical paper that might eventually help trapped-ion scaling" → probably zero

---

## Category 4: Paraphrased Articles (~100 examples)

Take the TITLES from the V8 training data (which you already generated) and rewrite them as if reported by different sources with different styles. This teaches style invariance.

### Styles to Use (cycle through these)

1. **Formal SEC filing language** — "The registrant hereby announces..."
2. **Casual tech blog** — "So IonQ just dropped some wild news..."
3. **Reddit r/investing post** — "DD on $IONQ: They just landed a huge contract..."
4. **Bloomberg terminal headline** — "IONQ WINS $200M DOD CONTRACT; SHARES +15% PREMARKET"
5. **Academic abstract style** — "We report on recent developments in trapped-ion quantum computing..."
6. **Skeptical journalist** — "IonQ claims another breakthrough, but should investors believe the hype?"
7. **Bullish analyst note** — "Reiterate Overweight. Contract win validates our thesis..."
8. **Bearish short-seller** — "Another day, another quantum computing company overpromising..."

### Important

The SCORES should be the same regardless of style. The model must learn that the same event reported in different ways produces the same signal. The thinking should note: "Despite the [casual/formal/skeptical] tone, the underlying event is [X] which warrants [Y] score."

---

## Market Context for Synthetic Examples

Since synthetic scenarios don't have real dates, generate plausible market context. Use these templates:

**Normal market (60% of examples):**
```
**Market Context (as of 2025-06-15):**
| Ticker | 5d Ret | 30d Ret | 52w Position | Liquidity |
|--------|--------|---------|--------------|-----------|
| IONQ | +1.2% | -3.5% | Mid-range | High |
| RGTI | -0.8% | +2.1% | Mid-range | High |
| QBTS | +0.5% | -1.2% | Mid-range | Medium |
| QUBT | -0.3% | +0.8% | Mid-range | Medium |
| QNT | +1.5% | +4.2% | Mid-range | High |
| IBM | +0.2% | +1.8% | Near high | Very High |
| HON | -0.1% | +0.5% | Mid-range | Very High |
```

**Bull market (20% of examples):** 5d returns +5% to +30%, 30d returns +15% to +60%

**Bear market (20% of examples):** 5d returns -5% to -25%, 30d returns -15% to -45%

---

## Validation

Run this before committing:

```python
import json
from collections import Counter

files = {
    'v8_synthetic.jsonl': 200,
    'v8_negatives.jsonl': 150,
    'v8_edge_cases.jsonl': 100,
    'v8_paraphrased.jsonl': 100,
}

for filename, target in files.items():
    path = f'data/training/{filename}'
    try:
        with open(path) as f:
            records = [json.loads(l) for l in f if l.strip()]
    except FileNotFoundError:
        print(f'MISSING: {filename}')
        continue
    
    errors = 0
    type_a, type_b, type_c = 0, 0, 0
    
    for i, r in enumerate(records):
        assistant = r['messages'][2]['content']
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
    print(f'{filename}: {total} examples (target: {target}) | Errors: {errors}')
    print(f'  A={type_a} ({type_a/total*100:.0f}%) B={type_b} ({type_b/total*100:.0f}%) C={type_c} ({type_c/total*100:.0f}%)')
    
    # Special check for negatives: should be nearly 100% Type B
    if 'negatives' in filename and type_b / total < 0.9:
        print(f'  WARNING: Negatives should be >90% Type B (zeros)')
    print()
```

---

## Pushing Results

```bash
cd quantum-alpha-intelligence

# Verify
wc -l data/training/v8_synthetic.jsonl      # ~200
wc -l data/training/v8_negatives.jsonl       # ~150
wc -l data/training/v8_edge_cases.jsonl      # ~100
wc -l data/training/v8_paraphrased.jsonl     # ~100

# Run validation (above script)
python validate_synthetic.py

# Commit
git add data/training/v8_synthetic.jsonl
git add data/training/v8_negatives.jsonl
git add data/training/v8_edge_cases.jsonl
git add data/training/v8_paraphrased.jsonl
git commit -m "V8 synthetic training components (550 examples)

Complements the 190 real-article V8 examples with:
- v8_synthetic.jsonl: ~200 synthetic quantum scenarios
- v8_negatives.jsonl: ~150 non-quantum content (teaches selectivity)
- v8_edge_cases.jsonl: ~100 ambiguous scenarios (teaches nuance)
- v8_paraphrased.jsonl: ~100 style-varied rewrites (teaches invariance)

All with <think> reasoning traces. Selectivity-first approach:
negatives are >90% Type B (all zeros).

V4 was 79% synthetic and achieved IC=0.075. These components
replicate that composition while adding thinking traces."

git push origin fix/label-quality
```

---

## Key Reminders

1. **Selectivity is #1.** The negatives and many synthetic/edge cases should output ALL ZEROS with reasoning for why.

2. **The thinking block must justify zeros explicitly.** "This is not quantum-specific / not material / already priced in / too speculative to trade on."

3. **Don't make all synthetic scenarios dramatic.** Many should be boring/incremental (Type B). "IonQ publishes a blog post" → zeros. "IonQ wins $200M contract" → strong signal. The boring ones are MORE important for training selectivity.

4. **Negatives must be genuinely unrelated.** Not "quantum computing article that's bearish" — literally "article about corn futures" or "Tesla earnings report." The model must learn that non-quantum content = instant zeros without even needing to reason deeply.

5. **Use the same system prompt** from the existing V8 training data (read it from `quantum_alpha_train_v8.jsonl`).
