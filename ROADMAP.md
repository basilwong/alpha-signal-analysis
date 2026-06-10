# Quantum Alpha Intelligence Platform: Roadmap to Final Product

## Project Overview

Quantum Alpha Intelligence is an NLP-driven alpha signal generator for the quantum computing sector. It uses language models to analyze news, research papers, and press releases, producing cross-sectional trading signals across all public quantum computing companies simultaneously. The platform compares multiple models (fine-tuned small vs. large base models) to demonstrate that domain-specific fine-tuning on a small model can match or exceed larger general-purpose models.

**Hackathon Submissions:**
1. **Build Small** (Hugging Face/Gradio) - Deadline: June 15, 2026
2. **Qwen Cloud Global AI Hackathon** (Memory Agent track) - Deadline: July 9, 2026

## Current Status (June 10, 2026)

### What's Done

| Component | Status | Details |
|-----------|--------|---------|
| Data collection | Done | 611 articles (Aug 2024 - Jun 2026), split into train (190) and eval (421) |
| Temporal split fix | Done | Train on 2024-2025, eval on 2026 (proper walk-forward) |
| Outcome contamination cleanup | Done | 186 training articles had price statements removed |
| Training data (V3) | Done | 386 combined examples (V2 + V3) |
| Model fine-tuning (V3) | Done | Qwen3-8B, QLoRA rank 64, loss 1.095, pushed to HF Hub |
| Market data | Done | 608 trading days for 10 tickers (Yahoo Finance) |
| V2 Frontend | Done | 3-tab Gradio app deployed to HF Space |
| Live inference | Done | @spaces.GPU integrated for real-time analysis |

### What's Running Now

| Task | Method | Progress | ETA |
|------|--------|----------|-----|
| Qwen3-8B fine-tuned predictions | Modal A100 (batches of 50) | Batch 0/9 in progress | ~7 hours total (across multiple runs) |
| Qwen3-8B base predictions | Qwen Cloud API (free) | ~24/421 | ~50 min |
| Qwen3-32B base predictions | Qwen Cloud API (free) | Starting | ~2 hours |

## Remaining Tasks to Final Product

### Step 1: Complete All Model Predictions

All 4 models need predictions on the same 421 evaluation articles:

| Model | Source | Cost | Status |
|-------|--------|------|--------|
| Qwen3-8B fine-tuned (LoRA) | Modal GPU | ~$15 | In progress (batch-based, volume commit per batch) |
| Qwen3-8B base (no fine-tuning) | Qwen Cloud API | $0 (1M free tokens) | Running |
| Qwen3-32B base | Qwen Cloud API | $0 (1M free tokens) | Running |
| Qwen3.7-Max | Qwen Cloud API | Exhausted (18K tokens left) | Skip or run on subset |

**Output files:**
- `data/eval/predictions_qwen3_8b_finetuned.jsonl`
- `data/eval/predictions_qwen3_8b_base.jsonl`
- `data/eval/predictions_qwen3_32b_base.jsonl`

### Step 2: Run Evaluation on All Models

Run `eval/run_evaluation.py` on each model's predictions to compute:
- IC at horizons +1, +2, +5, +10, +20 days
- Signal decay curve
- Direction accuracy
- IC by ticker, source type, event type
- Bootstrap confidence intervals

Then generate comparison metrics:
- Side-by-side IC table
- Overlaid decay curves
- Per-ticker comparison
- Statistical tests for difference between models

**Output:** `data/eval/results_comparison.json`

### Step 3: Update Frontend with Multi-Model Support

**Tab 1 (Signal Explorer) updates:**
- Add model selector dropdown (Qwen3-8B fine-tuned, Qwen3-8B base, Qwen3-32B base)
- When user selects a model, load that model's predictions for the event
- Show side-by-side signal vectors when comparing models
- Live analysis uses the fine-tuned model via ZeroGPU

**Tab 2 (Evaluation Dashboard) updates:**
- Model comparison table (IC for each model at each horizon)
- Overlaid decay curves (all models on one chart, color-coded)
- Model selector to view individual model metrics
- Highlight where fine-tuned 8B beats larger models

**Tab 3 (Sector Map):** No changes needed.

### Step 4: Deploy Final App to HF Space

- Push updated `app_v2.py` with multi-model support
- Push all prediction files (3 models x 421 articles)
- Push evaluation results
- Verify ZeroGPU works for live analysis
- Test all 3 tabs end-to-end

### Step 5: Submission Materials

| Item | Description | Deadline |
|------|-------------|----------|
| Demo video (2 min) | Screen recording showing: event browsing, model comparison, live analysis, evaluation metrics | June 15 |
| Social media post | Tweet/LinkedIn post about the project (required for submission) | June 15 |
| Field Notes blog post | Technical write-up of fine-tuning approach, evaluation methodology, and results (for Well-Tuned badge) | June 15 |
| Final submission | Submit Space URL + video + post to Build Small hackathon | June 15 |

## Resource Budget (Updated June 10)

| Resource | Budget | Used | Remaining | Allocated For |
|----------|--------|------|-----------|---------------|
| Modal credits | $280 | ~$25 | ~$255 | Fine-tuned model batch predictions (~$15 more) |
| Qwen Cloud (qwen3-max) | 1M tokens | ~982K | ~18K | Exhausted |
| Qwen Cloud (qwen3-8b) | 1M tokens | ~30K | ~970K | Base model predictions (421 articles) |
| Qwen Cloud (qwen3-32b) | 1M tokens | ~0 | ~1M | Base model predictions (421 articles) |
| HF ZeroGPU | $20 credits | ~$1 | ~$19 | Live inference on deployed Space |

## Critical Path

```
Complete predictions (all 3 models)
    → Run evaluation on each
        → Generate comparison metrics
            → Update frontend with multi-model support
                → Deploy to HF Space
                    → Record demo video + write blog post
                        → Submit (June 15)
```

**Estimated time remaining:** 2-3 days of work (predictions are the bottleneck, everything else is fast once data is ready).

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| Jun 5 | Base model: Qwen3-8B | Best zero-shot baseline in benchmarks |
| Jun 5 | Fine-tuning: QLoRA rank 64 | Balance between quality and VRAM usage |
| Jun 6 | Teacher model: qwen3-max | Free, strongest model available |
| Jun 7 | Output schema: cross-sectional signal vector | Required for Alphalens, realistic for quant integration |
| Jun 8 | Evaluation: custom abnormal returns + IC | Alphalens for IC; custom OLS for CAR (avoids GPL) |
| Jun 8 | Input cleaning: strip HTML/URLs | Fixes JSON parse failure rate |
| Jun 9 | Fix temporal split | External review identified inverted walk-forward |
| Jun 9 | Combined training data (386 examples) | 2x data improves loss from 1.36 to 1.095 |
| Jun 10 | Multi-model comparison | Demonstrates fine-tuning value; uses free API for base models |
| Jun 10 | Batch-based Modal runs with volume commit per batch | Prevents losing progress on timeout |
| Jun 10 | Use Qwen Cloud API for base model inference | Free (1M tokens/model), saves Modal credits for training |

## Known Limitations (To Document in Field Notes)

1. Temporal split was initially inverted (fixed, but first eval results used wrong split)
2. Evaluation window is Jan-Jun 2026 only (single market regime)
3. Daily granularity (no intraday timing)
4. Single-factor market model (SPY only, no sector factor)
5. 386 training examples is small (may cause JSON compliance issues)
6. Model inference takes ~60s per article (not suitable for real-time trading without optimization)
7. Correlation does not imply causation
8. No transaction cost modeling

## Stretch Goals: Fine-Tuning Methods and Model Experiments

### Goal 1: Compare Fine-Tuning Methods

Test whether different fine-tuning approaches produce better IC than our current QLoRA setup.

| Method | Description | Hypothesis | Effort | Resource |
|--------|-------------|-----------|--------|----------|
| **DoRA** | Weight-Decomposed Low-Rank Adaptation (ICML 2024) | May outperform LoRA on structured output tasks by decomposing magnitude and direction | 2 hours | Modal A100 (~$3) |
| **GRPO** | Group Relative Policy Optimization (DeepSeek) | Reinforcement learning with price-based reward could teach the model to optimize for IC directly | 4 hours | Modal A100 (~$10) |
| **Full LoRA rank 128** | Double the LoRA rank from 64 to 128 | More trainable parameters = better representation capacity | 2 hours | Modal A100 (~$3) |
| **Curriculum ordering** | Sort training data from simple to complex articles | Progressive difficulty may improve final model quality | 1 hour | Modal A100 (~$2) |
| **Longer training (8 epochs)** | Double the training epochs | May improve JSON compliance and signal quality | 2 hours | Modal A100 (~$3) |

**Evaluation for each:** Run the same 421-article evaluation pipeline and compare IC at +5d. The current baseline is IC = +0.078.

### Goal 2: Fine-Tune Different Base Models

Test whether a different base model produces better results when fine-tuned with the same data.

| Base Model | Size | Approach | Hypothesis | Effort | Resource |
|-----------|------|----------|-----------|--------|----------|
| **Qwen3-32B** | 32B | QLoRA (rank 64) | Larger model = higher capacity, may produce better signals | 4 hours | Modal H100 (~$15) |
| **Qwen3-30B-A3B** | 30B (MoE, 3B active) | QLoRA | MoE architecture may be more efficient for structured tasks | 3 hours | Modal A100 (~$5) |
| **Llama-3.1-8B** | 8B | QLoRA (rank 64) | Different architecture, benchmarks show strong fine-tuning gains | 2 hours | Modal A100 (~$3) |
| **Phi-4-mini** | 3.8B | QLoRA (rank 64) | Smallest model, tests if even 4B can produce meaningful signals | 2 hours | Modal A100 (~$2) |

**For each fine-tuned model:**
1. Train with the same 386-example dataset
2. Run predictions on the 421 evaluation articles
3. Compute IC and compare against current best (Qwen3-8B fine-tuned, IC +0.078 at +5d)
4. Add to the frontend model selector for interactive comparison

### Goal 3: Ensemble Methods

| Approach | Description | Hypothesis |
|----------|-------------|-----------|
| Simple average | Average signal vectors across multiple models | Reduces noise, may improve IC |
| Weighted by IC | Weight each model's signal by its historical IC | Allocates more weight to better-performing models |
| Stacking | Train a meta-model on the signal vectors of all base models | May capture complementary strengths |

### Prioritization

For the Build Small hackathon (June 15): Focus on DoRA as the single most impactful experiment (one-line change, potential IC improvement).

For the Qwen Cloud hackathon (July 9): Run the full suite of experiments and present the comparison in the demo.
