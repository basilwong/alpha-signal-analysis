# Roadmap: Current State to Final MVP

## Current State (What's Done)

- GitHub repo with full project structure
- HF Space created (`build-small-hackathon/quantum-alpha-intelligence`)
- 611 raw articles collected (arXiv + news, Aug 2024 to Jun 2026)
- 199 training examples generated (old schema: single-ticker sentiment)
- Qwen3-8B fine-tuned on Modal (old schema, LoRA rank 64, loss 1.96)
- Model published to HF Hub (`basilwong/quantum-alpha-qwen3-8b`)
- Basic Gradio app deployed (single-article analysis, current schema)
- Qwen Cloud API confirmed working (Singapore free tier, qwen3-max)
- Modal infrastructure confirmed working (A100 GPU access)
- Research docs: prior art, evaluation methodology, library analysis
- Frontend design V2 spec written

## What Needs to Change

The output schema has fundamentally changed. The model now needs to produce a **full signal vector across all 9 tickers** for every article, instead of just a primary ticker sentiment. This requires regenerating training data and retraining the model.

## Complete Roadmap

### Phase 1: New Output Schema and Training Data

**Objective**: Generate high-quality training data with the new cross-sectional signal vector schema.

| Step | Task | Dependency | Estimated Time | Resource |
|------|------|------------|----------------|----------|
| 1.1 | Write new system prompt with signal vector schema | None | 30 min | Local |
| 1.2 | Define static sector data (technology clusters, revenue exposures, competitive relationships) | None | 30 min | Local |
| 1.3 | Test new prompt with qwen3-max on 5 articles (verify output quality) | 1.1, 1.2 | 10 min | Qwen Cloud free tier |
| 1.4 | Generate training data for all 611 articles with new schema | 1.3 | ~60 min | Qwen Cloud free tier (~600K tokens) |
| 1.5 | Quality check: manually review 20 examples for correctness | 1.4 | 30 min | Manual |
| 1.6 | Split dataset: 200 for training, 411 for evaluation (temporal split) | 1.5 | 5 min | Local |

**Output**: `data/training/quantum_alpha_train_v2.jsonl` (200 examples, new schema)

### Phase 2: Model Retraining

**Objective**: Fine-tune Qwen3-8B on the new signal vector schema.

| Step | Task | Dependency | Estimated Time | Resource |
|------|------|------------|----------------|----------|
| 2.1 | Upload new training data to Modal volume | 1.6 | 1 min | Modal |
| 2.2 | Run fine-tuning (QLoRA, rank 64, 4 epochs, save per epoch) | 2.1 | ~5 min | Modal A100 (~$1.50) |
| 2.3 | Test inference on 5 held-out articles (verify signal vector output) | 2.2 | 5 min | Modal A100 (~$0.50) |
| 2.4 | Push model to HF Hub (`build-small-hackathon/quantum-alpha-qwen3-8b-v2`) | 2.3 | 2 min | Modal |
| 2.5 | (Optional) Run base Qwen3-8B on same 5 articles for comparison | 2.3 | 5 min | Modal A100 (~$0.50) |

**Output**: Fine-tuned model on HF Hub producing full signal vectors

### Phase 3: Generate Predictions for Evaluation

**Objective**: Run the fine-tuned model on all 411 evaluation articles to produce predictions.

| Step | Task | Dependency | Estimated Time | Resource |
|------|------|------------|----------------|----------|
| 3.1 | Write batch prediction script (runs model on all eval articles) | 2.4 | 30 min | Local |
| 3.2 | Run predictions on 411 evaluation articles | 3.1 | ~60 min | Modal A100 (~$3.00) |
| 3.3 | (Optional) Run base model predictions for comparison | 3.1 | ~60 min | Modal A100 (~$3.00) |
| 3.4 | Save predictions to `data/eval/predictions_v2.jsonl` | 3.2 | Automatic | Modal |

**Output**: `data/eval/predictions_v2.jsonl` (411 signal vectors with all tickers)

### Phase 4: Market Data and Abnormal Returns

**Objective**: Download historical prices and compute abnormal returns for evaluation.

| Step | Task | Dependency | Estimated Time | Resource |
|------|------|------------|----------------|----------|
| 4.1 | Download historical prices for all tickers + SPY (yfinance) | None | 5 min | Free |
| 4.2 | Compute daily returns for all tickers | 4.1 | 2 min | Local |
| 4.3 | Construct quantum sector basket (equal-weight IONQ+RGTI+QBTS) | 4.2 | 5 min | Local |
| 4.4 | For each event: estimate market model (OLS, 180-day window) | 4.2, 3.4 | 10 min | Local |
| 4.5 | Compute abnormal returns and CAR at windows (+1, +2, +5, +10, +20, +60) | 4.4 | 10 min | Local |
| 4.6 | Save to `data/eval/abnormal_returns.parquet` | 4.5 | 1 min | Local |

**Output**: Abnormal returns for every (event, ticker) pair at multiple horizons

### Phase 5: Evaluation Metrics (Alphalens + Custom)

**Objective**: Compute IC, signal decay, and all evaluation metrics.

| Step | Task | Dependency | Estimated Time | Resource |
|------|------|------------|----------------|----------|
| 5.1 | Install alphalens-reloaded, format data for alphalens input | 4.6, 3.4 | 30 min | Local |
| 5.2 | Run alphalens: compute IC, quantile returns, decay curve | 5.1 | 5 min | Local |
| 5.3 | Compute custom metrics: direction accuracy, magnitude accuracy, cross-asset accuracy | 5.1 | 30 min | Local |
| 5.4 | Compute signal decay curve (IC at horizons 1,2,5,10,20,60) | 5.1 | 10 min | Local |
| 5.5 | Compute IC by subset (source, ticker, event type) with Bonferroni | 5.2 | 10 min | Local |
| 5.6 | Bootstrap confidence intervals (1000 resamples) | 5.2 | 5 min | Local |
| 5.7 | (If base model predictions available) Compute comparison metrics | 5.2, 3.3 | 10 min | Local |
| 5.8 | Save all results to `data/eval/results.json` | 5.2-5.7 | 1 min | Local |

**Output**: `data/eval/results.json` with all metrics, `data/eval/charts/` with visualizations

### Phase 6: Frontend Build

**Objective**: Build the three-tab Gradio interface per the V2 design spec.

| Step | Task | Dependency | Estimated Time | Resource |
|------|------|------------|----------------|----------|
| 6.1 | Write static sector data file (clusters, exposures, profiles) | 1.2 | 30 min | Local |
| 6.2 | Build Tab 1: Signal Explorer (event navigation + signal vector chart) | 3.4, 4.6 | 3 hours | Local |
| 6.3 | Build Tab 1: Predicted vs Actual time series overlay | 4.6 | 2 hours | Local |
| 6.4 | Build Tab 1: Event metrics cards + reasoning trace | 5.8 | 1 hour | Local |
| 6.5 | Build Tab 1: Live analysis mode (paste article, run inference) | 2.4 | 1 hour | Local |
| 6.6 | Build Tab 1: Model selector toggle | 2.4, 2.5 | 1 hour | Local |
| 6.7 | Build Tab 2: Evaluation Dashboard (summary metrics, charts) | 5.8 | 2 hours | Local |
| 6.8 | Build Tab 3: Sector Map (network graph + propagation simulator) | 6.1 | 2 hours | Local |
| 6.9 | Add info tooltips to all metrics | 6.2-6.8 | 1 hour | Local |
| 6.10 | Test full app locally | 6.2-6.9 | 1 hour | Local |

**Output**: Complete Gradio app with all three tabs functional

### Phase 7: Deployment and Submission

**Objective**: Deploy to HF Space and prepare hackathon submission materials.

| Step | Task | Dependency | Estimated Time | Resource |
|------|------|------------|----------------|----------|
| 7.1 | Push final app + pre-computed data to HF Space | 6.10 | 10 min | HF |
| 7.2 | Verify app runs on ZeroGPU (live analysis mode) | 7.1 | 15 min | HF |
| 7.3 | Record 2-minute demo video | 7.2 | 1 hour | Local |
| 7.4 | Write social media post | 7.3 | 15 min | Local |
| 7.5 | Write Field Notes blog post (for badge) | 5.8 | 1 hour | Local |
| 7.6 | Submit to Build Small hackathon | 7.3, 7.4 | 10 min | HF |

**Output**: Live hackathon submission

### Phase 8: Experimentation (Post-Submission, for Qwen Cloud Hackathon)

| Step | Task | Dependency | Estimated Time | Resource |
|------|------|------------|----------------|----------|
| 8.1 | Experiment: DoRA fine-tuning, compare IC vs LoRA | 2.4 | 2 hours | Modal |
| 8.2 | Experiment: Qwen3-32B fine-tuning (final version) | 1.6 | 4 hours | Modal |
| 8.3 | Experiment: Curriculum ordering | 1.6 | 2 hours | Modal |
| 8.4 | Add persistent memory (ChromaDB) for V1 agent | 7.1 | 4 hours | Local |
| 8.5 | Deploy on Alibaba Cloud ECS | 8.4 | 2 hours | Alibaba Cloud |
| 8.6 | Submit to Qwen Cloud hackathon (July 9) | 8.5 | 2 hours | DevPost |

## Resource Budget

| Resource | Budget | Estimated Usage | Remaining |
|----------|--------|-----------------|-----------|
| Modal credits | $280 | ~$12 (training + inference runs) | ~$268 |
| Qwen Cloud free tier | 1M tokens | ~800K tokens (training data gen) | ~200K |
| HF ZeroGPU | $20 credits | Minimal (live inference only) | ~$19 |
| Time (to Build Small deadline, June 15) | 8 days | Phases 1-7 | Tight but doable |

## Critical Path

The longest dependency chain determines the minimum time to completion:

```
Schema design (30min)
    → Training data generation (60min)
        → Fine-tuning (5min)
            → Batch predictions on eval set (60min)
                → Market data + abnormal returns (30min)
                    → Alphalens evaluation (30min)
                        → Frontend build (12+ hours)
                            → Deployment + submission (2 hours)
```

**Minimum calendar time**: 3-4 days of focused work.

The frontend (Phase 6) is the largest single block of work at ~12 hours. Everything before it is pipeline work that can run mostly unattended (data generation, training, batch predictions).

## Suggested Execution Order (Parallelizable)

**Day 1**: Phases 1 + 2 (new schema, training data, retrain model)
**Day 2**: Phases 3 + 4 (batch predictions + market data, runs in parallel)
**Day 3-4**: Phase 5 + 6 (evaluation + frontend build)
**Day 5**: Phase 7 (deployment, video, submission)

This gets us to submission by June 12, with 3 days of buffer before the June 15 deadline.
