# Quantum Alpha Intelligence Platform: Project Roadmap

## Project Overview

Quantum Alpha Intelligence is an NLP-driven alpha signal generator for the quantum computing sector. It uses a fine-tuned small language model (Qwen3-8B) to analyze news articles, research papers, press releases, and other text sources, producing cross-sectional trading signals across all public quantum computing companies simultaneously.

The project is being submitted to two hackathons:
1. **Build Small** (Hugging Face/Gradio) - Deadline: June 15, 2026
2. **Qwen Cloud Global AI Hackathon** (Memory Agent track) - Deadline: July 9, 2026

## Core Architecture

```
Raw Text Input (news, arXiv, SEC, press releases)
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Fine-tuned Qwen3-8B (signal vector schema)     │
│  Produces scores for ALL 9 tickers per article  │
└─────────────────────────────────────────────────┘
    │
    ▼
Signal Vector: {IONQ: +1.8, RGTI: -0.7, QBTS: -0.3, ...}
    + event_type, time_horizon, signal_decay, reasoning
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Evaluation Pipeline (Alphalens + custom)        │
│  Computes IC, CAR, signal decay vs actual prices│
└─────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────┐
│  Gradio Frontend (3 tabs)                        │
│  Signal Explorer | Evaluation Dashboard | Map   │
└─────────────────────────────────────────────────┘
```

## Quantum Computing Universe (9 Tickers)

| Ticker | Company | Technology | Quantum Revenue |
|--------|---------|-----------|-----------------|
| IONQ | IonQ | Trapped Ion | 100% |
| RGTI | Rigetti Computing | Superconducting | 100% |
| QBTS | D-Wave Quantum | Quantum Annealing | 100% |
| QUBT | Quantum Computing Inc. | Neutral Atom | 100% |
| IBM | IBM | Superconducting | ~2% |
| GOOGL | Alphabet/Google | Superconducting | <0.1% |
| MSFT | Microsoft | Topological | <0.1% |
| HON | Honeywell (Quantinuum) | Trapped Ion | ~5% |
| NVDA | NVIDIA | Adjacent/Enabler | ~1% |

## Infrastructure

| Service | Purpose | Credentials |
|---------|---------|-------------|
| GitHub | Development repo (`basilwong/quantum-alpha-intelligence`) | PAT with repo scope |
| HF Space | Public deployment (`build-small-hackathon/quantum-alpha-intelligence`) | Write token |
| HF Hub | Model hosting (`basilwong/quantum-alpha-qwen3-8b`) | Write token |
| Modal | GPU compute (fine-tuning + batch inference) | Workspace: `ac-PGYLNihy2INHkVQupXFTUV` |
| Qwen Cloud | Teacher model (qwen3-max, Singapore free tier) | DashScope API key |
| Yahoo Finance | Market data (free, no key needed) | N/A |

## Current Status (Updated June 8, 2026)

### Completed

| Phase | Status | Key Outcome |
|-------|--------|-------------|
| Phase 1: Signal Vector Schema | DONE | 199/200 training examples generated with cross-sectional schema |
| Phase 2: Model Retraining | DONE | Qwen3-8B fine-tuned on V2 schema, loss 1.36, pushed to HF Hub |
| Phase 4: Market Data | DONE | 608 trading days for all 10 tickers (Jan 2024 - Jun 2026) |
| Phase 3: Batch Predictions | IN PROGRESS | 49/411 successful so far, retry running with cleaned inputs |

### In Progress

**Phase 3 (Batch Predictions)** is running on Modal (detached). The second run includes:
- HTML/URL stripping from article inputs (fixes JSON parse errors)
- Resume support (carries forward 49 previous successes)
- Per-article timing data for performance analysis
- 3-hour timeout

Run URL: https://modal.com/apps/basilwong/main/ap-ylQ60BU6z3Od8hlXUirIS6

### Lessons Learned

1. **Modal timeout issues**: `modal run` ties the remote function's lifecycle to the local client. If the sandbox connection drops, the app stops. Solution: use `modal run --detach` and make the function fully self-contained (reads from volume, writes to volume).

2. **JSON parse errors (21% failure rate on first run)**: The fine-tuned model produces malformed JSON for ~20% of articles. Root cause: many RSS-sourced articles contain HTML tags and Google News redirect URLs that confuse the model. Solution: strip HTML/URLs before inference. Longer-term: more training data or constrained decoding.

3. **Training data quality**: Some articles in our dataset are not actually about quantum computing (arXiv search was too broad) and some are just headlines with URLs (RSS artifacts). These are kept in the evaluation set intentionally to test whether the model correctly identifies irrelevant content.

4. **Speed**: Processing 411 articles on A100 takes ~60-90 minutes (8-15 seconds per article depending on input length). The model generates ~500-1000 output tokens per article for the full signal vector schema.

## Remaining Roadmap

### Phase 3 (Completion): Batch Predictions

Currently running. When complete:
- Download results: `modal volume get quantum-alpha-outputs predictions_v2_final.jsonl data/eval/predictions_v2_final.jsonl`
- Download timing: `modal volume get quantum-alpha-outputs prediction_timing.jsonl data/eval/prediction_timing.jsonl`
- Analyze error rate and timing distribution

### Phase 5: Evaluation Metrics

**Objective**: Compute IC, signal decay, and all evaluation metrics using Alphalens and custom code.

| Step | Task | Estimated Time |
|------|------|----------------|
| 5.1 | Install alphalens-reloaded, format predictions as factor panel | 30 min |
| 5.2 | Compute abnormal returns (custom OLS market model, 180-day estimation window) | 30 min |
| 5.3 | Run alphalens: IC analysis, quantile returns, forward return spreads | 10 min |
| 5.4 | Compute signal decay curve (IC at horizons 1, 2, 5, 10, 20, 60 days) | 10 min |
| 5.5 | Compute IC by subset (source type, ticker, event type) with Bonferroni correction | 15 min |
| 5.6 | Bootstrap confidence intervals (1000 resamples) | 5 min |
| 5.7 | Compute direction accuracy, magnitude calibration, cross-asset accuracy | 20 min |
| 5.8 | Generate all visualization charts (Plotly) | 30 min |
| 5.9 | Save structured results to `data/eval/results.json` | 5 min |

**Dependencies**: Phase 3 complete + Phase 4 complete (both done or in progress)

### Phase 6: Frontend Build

**Objective**: Build the three-tab Gradio interface per the V2 design spec (`docs/frontend_design_v2.md`).

| Step | Task | Estimated Time |
|------|------|----------------|
| 6.1 | Build Tab 1: Signal Explorer - event navigation (timeline slider + filters) | 2 hours |
| 6.2 | Build Tab 1: Signal vector bar chart (hero element) | 1.5 hours |
| 6.3 | Build Tab 1: Predicted vs Actual time series overlay (Plotly) | 2 hours |
| 6.4 | Build Tab 1: Event metrics cards + forward returns table | 1 hour |
| 6.5 | Build Tab 1: Model reasoning trace (expandable) | 30 min |
| 6.6 | Build Tab 1: Live analysis mode (paste text, run inference) | 1.5 hours |
| 6.7 | Build Tab 1: Model selector toggle (base vs fine-tuned) | 1 hour |
| 6.8 | Build Tab 2: Evaluation Dashboard (summary metrics, decay curve, IC by subset, scatter) | 2.5 hours |
| 6.9 | Build Tab 3: Sector Map (network graph + signal propagation simulator) | 2 hours |
| 6.10 | Add info tooltips (i buttons) to all metrics and charts | 1 hour |
| 6.11 | End-to-end testing | 1 hour |

**Total estimated: ~16 hours**

### Phase 7: Deployment and Submission

| Step | Task | Estimated Time |
|------|------|----------------|
| 7.1 | Push final app + pre-computed data to HF Space | 15 min |
| 7.2 | Verify ZeroGPU works for live analysis mode | 30 min |
| 7.3 | Record 2-minute demo video | 1 hour |
| 7.4 | Write social media post (required for submission) | 15 min |
| 7.5 | Write Field Notes blog post (for Well-Tuned badge) | 1.5 hours |
| 7.6 | Final submission to Build Small hackathon | 10 min |

### Phase 8: Qwen Cloud Hackathon (Post June 15)

| Step | Task | Estimated Time |
|------|------|----------------|
| 8.1 | Experiment: DoRA fine-tuning, compare IC vs LoRA | 2 hours |
| 8.2 | Experiment: Qwen3-32B fine-tuning (final version) | 4 hours |
| 8.3 | Experiment: Curriculum ordering (simple to complex) | 2 hours |
| 8.4 | Add persistent memory layer (Qwen3.7-Max + vector DB on Alibaba Cloud) | 6 hours |
| 8.5 | Deploy full backend on Alibaba Cloud ECS | 3 hours |
| 8.6 | Architecture diagram + public repo + demo video | 2 hours |
| 8.7 | Submit to Qwen Cloud hackathon (July 9 deadline) | 1 hour |

## Resource Budget (Updated)

| Resource | Budget | Used So Far | Remaining |
|----------|--------|-------------|-----------|
| Modal credits | $280 | ~$8 (training runs + batch predictions) | ~$272 |
| Qwen Cloud free tier | 1M tokens | ~800K (V2 training data generation) | ~200K |
| HF ZeroGPU | $20 credits | ~$0.50 (initial app testing) | ~$19.50 |
| Time to Build Small deadline | 7 days (June 8 - June 15) | Day 1 in progress | 6 days |

## Key Files and Directories

```
quantum-alpha-intelligence/
├── app.py                              # Gradio app entry point (current: basic, needs V2 rebuild)
├── ROADMAP.md                          # This file
├── TODO.md                             # Task checklist
├── requirements.txt                    # Python dependencies
├── src/
│   ├── config.py                       # Tickers, model config
│   ├── sector_data.py                  # Technology clusters, revenue exposures, competitive relationships
│   ├── model/inference.py              # Model loading and inference wrapper
│   └── api/app.py                      # Original Gradio Server app (to be replaced by V2)
├── eval/
│   ├── market_data.py                  # Yahoo Finance data provider + sector basket
│   └── (abnormal_returns.py)           # To be built (Phase 5)
├── scripts/
│   ├── generate_training_data_v2.py    # Teacher model pipeline (V2 signal vector schema)
│   ├── generate_predictions_v2.py      # Batch prediction on Modal (cleaned inputs, timing)
│   ├── modal_finetune.py               # Fine-tuning script (Unsloth + QLoRA on Modal)
│   ├── test_finetuned_model.py         # Inference test (3/3 correct on V1)
│   ├── collect_historical_articles.py  # Article collection (arXiv + RSS)
│   └── analyze_eval_articles.py        # Data quality analysis
├── data/
│   ├── raw/articles.jsonl              # 611 raw articles (Aug 2024 - Jun 2026)
│   ├── training/
│   │   ├── quantum_alpha_train.jsonl   # V1 training data (199 examples, old schema)
│   │   └── quantum_alpha_train_v2.jsonl # V2 training data (199 examples, signal vector)
│   ├── eval/
│   │   ├── predictions_v2.jsonl        # First run results (49 success, 13 errors)
│   │   ├── predictions_v2_final.jsonl  # (In progress) Cleaned retry results
│   │   └── problematic_articles.json   # Articles flagged for quality issues
│   └── market/
│       ├── IONQ.parquet                # Historical prices (608 trading days)
│       ├── RGTI.parquet
│       ├── ... (all 9 tickers + SPY)
│       └── SPY.parquet
├── docs/
│   ├── end_to_end_design.md            # System architecture
│   ├── frontend_design_v2.md           # V2 frontend spec (3 tabs)
│   ├── evaluation_pipeline_design.md   # Evaluation engineering design
│   ├── quantitative_evaluation_deep_dive.md  # AR, IC, signal decay methodology
│   ├── library_background_alphalens_eventstudy.md  # Library analysis
│   ├── research_prior_art_report.md    # Academic literature review
│   ├── evaluation_methodology_report.md # Evaluation framework
│   ├── hackathon_opportunities_report.md # Multi-hackathon strategy
│   └── huggingface_space_setup.md      # Deployment guide
└── frontend/                           # Original custom HTML/CSS/JS (to be replaced)
```

## Hackathon Submission Requirements

### Build Small (June 15)

- Gradio app hosted on HF Space under `build-small-hackathon` org
- Model must be 32B parameters or fewer
- 2-minute demo video
- Social media post
- Badges targeting: Well-Tuned, Off-Brand, Field Notes

### Qwen Cloud Global (July 9)

- Must use Qwen models via Qwen Cloud (DashScope API)
- Must deploy backend on Alibaba Cloud infrastructure
- Public repo with license file
- Architecture diagram
- Demo video under 3 minutes
- Track: Memory Agent (persistent memory that accumulates domain expertise)

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| Jun 5 | Base model: Qwen3-8B | Best zero-shot baseline in benchmarks, strong financial sentiment |
| Jun 5 | Fine-tuning: QLoRA rank 64 | Balance between quality and VRAM usage on A100 40GB |
| Jun 6 | Teacher model: qwen3-max (Singapore free tier) | Free, strongest model available, no API cost |
| Jun 7 | Output schema: cross-sectional signal vector | Required for Alphalens evaluation, more realistic for quant integration |
| Jun 8 | Evaluation: Alphalens + custom abnormal returns | Alphalens is industry standard for IC; custom code for CAR avoids GPL |
| Jun 8 | Input cleaning: strip HTML/URLs | Fixes 21% JSON parse failure rate from RSS artifacts |
| Jun 8 | Keep non-quantum articles in eval set | Tests model's ability to correctly identify irrelevant content |
| Jun 8 | Modal --detach for long runs | Prevents sandbox timeout from killing remote GPU jobs |


## Next Iteration (Planned)

The following improvements are planned based on evaluation results and external review feedback. They are ordered by priority.

### Priority 1: Methodological Fixes

These address issues that could affect the validity of our evaluation results.

| Item | Issue | Fix | Effort |
|------|-------|-----|--------|
| Temporal split | Training data is newer than evaluation data (inverted walk-forward) | Re-sort articles chronologically, use date-based cutoff (train on pre-2026, eval on 2026) | 2 hours (re-sort + retrain + re-evaluate) |
| Outcome contamination | Some training articles contain explicit price movements ("stock rose 8%") | Audit training data, strip sentences containing price outcomes | 1 hour |
| Narrow eval window | Evaluation predictions cluster in 12 calendar days (May 10-22, 2026) | Collect articles with broader date distribution, ensure eval spans full 2-year range | 2 hours |

### Priority 2: Result Improvements

These would improve the quality and credibility of results without changing the core methodology.

| Item | Current State | Improvement | Effort |
|------|--------------|-------------|--------|
| JSON compliance | 7% error rate | More training data (500+ examples) or constrained decoding | 3 hours |
| Institutional metrics | IC + direction accuracy only | Add factor turnover and cross-sectional dispersion | 30 min |
| Training data volume | 199 examples | Scale to 500-1000 examples for better generalization | 2 hours (Qwen Cloud tokens needed) |
| arXiv signal quality | IC = -0.013 (not working) | Investigate why academic papers don't produce useful signals; may need different prompting | 2 hours |

### Priority 3: Experiments (Qwen Cloud Hackathon)

| Experiment | Hypothesis | Effort |
|-----------|-----------|--------|
| DoRA fine-tuning | May outperform LoRA on structured output tasks | 2 hours |
| Qwen3-32B fine-tuning | Larger model = higher IC, better JSON compliance | 4 hours |
| Curriculum ordering | Training on simple articles first, complex later, improves final quality | 2 hours |
| Persistent memory (Qwen3.7-Max + vector DB) | Agent that accumulates sector expertise over time produces better signals | 6 hours |
| Signal smoothing (Kalman filter) | Reduces noise in multi-article-per-day scenarios | 1 hour |

### Priority 4: Submission Materials

| Item | Status | Deadline |
|------|--------|----------|
| Deploy final V2 app to HF Space | Frontend built, needs deployment with real eval data | June 14 |
| Demo video (2 min) | Not started | June 15 |
| Social media post | Not started | June 15 |
| Field Notes blog post | Not started | June 15 |
| Qwen Cloud architecture diagram | Not started | July 9 |
| Qwen Cloud demo video (3 min) | Not started | July 9 |
