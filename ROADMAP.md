# Alpha Signal Analysis Platform: Roadmap

## Project Overview

Alpha Signal Analysis is an NLP-driven alpha signal generator for the quantum computing sector. It uses fine-tuned small language models to analyze news, research papers, and press releases, producing cross-sectional trading signals across 9 public quantum computing companies simultaneously. The platform compares multiple models and fine-tuning approaches to demonstrate that domain-specific fine-tuning on a small model can match or exceed larger general-purpose models.

**Hackathon Submissions:**
1. **Build Small** (Hugging Face/Gradio) — Deadline: June 15, 2026
2. **Qwen Cloud Global AI Hackathon** (Memory Agent track) — Deadline: July 9, 2026

---

## Current Status (June 10, 2026)

### Completed

| Component | Details |
|-----------|---------|
| Data collection | 611 articles (Aug 2024 - Jun 2026), split into train (190) and eval (421) |
| Walk-forward split | Train on 2024-2025, eval on 2026 (proper temporal ordering) |
| Training data V1 | 386 examples from qwen3-max teacher (used for initial fine-tuning) |
| Model fine-tuning V1 | Qwen3-8B, QLoRA rank 64, loss 1.095, pushed to HF Hub |
| Batch predictions (fine-tuned) | 414/421 success (98.3%) via Modal A100 batches |
| API predictions (base 8B) | 421/421 success via Qwen Cloud free tier |
| API predictions (Max) | 421/421 success via Qwen Cloud free tier |
| API predictions (30B thinking) | In progress via Qwen Cloud |
| Multi-model evaluation | IC comparison across 3 models complete |
| Market data | 608 trading days for 10 tickers (Yahoo Finance) |
| V2 Frontend | 3-tab Gradio app with model selector, deployed to HF Space |
| Evaluation pipeline | Custom abnormal returns + IC computation |

### Key Results (V1 Evaluation)

| Model | IC +5d | IC +10d | Dir Acc +5d |
|-------|--------|---------|-------------|
| **Qwen3-8B Fine-tuned** | **+0.078*** | -0.013 | **55.4%** |
| Qwen3-8B Base | +0.015 | -0.031 | 53.7% |
| Qwen3.7-Max Base | +0.028 | +0.034 | 52.1% |

The fine-tuned 8B model outperforms both the base 8B and the much larger Max model at the +5d horizon. This validates the "Build Small" thesis.

---

## Next Phase: High-Quality Training Data via Manus

### The Strategy Shift

Based on the fine-tuning research report, our current 386 training examples are at the minimum threshold. The literature recommends 800-1,000 examples for robust domain-specific fine-tuning. More importantly, **quality matters more than quantity**: rich reasoning traces from a strong teacher produce significantly better student models than shallow outputs from a weaker teacher.

We are switching from the Qwen Cloud API (qwen3-max) to **Manus as the teacher model**. Manus provides:
- Web browsing for deep research on each article
- Multi-step reasoning with tool use
- Access to frontier models (Claude, GPT-5.5)
- Structured output with guaranteed schema compliance
- Highest possible quality per training example

### Dataset Composition (~1,000 examples)

| Category | Count | Purpose |
|----------|-------|---------|
| Real articles (with web research) | 190 | Core domain knowledge |
| Multi-turn follow-ups | 170 | Teaches reasoning about own outputs |
| Synthetic articles | 200 | Covers scenarios not in real data |
| Paraphrased articles | 190 | Content > style invariance |
| Negative examples | 150 | Prevents hallucinated signals |
| Edge cases | 100 | Teaches measured uncertainty |
| **Total training** | **~1,000** | |
| Evaluation predictions | 421 | Walk-forward IC measurement |

### Execution

The Manus teacher pipeline runs concurrently (10-50 tasks simultaneously) using the Manus API with `agent_profile: "max"`. Estimated runtime: 3-12 hours depending on concurrency level.

Prompt file: `/home/ubuntu/manus_teacher_pipeline_prompt.md`

---

## Prize-Targeted Stretch Goals

Based on analysis of the Build Small hackathon tracks and awards:

### High Priority (Highest prize-to-effort ratio)

| Goal | Targets | Prize Potential | Effort |
|------|---------|----------------|--------|
| Fine-tune MiniCPM-2B (OpenBMB) | OpenBMB Award + Tiny Titan | $4,000 | 3 hours |
| Run GPT-5.5 on eval set (Batch API) | OpenAI Track | $5,000 | 1 hour |
| Migrate to `gr.Server` | Off-Brand award + badge | $1,500 | 6 hours |
| Demo video + Field Notes blog | Best Demo + Field Notes badge | $2,000 | 3 hours |

### Medium Priority

| Goal | Targets | Prize Potential | Effort |
|------|---------|----------------|--------|
| Publish agent traces on Hub | Sharing is Caring badge | Bonus points | 30 min |
| Stack 4+ badges | Bonus Quest Champion | $2,000 | Cumulative |
| Modal usage documentation | Modal Awards | $10,000 credits | 1 hour |

### Fine-Tuning Experiments (Post Build Small, for Qwen Cloud hackathon)

| Method | Hypothesis | Effort |
|--------|-----------|--------|
| DoRA | May outperform LoRA on structured output | 2 hours |
| GRPO (RL with price reward) | Directly optimizes for IC | 4 hours |
| Higher LoRA rank (128) | More capacity | 2 hours |
| Curriculum ordering | Simple → complex improves quality | 1 hour |
| Qwen3-32B fine-tuning | Larger model, higher IC ceiling | 4 hours |

### Model Comparison (for evaluation dashboard)

| Model | Source | Cost | Status |
|-------|--------|------|--------|
| Qwen3-8B Fine-tuned (LoRA) | Modal | ~$15 | Complete |
| Qwen3-8B Base | Qwen Cloud | $0 | Complete |
| Qwen3.7-Max Base | Qwen Cloud | $0 | Complete |
| Qwen3-30B Thinking | Qwen Cloud | $0 | In progress |
| GPT-5.5 | OpenAI Batch API | ~$11 | Planned |
| MiniCPM-2B Fine-tuned | Modal | ~$2 | Planned |
| Manus Teacher | Manus API | Credits | Planned |

---

## Execution Order (Remaining Work)

### Immediate (Before June 15 deadline)

1. **Manus teacher pipeline** — Generate ~1,000 high-quality training examples (running in separate session)
2. **GPT-5.5 Batch API** — Run 611 articles for OpenAI track comparison + potential teacher data (~$11)
3. **Retrain Qwen3-8B** on Manus teacher data (Modal, ~$3)
4. **Fine-tune MiniCPM-2B** on same data (Modal, ~$2) → OpenBMB + Tiny Titan prizes
5. **Re-run evaluation** on all models with new fine-tuned weights
6. **Update frontend** with new evaluation results and additional models
7. **Deploy final app** to HF Space
8. **Record demo video** (2 min) + write Field Notes blog + social post
9. **Submit** to Build Small hackathon

### Post June 15 (Qwen Cloud hackathon, deadline July 9)

10. Run fine-tuning experiments (DoRA, GRPO, 32B, curriculum)
11. Add persistent memory layer (Qwen3.7-Max + vector DB on Alibaba Cloud)
12. Deploy backend on Alibaba Cloud ECS
13. Architecture diagram + public repo + demo video
14. Submit to Qwen Cloud hackathon

---

## Resource Budget

| Resource | Budget | Used | Remaining | Allocated For |
|----------|--------|------|-----------|---------------|
| Modal credits | $280 | ~$30 | ~$250 | Fine-tuning (retrain + MiniCPM + experiments) |
| Qwen Cloud (qwen3-8b) | 1M tokens | ~421K | ~579K | Done |
| Qwen Cloud (qwen3.7-max) | 1M tokens | ~364K | ~636K | 30B thinking run |
| Qwen Cloud (qwen3-32b) | 1M tokens | Broken | N/A | Bug filed with Alibaba |
| OpenAI credits | $20 | $0 | $20 | GPT-5.5 Batch API (~$11) |
| Manus credits | Unlimited | ~1 task | Unlimited | Teacher pipeline (~1,400 tasks) |
| HF ZeroGPU | $20 | ~$1 | ~$19 | Live inference on Space |

---

## Decision Log

| Date | Decision | Reasoning |
|------|----------|-----------|
| Jun 5 | Base model: Qwen3-8B | Best zero-shot baseline in benchmarks |
| Jun 5 | Fine-tuning: QLoRA rank 64 | Balance between quality and VRAM |
| Jun 6 | Teacher model: qwen3-max | Free, strongest available at the time |
| Jun 7 | Cross-sectional signal vector schema | Required for Alphalens, realistic for quant |
| Jun 8 | Custom abnormal returns + IC evaluation | Alphalens for IC; custom OLS for CAR |
| Jun 9 | Fix temporal split (walk-forward) | External review identified inverted ordering |
| Jun 10 | Multi-model comparison | Demonstrates fine-tuning value |
| Jun 10 | Batch-based Modal runs | Prevents losing progress on timeout |
| Jun 10 | qwen3-32b broken, use 30b-thinking instead | Bug filed, model returns empty responses |
| Jun 10 | **Switch teacher to Manus API** | Highest quality (web research + reasoning + frontier models) |
| Jun 10 | **Scale to ~1,000 training examples** | Research report recommends 800-1,000 for robust fine-tuning |
| Jun 10 | **Data augmentation via Manus** | Synthetic, paraphrased, negative, edge cases, multi-turn |
| Jun 10 | **Concurrent Manus tasks** | 10-50x speedup over sequential processing |
| Jun 10 | **Target OpenBMB + Tiny Titan** | Fine-tune MiniCPM-2B for $4,000 in prizes |
| Jun 10 | **Target OpenAI Track** | GPT-5.5 Batch API for $5,000 prize comparison |

---

## Known Limitations (To Document in Field Notes)

1. Evaluation window is Jan-Jun 2026 only (single market regime)
2. Daily granularity (no intraday timing)
3. Single-factor market model (SPY only, no sector factor)
4. Correlation does not imply causation
5. No transaction cost modeling
6. Small sample size limits statistical power for subset analyses
7. Model inference takes ~60s per article on GPU (not real-time without optimization)
8. qwen3-32b API is broken (filed bug report with Alibaba)
