---
title: Alpha Signal Analysis
emoji: ⚛️
colorFrom: gray
colorTo: gray
sdk: gradio
sdk_version: 6.16.0
python_version: '3.12'
app_file: app.py
pinned: true
license: mit
short_description: analysis of mid-frequency alpha signal predictions
tags:
  - gradio
  - build-small-hackathon
  - backyard-ai
  - nvidia
  - nemotron
  - modal
  - codex
  - off-brand
  - track:backyard
  - sponsor:nvidia
  - sponsor:modal
  - sponsor:openai
  - achievement:offbrand
---

# ⚛️ Alpha Signal Analysis

## 📌 TL;DR

[Citadel GQS](https://www.citadel.com/what-we-do/global-quantitative-strategies/) recently [announced they are actively looking to purchase alternative signal sources](https://finance.yahoo.com/markets/stocks/articles/citadel-set-pay-trading-ideas-094103745.html). This project proved that smaller fine-tuned models can produce commercially viable signal quality, performing significantly better than larger general purpose models at this task. 

2 Components:

1. OpenReasoning-Nemotron family of models fine tuned to generate alpha signal for a niche sector of equities. 

2. UI for signal analysis and debugging. The dashboard tracks predictive power across model iterations, measures Information Coefficient against realized market returns, and provides live inference. Paste any article, compare model versions side by side against actual price movements, and inspect the full reasoning trace behind every score.

---

## 🔗 Submission Links

| Resource | Link |
|----------|------|
| Live Demo | [Alpha Signal Analysis Space](https://build-small-hackathon-alpha-signal-analysis.hf.space) |
| Demo Video | [TBD] |
| Social Media Post | [TBD — Substack] |
| GitHub Repository | [TBD] |

---

## ✅ Prize Submissions

| Prize/Track | Why We Chose This |
|-------|---------------|
| **NVIDIA Nemotron Hardware Prize** | We needed reasoning to improve signal quality. This meant Nemotron-Reasoning family was the natural choice, and the results were: 2x signal strength over non-reasoning alternatives. |
| **Best Use of Modal** | The project required dozens of rapid GPU bursts for training and batch inference. Modal's serverless model let us run 30+ hours of A100 compute for under $150 total with a 90x inference speedup. Over $170 credits spent in 5 days. 100+ total app runs. |
| **Best Use of Codex** | We used Codex as the teacher model in a knowledge distillation pipeline. Hundreds of individual Codex sessions analyzed articles and generated structured training data, closing the loop between evaluation results and data generation. |

---

### <img src="assets/nvidia_logo.png" height="30"> NVIDIA Nemotron Hardware Prize

Our production model is **OpenReasoning-Nemotron-7B**. We chose Nemotron because this project requires a model that reasons before it scores. Financial signal generation demands that the model assess competitive dynamics, weigh conflicting implications, and commit to a reasoning chain before producing numeric scores. Nemotron's native `<think>` block architecture enables our GRPO training loop: the model explores different reasoning paths, and reinforcement learning selects the paths that correlate with actual market returns.

**Result:** IC of +0.157 (p=0.006) across all horizons. Qwen3-8B achieved +0.078 at 5 days that degraded to zero at longer horizons. Nemotron produced 2x the signal strength with consistency at 1, 5, 10, and 20 days. Full comparison in [All Model Results](#all-model-results).

---

### <img src="assets/modal_logo.png" height="30"> Best Use of Modal

Modal is the computational backbone of this project. Every training run, batch inference job, and experimental iteration ran on Modal A100 GPUs. The serverless model was essential because the project required dozens of short, high intensity compute bursts rather than persistent allocation. Over $170 in credits spent across 100+ app runs in 5 days.

| Metric | Before | After |
|--------|--------|-------|
| Prediction speed | 54 sec/article | 0.6 sec/article |
| Full evaluation (421 articles) | 6+ hours (timed out) | 4.4 minutes |
| Cost per evaluation | ~$25 | ~$0.12 |

This efficiency made the entire V7 experimental cycle possible. Details in [Understanding Why Everything Was Slow](#understanding-why-everything-was-slow) and [The Fine Tuning Journey](#the-fine-tuning-journey-from-timeouts-to-signal).

---

### <img src="assets/openai_logo.png" height="30"> Best Use of Codex

Codex (GPT-5.5) was used as the **teacher model in a knowledge distillation pipeline** for generating financial training data. A programmatic loop iterated over hundreds of input articles, spawning individual Codex sessions that performed deep analytical work: reading articles, researching companies, assessing competitive dynamics, and producing structured signal vectors with reasoning traces.

The unique contribution is the closed loop: Codex analyzed evaluation results from previous model versions (identifying anti-predictive tickers, diagnosing distribution mismatches), then those insights directly shaped what training data Codex generated next. The coding agent became a data scientist. Full details in [The Training Data Strategy](#the-training-data-strategy-a-story-of-iterative-refinement) and [Why Manus as the Teacher](#why-manus-as-the-teacher).

<!-- TODO: Add links to relevant Codex-attributed commits in GitHub repo once repo is public -->

---

## Table of Contents

1. [The Hypothesis](#the-hypothesis)
2. [Why Quantum Computing Specifically](#why-quantum-computing-specifically)
3. [The Academic Foundation](#the-academic-foundation)
4. [System Architecture](#system-architecture)
    - [Data Ingestion Pipeline](#data-ingestion-pipeline)
    - [Signal Generation Model](#signal-generation-model)
    - [Evaluation Framework](#evaluation-framework)
    - [Interactive Dashboard](#interactive-dashboard)
5. [Technology Stack](#technology-stack)
6. [The Training Data Strategy: A Story of Iterative Refinement](#the-training-data-strategy-a-story-of-iterative-refinement)
    - [Version 3: The First Structured Dataset (187 Examples)](#version-3-the-first-structured-dataset-187-examples)
    - [Version 4: Scale, Quality, and Empirical Grounding (881 Examples)](#version-4-scale-quality-and-empirical-grounding-881-examples)
    - [Version 5: Reasoning Traces for Thinking Models (1,121 Examples)](#version-5-reasoning-traces-for-thinking-models-1121-examples)
    - [Version 5.1: Directional Balance (Bearish Examples)](#version-51-directional-balance-bearish-examples)
    - [Version 5.2: Robustness Training (Drawdown, Sideways, Conflicting)](#version-52-robustness-training-drawdown-sideways-conflicting)
    - [Cross Version Summary](#cross-version-summary)
7. [The Fine Tuning Journey: From Timeouts to Signal](#the-fine-tuning-journey-from-timeouts-to-signal)
    - [The Naive Approach: Qwen3-8B and the First Training Run](#the-naive-approach-qwen3-8b-and-the-first-training-run)
    - [Understanding Why Everything Was Slow](#understanding-why-everything-was-slow)
    - [The Evaluation Framework in Detail](#the-evaluation-framework-in-detail)
    - [All Model Results](#all-model-results)
        - [Fine Tuned Models (OpenReasoning-Nemotron-7B)](#fine-tuned-models-openreasoning-nemotron-7b)
        - [Base Models (No Fine Tuning)](#base-models-no-fine-tuning)
        - [Teacher Models (Direct Predictions)](#teacher-models-direct-predictions)
    - [What Each Model Version Revealed](#what-each-model-version-revealed)
    - [The Training Approaches Explained](#the-training-approaches-explained)
    - [The Overfitting Mistake and Recovery](#the-overfitting-mistake-and-recovery)
    - [What the Signal Actually Looks Like](#what-the-signal-actually-looks-like)
        - [Why Manus as the Teacher](#why-manus-as-the-teacher)
8. [Lessons Learned](#lessons-learned)
    - [From Training Data Development](#from-training-data-development)
    - [From Fine Tuning](#from-fine-tuning)
9. [Frontend Engineering: Deploying a Custom Trading Terminal on Hugging Face Spaces](#frontend-engineering-deploying-a-custom-trading-terminal-on-hugging-face-spaces)
    - [The Port Binding Problem](#the-port-binding-problem)
    - [The Solution: gradio.Server Mode](#the-solution-gradioserver-mode)
    - [The Frontend Architecture](#the-frontend-architecture)
    - [The Trading Terminal Interface](#the-trading-terminal-interface)
    - [Final Architecture and Deployment](#final-architecture-and-deployment)
10. [Conclusion](#conclusion)
11. [References](#references)

---

## The Hypothesis

Most retail investors and even many institutional analysts fundamentally misunderstand the quantum computing sector. When IonQ announces "35 algorithmic qubits," the market reaction depends entirely on whether participants understand the distinction between physical qubits, logical qubits, and algorithmic qubits, and whether they can contextualize that number against the company's stated roadmap. When a research group publishes a paper demonstrating error correction below the fault tolerance threshold on a superconducting architecture, the implications for Rigetti versus IonQ versus D-Wave are non obvious to anyone without a physics background.

This creates an information asymmetry that persists longer than in most sectors. In traditional equity markets, earnings surprises get priced in within minutes. In quantum computing, a technical milestone announcement can take days or weeks to be fully reflected in stock prices because the analyst community lacks the domain expertise to rapidly assess significance. Lopez-Lira and Tang (2024) documented this phenomenon empirically: GPT-4 sentiment scores predict next day stock returns with a Sharpe ratio of 3.8, and crucially, predictability is strongest for smaller stocks and low readability (highly complex) news [1]. Quantum computing sits squarely in this sweet spot: small cap stocks, highly technical announcements, and a market that struggles to parse the signal from the noise.

Our hypothesis is straightforward: a small language model, fine tuned specifically on quantum computing financial analysis, can produce trading signals that outperform both larger general purpose models and the base model without fine tuning. The key insight is that domain specific fine tuning on a narrow task can compensate for raw model scale. This is the "Build Small" thesis: you do not need 200 billion parameters to extract alpha from quantum computing news if you have the right training data and the right evaluation framework.

## Why Quantum Computing Specifically

The quantum computing sector has several properties that make it uniquely suited to NLP based alpha generation.

**Slow information diffusion.** Unlike earnings reports or macroeconomic data (which get priced in within seconds by algorithmic traders), quantum computing breakthroughs propagate slowly through the market. A paper published on arXiv about logical qubit error rates might take 3 to 10 trading days to be fully reflected in stock prices because most market participants cannot assess its significance without domain expertise. Truong (2025) documented this pattern explicitly: industry specific sentiment reveals unique opportunities due to gradual information diffusion within specialized sectors [2].

**High technical complexity creates persistent mispricing.** The quantum computing sector has at least five distinct technological approaches (trapped ion, superconducting, quantum annealing, topological, neutral atom), each with different strengths, timelines, and commercial viability. A breakthrough in one approach has asymmetric implications across the competitive landscape. For example, a superconducting error correction advance is bullish for Rigetti and IBM but bearish for IonQ and Honeywell (trapped ion competitors). Most investors do not understand these competitive dynamics, creating persistent mispricing that a domain expert model can exploit.

**Small universe with clear competitive structure.** The publicly traded quantum computing universe consists of 10 companies, ranging from pure play quantum firms (IonQ, Rigetti, D-Wave, Quantum Computing Inc., Quantinuum via Honeywell) to diversified technology companies with quantum divisions (IBM, Google, Microsoft, NVIDIA). This small, well defined universe makes cross sectional analysis tractable: every piece of news can be scored against every company simultaneously, producing a complete signal vector rather than a single stock sentiment label.

**Multiple information sources with varying lead times.** Quantum computing information flows through a predictable pipeline: academic preprints on arXiv (earliest signal, often weeks before press coverage), company press releases, SEC filings, financial news articles, and social media commentary. A system that ingests at the source (arXiv) can capture signals before they reach mainstream financial media.

## The Academic Foundation

This project builds on several established research threads that collectively provide the theoretical and empirical justification for the approach.

**LLMs for financial sentiment prediction.** Lopez-Lira and Tang (2024) demonstrated that large language models can predict stock returns from news headlines with statistical significance. Their key finding, that predictive power is strongest for complex, low readability text, directly motivates our focus on technical quantum computing content [1]. Basic models like BERT failed at this task, suggesting that the ability to translate technical language into financial implications is an emergent capability of larger models, or in our case, a capability that can be instilled through domain specific fine tuning.

**Teacher student distillation for financial NLP.** The FinGPT project (Wang et al., 2023) established the paradigm of using instruction tuning with LoRA to adapt general purpose LLMs to financial tasks [3]. Their approach, which we follow, uses a larger "teacher" model to generate training labels and then trains a smaller "student" model to reproduce those outputs. The Orca paper (Mukherjee et al., 2023) showed that the quality of the reasoning traces in the training data matters more than the volume: a 13B model trained on rich explanations from GPT-4 achieved remarkable reasoning capabilities, outperforming models trained on shallow outputs [4].

**Cross sectional signal generation.** The GPT-Signal paper (Wang et al., 2024) demonstrated that LLMs can generate novel return predictive formulaic alphas across sectors (IT, Healthcare, Energy) that consistently outperform baseline signals over a 5 year backtest [5]. TradExpert (Ding et al., 2025) proved that a Mixture of Experts approach, using specialized LLMs for different information types, outperforms single model systems [6]. Our architecture draws from both: we produce cross sectional signals (scoring all companies simultaneously) and use specialized prompting for different source types (news, arXiv, SEC filings).

**Event study methodology for signal evaluation.** Rather than relying on simple directional accuracy (which can be misleading), we evaluate our signals using the Information Coefficient (Spearman rank correlation between predicted signals and realized abnormal returns), following the standard quantitative finance methodology described in Grinold and Kahn's "Active Portfolio Management" [7]. This approach controls for market movements using a factor model and measures whether the model's relative rankings of stocks correlate with their subsequent relative performance.

## System Architecture

The Alpha Signal Analysis platform consists of four major components that work together to ingest information, generate signals, evaluate performance, and present results interactively.

### Data Ingestion Pipeline

The system collects quantum computing related content from multiple sources. ArXiv preprints from the quant-ph and cs.ET categories represent the earliest possible signal, often published weeks before press coverage. The system monitors for papers mentioning quantum computing companies or relevant technical milestones. Financial news via RSS feeds from Google News and Yahoo Finance provides real time coverage of business events, partnerships, earnings, and analyst commentary. SEC filings from EDGAR (10-K, 10-Q, and 8-K filings) for the quantum computing companies capture regulatory disclosures and financial data.

Each article is timestamped, source tagged, and cleaned (HTML stripped, URLs removed) before being passed to the signal generation model.

### Signal Generation Model

The core of the system is a fine tuned OpenReasoning-Nemotron-7B model that takes an article as input and produces a structured JSON output containing several components. The cross sectional signal vector provides a score for every company in the quantum universe (10 tickers), ranging from -2.0 (strongly bearish) to +2.0 (strongly bullish). Scores are scaled by each company's quantum revenue exposure: pure play companies get the full range, while diversified companies like Google are capped at small magnitudes because quantum news has negligible impact on their overall stock price. Of the 10 tickers in the universe, 7 are actively scored (IONQ, RGTI, QBTS, QUBT, IBM, HON, QNT) while 3 are held inactive at 0.0 (MSFT, GOOGL, NVDA) based on empirical evidence that these tickers added noise or were anti-predictive.

The output also includes event classification (what type of event this represents), time horizon (how long the signal is expected to persist), technical translation (a plain language explanation of why this event matters commercially), and a chain of thought reasoning process explaining how the model arrived at each score.

The signal vector is the key innovation. Rather than producing a single sentiment label for a single stock (which is what most financial NLP systems do), we produce a complete cross section that captures the competitive dynamics: a trapped ion breakthrough is simultaneously bullish for IonQ and bearish for Rigetti, with magnitudes determined by the specific nature of the advance.

### Evaluation Framework

The evaluation framework uses established quantitative finance methodology. Cumulative Abnormal Returns (CAR) are computed for each event by estimating the stock's beta to SPY using 180 days of pre-event data (with a 14 day gap to avoid contamination), then computing the difference between actual returns and expected returns (alpha + beta times SPY return) over the forward horizon. This isolates the company specific, idiosyncratic price movement from broader market trends.

The Information Coefficient (IC), defined as the Spearman rank correlation between the model's predicted signal scores and the realized cumulative abnormal returns, serves as the primary evaluation metric. An IC above 0.05 is considered meaningful in quantitative finance; above 0.10 is exceptional. Pearson correlation is also computed and reported as a secondary metric, with both methods showing consistent results.

Signal Decay Analysis computes IC at multiple holding periods (1, 2, 5, 10, 20 trading days) to measure how quickly the signal's predictive power fades. This validates the hypothesis about slow information diffusion in the quantum sector.

Multi model comparison runs the same evaluation on multiple models (fine tuned vs. base, small vs. large, SFT vs. reinforcement learning) to demonstrate that domain specific fine tuning adds measurable value.

### Interactive Dashboard

The platform is deployed as a Hugging Face Space with a custom trading terminal frontend. The dashboard provides live analysis, historical predictions, an evaluation dashboard, and a sector map. The frontend engineering and deployment architecture are detailed in a dedicated section below.

## Technology Stack

The following table summarizes the technology choices across the platform:

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Base model | OpenReasoning-Nemotron-7B | 7B reasoning model with native thinking blocks |
| Fine tuning | QLoRA (r=16, alpha=32) via Unsloth | Scaling factor 2.0, fits on single A100 |
| Training compute | Modal (A100 40GB GPUs) | Serverless GPU for training runs (~$3 per run) |
| Teacher model | Manus 1.6-max (agentic AI with web browsing) | Generates high quality training labels with deep research |
| Batch inference | vLLM with merged weights | PagedAttention, continuous batching, 90x faster than naive |
| Live inference | HF ZeroGPU | On demand GPU allocation for live predictions |
| Evaluation | Custom (scipy, statsmodels, pandas) | CAR computation, IC (Spearman + Pearson), signal decay |
| Frontend | Custom HTML/CSS/JS + Plotly | Trading terminal UI |
| Backend | FastAPI + Gradio Server | API endpoints + GPU inference |
| Deployment | Hugging Face Spaces | Public demo |
| Version control | GitHub (private) | Development workspace |

It is worth noting that both the base model and the teacher model evolved over the course of the project. The initial fine tuning attempt used Qwen3-8B with a suboptimal LoRA configuration (r=64, alpha=16, scaling factor 0.25). After an adversarial review flagged that this configuration effectively dampens LoRA updates by 4x, and after verifying against Unsloth's official hyperparameters guide (which recommends alpha/rank >= 1.0), the project switched to OpenReasoning-Nemotron-7B with r=16, alpha=32 (scaling factor 2.0) for all subsequent production runs.

On the teacher side, early versions (V1 through V3) used Qwen3.7-max via DashScope, which produced the initial 187 training examples. Starting with V4, the project transitioned to Manus 1.6-max, an agentic AI system capable of browsing the web, researching companies, verifying claims, and producing richly reasoned signal vectors. This transition represented a qualitative leap in training label quality.

## The Training Data Strategy: A Story of Iterative Refinement

The research literature is clear: for domain specific fine tuning, quality matters more than quantity, but you need a minimum threshold of examples to achieve robust performance. The LIMA paper (Zhou et al., 2023) showed that 1,000 carefully curated examples can match the performance of models trained on orders of magnitude more data [8]. The Phi series (Gunasekar et al., 2023) proved that "textbook quality" data enables small models to match or outperform models ten times their size [9]. AlpaGasus (Chen et al., 2024) demonstrated that training on 9,000 filtered high quality examples outperformed the full 52,000 Alpaca dataset [10].

What follows is the detailed story of how the training data evolved across five major versions, each addressing specific deficiencies identified through empirical evaluation, adversarial analysis, and domain expertise.

### Version 3: The First Structured Dataset (187 Examples)

V3 was the first dataset in proper chat format (system/user/assistant messages) suitable for fine tuning. It used 9 tickers (IONQ, RGTI, QBTS, QUBT, IBM, GOOGL, MSFT, HON, NVDA) with a basic system prompt and no market context. The teacher model at this stage was Qwen3.7-max via DashScope.

The dataset consisted of 187 examples from real articles spanning August 2024 through December 2025. It included a `signal_decay` field (fast/medium/slow) but had no chain of thought field, no market context, and no validation against actual returns. The system prompt was approximately 3,600 characters with basic scoring guidelines.

When evaluated against actual 5 day forward returns, the results were sobering. The overall IC was 0.055 (barely significant at p=0.03). More troubling, several tickers were actively anti-predictive: NVDA had an IC of -0.175 (p=0.0008), MSFT was -0.033 (pure noise), and GOOGL was -0.023 (pure noise). ArXiv articles as a category had an IC of -0.058, meaning the model's predictions for academic papers were worse than random. Direction accuracy was 53%, barely above a coin flip.

These results, while disappointing on the surface, were enormously informative. They revealed that three of the nine tickers were adding noise or actively degrading the signal, that the model had no ability to assess academic papers, and that the scoring philosophy ("sentiment about quantum") was fundamentally misaligned with what actually moves stock prices.

### Version 4: Scale, Quality, and Empirical Grounding (881 Examples)

V4 was a complete rebuild using the Manus API (manus-1.6-max agent profile) as the teacher, with 16 specific fixes applied based on empirical analysis. The scale increased from 187 to 881 examples, a 370% increase.

The structural fixes were extensive. The `signal_decay` field was removed because it showed no correlation with empirical decay patterns. A `chain_of_thought` field was added for reasoning traces, inspired by the Orca paper's finding that explanation traces improve student model quality. A total of 307 placeholder chain of thought entries (which contained "REDACTED" instead of actual reasoning) were repaired by synthesizing reasoning from the signal rationale. The ticker QNT (Quantinuum) was added as the 10th ticker following its June 2026 IPO.

The ticker universe underwent a fundamental overhaul. MSFT, GOOGL, and NVDA were moved to "inactive" status with hard coded 0.0 scores. The rationale was straightforward: IC analysis proved these tickers added noise or were anti-predictive. Their news still flows through the system and affects active ticker scores (a Google quantum breakthrough still impacts IONQ and RGTI), but the model no longer attempts to predict price movements in these diversified mega-cap stocks based on quantum news alone.

New data categories were introduced: 70 arXiv rebalancing examples (10 important, 45 incremental, 15 unrelated), addressing the distribution mismatch where training was 1% arXiv but evaluation was 37% arXiv; and 35 QNT competitive dynamics examples covering sector wide and zero sum scenarios.

Market context was added to every example in the form of a full table with 5 day returns, 30 day returns, 52 week position, liquidity tier, and market regime. All 187 original real articles were retroactively enriched with this context from parquet data. This was inspired by Jansen's (2020) observation that "alternative data requires context about what's already priced in" [11].

The prompt improvements included a conditional arXiv score cap (0.5 default, 1.0 for company hardware papers), a source aware minimum conviction threshold, a technology validation rule (grounded in the empirical observation that RGTI gained 89% on Google's Willow announcement), and an IONQ-QNT competitive dynamics framework.

The following table summarizes where each improvement came from:

| Fix | Inspiration Source |
|-----|-------------------|
| Remove anti-predictive tickers | IC analysis per Jansen (2020, Ch. 6): "Remove signals that don't predict" |
| ArXiv score cap | Empirical finding: arXiv IC was -0.058. Papers rarely move stocks directly. |
| Market context | Jansen (2020, Ch. 15): "Alternative data requires context about what's already priced in" |
| Chain of thought repair | Orca (Mukherjee et al., 2023): "Rich explanation traces produce superior student models" |
| Technology validation rule | Market data: RGTI +89% on Google Willow (Dec 9, 2024). Empirical ground truth. |
| Minimum conviction (source aware) | Iterative testing: first version over zeroed (IONQ 29% to 71% zero rate). Refined to be source aware. |
| QNT examples | Domain knowledge: Quantinuum IPO on June 4, 2026. Competitive dynamics with IONQ needed explicit training. |
| ArXiv rebalancing | Distribution mismatch: training was 1% arXiv but eval was 37% arXiv. LIMA principle: match eval distribution. |
| 881 examples target | LIMA (Zhou et al., 2023): "1,000 carefully curated examples" is sufficient for instruction tuning |
| Adversarial review process | AlpaGasus (Chen et al., 2024): quality filtering outperforms quantity |

The evaluation results from the V4 prompt improvements alone (applied to the same 421 evaluation articles, without retraining the model) were striking:

| Metric | V3 Baseline | V4 Prompt | Change |
|--------|-------------|-----------|--------|
| Overall IC | +0.055 | +0.093 | +46% relative |
| p-value | 0.03 | 0.0003 | 10x more significant |
| Direction Accuracy | 53.0% | 55.2% | +2.2 percentage points |
| ArXiv IC | -0.058 | +0.037 | Fixed (was anti-predictive) |
| RGTI IC | +0.127 | +0.203 | +60% (p=0.001) |
| IBM IC | -0.061 | +0.050 | Flipped positive |
| HON IC | -0.023 | +0.083 | Flipped positive |

The single most impactful discovery from V4 was that prompt engineering on the teacher model improves signal quality more than scaling data quantity. The 16 fixes collectively improved IC by 46% without changing the model or adding more articles. This aligns with the LIMA finding that "almost all knowledge in large language models is learned during pretraining, and only limited instruction tuning data is necessary to teach models to produce high quality output" [8].

### Version 5: Reasoning Traces for Thinking Models (1,121 Examples)

V5 represents a complete regeneration of all 881 examples with explicit `<think>...</think>` reasoning blocks that drive the scores. This is not a post processing step. The thinking genuinely produces the scores. The format places the reasoning block before the structured JSON output, forcing the model to commit to a reasoning chain before producing scores.

The scoring philosophy was further refined in V5. Scores now explicitly reflect expected stock movement over 5 trading days, grounded in how news changes investor expectations about competitive position and technology validation. The guiding principle is that milestones toward fault tolerant quantum computing are what drive these stocks.

The motivation for V5 comes from the decision to fine tune OpenReasoning-Nemotron-7B, a reasoning model that natively generates `<think>` blocks before responding. Research on chain of thought fine tuning shows three critical findings. First, models trained without reasoning traces learn to skip reasoning, producing answers directly, which degrades calibration and consistency. Second, the Orca insight extends to reasoning models: just as Orca showed that explanation traces improve student quality, reasoning traces in the training data teach the student to reason before scoring. Third, thinking then scoring produces more consistent outputs because when the model must commit to a reasoning chain before producing scores, contradictions between reasoning and scores become visible and self correcting.

All 881 base examples were successfully generated with a 100% success rate after retries. Thinking blocks averaged 187 words (within the target of 100 to 300 tokens). Zero validation issues were found on scores, ranges, or inactive tickers. Total generation time was approximately 4.5 hours using 10 concurrent tasks.

### Version 5.1: Directional Balance (Bearish Examples)

The initial V5 base was 91.4% bullish at the per ticker level. This is natural: most quantum computing articles are about progress and breakthroughs. But a model trained on 91% bullish data becomes a "hype machine" that cannot predict downside. You cannot rely on the natural distribution of news to produce balanced training data for a trading signal model.

To fix this, 172 explicitly bearish examples were generated across nine categories:

| Category | Examples | Purpose |
|----------|----------|----------|
| Earnings misses / guidance cuts | ~30 | Company specific negative fundamentals |
| Technical setbacks | ~22 | Hardware failures, missed milestones |
| Competitive displacement | ~20 | Larger companies winning, classical breakthroughs |
| Capital markets / dilution | ~16 | Secondary offerings, insider selling, going concern |
| Executive departures | ~12 | Key talent leaving, governance issues |
| Negative analyst coverage | ~12 | Short reports, downgrades, bubble narratives |
| Sector selloffs | ~10 | Macro driven, quantum winter, funding failures |
| Regulatory / legal | ~10 | SEC investigations, patent lawsuits, export controls |
| Priced in / overextended | ~25 | Stocks up 40 to 60%, trivial news producing bearish or zero signals |

The "priced in" category deserves special attention. It was inspired by the observation that the model had no training on mean reversion dynamics. Jansen (2020, Ch. 4) notes that "signals must account for what is already reflected in prices" [11]. If a stock is up 50% in a week, bullish news is already priced in and the risk is to the downside.

After adding these examples, 31.1% of all examples contain at least one bearish score (target: 30 to 40%). The model now learns that negative scores are valid, common, and often correct.

### Version 5.2: Robustness Training (Drawdown, Sideways, Conflicting)

A critical gap analysis revealed the model was fragile in drawdowns. It had never seen examples where stocks were already deeply negative. Three categories were added totaling 68 examples.

**Drawdown behavior (28 examples)** covers four scenarios. Continuation: stocks down 20 to 35%, more bad news arrives, the correct output is to stay bearish. Recovery: stocks down 30 to 50%, genuinely good news arrives, the correct output is cautiously bullish. Noise: stocks down 15 to 25%, irrelevant news arrives, the correct output is zero scores (do not react). Macro: broad market crash with quantum stocks caught in liquidation, the correct output is zero (not quantum specific).

**Sideways/choppy market (20 examples)** covers stocks going nowhere for months with incremental news. The correct output is zero or near zero. This teaches the model that "no opinion" is the correct output most of the time in quiet markets.

**Conflicting signals (20 examples)** covers scenarios like "great earnings but dilutive offering," "won a contract but CTO resigned," and "technology validated but commercially unviable at current cost." These teach moderate, balanced scores rather than always going to extremes. Without explicit training on mixed signals, the model learns to always go to extremes. The conflicting signal examples teach that moderate scores (0.3 to 0.8) are often more appropriate than maximum conviction (1.5 to 2.0).

The drawdown scenarios address a known failure mode in quantitative models described in Jansen (2020, Ch. 8): models trained primarily on bull market data fail catastrophically during regime changes [11]. By explicitly training on drawdown behavior, we reduce the probability of the model producing dangerously wrong signals during market stress.

### Cross Version Summary

The following table captures the full evolution of the training data across all versions:

| Dimension | V3 | V4 | V5 |
|-----------|----|----|-----|
| Examples | 187 | 881 | 1,121 |
| Tickers | 9 (all active) | 10 (7 active + 3 inactive) | 10 (7 active + 3 inactive) |
| Reasoning | None | chain_of_thought field | `<think>` block (drives scores) |
| Market context | None | Full table (5d, 30d, 52w, liquidity, regime) | Full table |
| ArXiv handling | Same as news | Capped at 0.5, source aware conviction | Capped, paper must be read |
| Scoring philosophy | "Sentiment about quantum" | "Expected stock movement (empirically validated)" | "Expected stock movement (reasoning driven)" |
| Validation | None | 12 point automated checks | 12 point + thinking quality |
| Teacher model | Qwen3.7-max (DashScope) | Manus 1.6-max (API) | Manus 1.6-max (API) |
| QNT ticker | No | Yes | Yes |
| Tech validation rule | No | Yes (RGTI +89% evidence) | Yes |
| Directional balance | Unknown | 58% bull / 42% bear | 31% of examples have bearish content |
| Drawdown training | No | No | Yes (28 examples) |
| Conflicting signals | No | No | Yes (20 examples) |

## The Fine Tuning Journey: From Timeouts to Signal

With the training data in hand, the next challenge was turning it into a working model. This section documents the iterative process of fine tuning, the infrastructure problems encountered along the way, and the critical discovery that the training approach matters as much as the training data.

### Experimental Design and Overfitting Prevention

Before detailing the iterative process, it is critical to outline the experimental design that ensures these results are robust and not overfit:

- **Strict Train/Test Separation:** 184 training articles with return data were used exclusively for training. The 421 evaluation articles were completely held out and never seen during training or reward computation.
- **Look-Ahead Bias Prevention:** A 14-day gap was enforced between the 180-day beta estimation window and the event date to prevent contamination.
- **Market Neutrality:** CAPM adjustment removes the market component. The model only gets credit for predicting the idiosyncratic, stock-specific return.
- **Two-Tailed Significance:** All reported p-values are two-tailed.
- **Why These Results Are Not Overfit:** GRPO generalizes because it learns reward-maximizing reasoning patterns, not article-specific predictions. An IC of +0.157 at 5 days with N=1540 observations yields a robust p=0.006 (significant even with Bonferroni correction for multiple hypothesis testing across 8 models). Furthermore, the consistency of the signal across all horizons (1, 5, 10, and 20 days are all positive and significant) is strong evidence against overfitting, as overfit models typically show strong signal at one specific horizon and noise at others.

### The Naive Approach: Qwen3-8B and the First Training Run

The project started where most people start: take a capable open source model, fine tune it on the domain data, and deploy it for predictions. The initial model choice was Qwen3-8B, selected for its strong benchmark performance, good support in the Unsloth training framework, and 8 billion parameters as a balance between capability and cost.

The training setup used Modal's serverless GPU platform with a single A100-40GB. The initial QLoRA configuration was r=64, alpha=16 (scaling factor 0.25), which was later identified as suboptimal. Training completed in 36 minutes and produced a final loss of 0.68. So far, so good.

The first problem appeared at inference time. The prediction script loaded the model using the standard HuggingFace approach with `AutoModelForCausalLM` and ran sequential generation, one article at a time. Each article took 54 seconds. The full 421 article evaluation would take over 6 hours. Modal's function timeout killed it at 2 hours.

| Step | Time | Cost |
|------|------|------|
| Training (Qwen3-8B, 881 examples) | 36 minutes | ~$2.50 |
| Prediction per article | 54 seconds | ~$0.03 |
| Full evaluation (421 articles) | 6+ hours (timed out) | ~$25 (if completed) |

### Understanding Why Everything Was Slow

The breakthrough came from reading Modal's blog post on host overhead. The core insight: a modern GPU can complete about one million floating point operations in a nanosecond. Every nanosecond the GPU sits idle waiting for the CPU to decide what to do next wastes a million operations.

The pipeline was the worst case. The combination of `bitsandbytes` dynamic quantization, LoRA adapter computation at runtime, and HuggingFace's sequential `model.generate()` created thousands of synchronization points per token. The GPU was spending more time waiting than computing.

**Fix 1: Merge the LoRA adapter permanently.** Instead of computing adapter activations at runtime, the weights were merged once using standard PEFT. This eliminates the adapter overhead entirely at inference time. The merged model is a standard transformer with no additional computation paths.

**Fix 2: Switch to vLLM for batch inference.** Following Modal's high performance LLM inference guide, the project adopted vLLM for throughput oriented workloads. The key difference: instead of processing articles sequentially, all 421 prompts are submitted simultaneously as a single batch. vLLM's scheduler handles continuous batching internally using PagedAttention (which allocates KV cache in small blocks without fragmentation), automatically frees memory from finished sequences for new sequences, and interleaves prefill and decode operations across all sequences.

**Fix 3: Use the tokenizer's chat template for prompt formatting.** The model was trained with a specific token format produced by `tokenizer.apply_chat_template()`. When the inference prompt does not match this format exactly, the model produces garbage. The project initially used a manually constructed prompt string, which caused 29.5% of predictions to fail JSON parsing. Switching to the tokenizer's native template fixed this.

The combined result was transformative:

| Step | Time | Cost |
|------|------|------|
| Model merge (one time) | 5 minutes | ~$0.50 |
| Prediction per article (vLLM) | 0.6 seconds | ~$0.0003 |
| Full evaluation (421 articles) | 4.4 minutes | ~$0.12 |
| **Improvement factor** | **90x faster** | **200x cheaper** |

### The Evaluation Framework in Detail

Before discussing signal quality, it is important to understand precisely how the evaluation works. The system computes Cumulative Abnormal Returns (CAR) for each prediction, which strips out the market component of stock returns. For each prediction, the process is: estimate the stock's beta to SPY using 180 days of pre-event data (with a 14 day gap to avoid contamination from the event itself), compute the expected return after the event as alpha plus beta times SPY return, then define the abnormal return as actual return minus expected return. The CAR is the sum of abnormal returns over the forward horizon.

This means the IC measures whether the model predicts the idiosyncratic (non-market) component of returns. If the entire quantum sector goes up 10% because the market rallied 8%, and a stock has beta of 1.2, the expected return is approximately 9.6%, and only the residual 0.4% counts as "abnormal." The model only gets credit for predicting the stock specific component.

The primary IC metric is the Spearman rank correlation between predicted signal scores and realized CARs, which is the industry standard for cross sectional signal evaluation in quantitative finance. Pearson correlation is also computed as a secondary metric. Both methods show consistent results across all model versions.

### All Model Results

With inference solved, the full evaluation was run across all model versions. The following table presents the complete results:

| Model | Training Data | IC@1d | IC@5d | IC@10d | IC@20d | Dir Acc | N obs |
|-------|--------------|-------|-------|--------|--------|---------|-------|
| **V7d GRPO (clean)** | 184 train articles w/ returns | **+0.151** | **+0.157** | **+0.160** | **+0.159** | **58.6%** | 1540 |
| V7b Rejection (clean) | Best of 4 from train articles | +0.041 | +0.139 | +0.016 | +0.130 | 55.1% | 938 |
| V7c DPO (clean) | Preference pairs from train | +0.063 | +0.070 | +0.077 | +0.136 | 56.6% | 1444 |
| V4 SFT (no thinking) | 881 teacher labeled articles | +0.032 | +0.075 | +0.128 | +0.136 | 52.9% | 3265 |
| V7a SFT (with thinking) | 881 articles + think traces | -0.024 | -0.010 | +0.048 | +0.090 | 51.8% | 6688 |
| V6 SFT (thinking+bearish) | 1121 articles w/ bearish | -0.060 | -0.094 | +0.041 | +0.103 | 48.1% | 7704 |
| Qwen3-8B Fine tuned | 881 teacher labeled articles | +0.009 | +0.078 | -0.013 | +0.001 | 55.4% | 2937 |
| Manus Teacher (direct) | N/A (inference only) | +0.021 | +0.042 | -0.065 | -0.050 | 52.3% | 8597 |

### What Each Model Version Revealed

**V4 Supervised Fine Tuning (SFT), no thinking traces.** This was the baseline that everything else was measured against. Trained on 881 articles with teacher opinion labels using standard SFT. It achieved IC +0.128 at 10 days (p=0.002). The model was a "calibrated hype machine" with 82% bullish predictions and 67% of ticker article pairs scored as zero. Its strength was selectivity: when it had an opinion, it was usually directionally correct for bullish calls.

**V6 SFT with thinking traces and bearish supplements.** This version added the 240 bearish training examples from V5.1/V5.2 and included `<think>` reasoning traces. The result was a dramatic hurt to performance: IC went negative at short horizons (-0.094 at 5 days). The bearish labels from the teacher's opinion were anti-correlated with actual returns. The model learned to be bearish when the teacher thought it should be, but the market disagreed. This was the critical lesson: teacher opinion bearish labels are harmful when used for supervised fine tuning. The teacher's intuitions about what should be negative did not match what the market actually did.

**V7a SFT with thinking traces only, no bearish supplements.** This version removed the bearish supplements but kept thinking traces. It was still worse than V4. The thinking traces made the model less selective (6,688 observations versus 3,265 for V4). The model reasoned its way into making predictions it should not have been making. More reasoning led to more opinions, but those additional opinions were noise.

**V7b Rejection Sampling.** This approach generated 4 candidate predictions per training article using the SFT model, scored each candidate against actual 5 day returns, kept the best one, and retrained SFT on those "verified good" examples. It showed a strong IC at the 2 day horizon but was inconsistent at other horizons. The strength of this approach is its simplicity and stability. The weakness is that it only learns from positive examples and does not learn what to avoid.

**V7c Direct Preference Optimization (DPO).** This approach used the same candidates as rejection sampling but trained with preference pairs. The model learns to prefer predictions that correlate with reality over predictions that do not. It was the most consistent across horizons (all positive) with 56.6% direction accuracy, but no single horizon reached high significance. DPO is more informative than rejection sampling because it learns what to avoid, not just what to do. However, it is sensitive to pair quality and requires meaningful differences between best and worst candidates.

**V7d Group Relative Policy Optimization (GRPO).** This was the breakthrough. GRPO is a reinforcement learning approach where the model generates multiple candidates, all are scored by a reward function grounded in actual market returns, and the model is updated via policy gradient to maximize expected reward. The reward function combines three components: direction accuracy (40% weight), magnitude correlation which is the IC itself (40% weight), and a selectivity bonus that rewards silence on noise (20% weight).

The result was consistent IC of +0.15 to +0.16 across all horizons (1, 5, 10, and 20 days), all statistically significant, with 58.6% direction accuracy. GRPO optimizes the actual objective (IC) directly. It learns from exploration rather than imitation. It handles bearish signals correctly because bearish predictions are only reinforced when stocks actually decline. And it learns selectivity naturally because the reward function rewards silence on noise.

**V8 SFT and GRPO (GPT 5.5 Teacher).** These models were trained using data generated by Codex/GPT 5.5 instead of Manus. Surprisingly, despite GPT 5.5 being a more capable model, the Manus trained models performed better. V4 SFT (Manus) beats V8 SFT (GPT 5.5) at 10 and 20 days, and V7d GRPO (Manus) dramatically beats V8 GRPO (GPT 5.5) at all horizons. The likely explanation is that Manus uses internet access and sub agents to research each article before labeling, producing labels grounded in real world context (company financials, technology details, competitive dynamics) that a "closed book" model cannot access regardless of its reasoning capability.

**Qwen3-8B Fine tuned.** Trained on the same 881 teacher labeled articles as V4 but using the original Qwen3-8B base model with the suboptimal LoRA configuration (r=64, alpha=16). The results were markedly worse: IC of +0.078 at 5 days but -0.013 at 10 days and +0.001 at 20 days. This confirms that the switch to OpenReasoning-Nemotron-7B with corrected LoRA hyperparameters was the right decision.

**Manus Teacher (direct inference).** Running the Manus teacher model directly on all evaluation articles (without any fine tuning, just using the teacher for inference) produced IC of +0.042 at 5 days, which degraded to -0.065 at 10 days and -0.050 at 20 days. The fine tuned student (V7d GRPO at +0.157 for 5 days) dramatically outperforms the teacher (+0.042 for 5 days). This is the hallmark of successful knowledge distillation: the student generalizes better than the teacher on specific examples.

### The Training Approaches Explained

**Supervised Fine Tuning (SFT)** is the simplest approach. The model is shown input output pairs and trained to reproduce the outputs. Its strength is teaching format compliance (JSON output) and basic domain knowledge. Its weakness is that the model can only be as good as the teacher labels. If the teacher is wrong about what constitutes a bearish signal, the student will be wrong too.

**Rejection Sampling + SFT** generates multiple candidates, keeps the best one (scored against reality), and retrains. It uses actual return data to select training examples but only learns from positive examples.

**Direct Preference Optimization (DPO)** trains the model to prefer predictions that correlate with actual returns over predictions that do not. It uses preference pairs (best versus worst candidate for each article) and learns both what to do and what to avoid.

**Group Relative Policy Optimization (GRPO)** is the most powerful approach. It is reinforcement learning directly from market returns. The model generates multiple candidates, all are scored by the reward function, and the model is updated via policy gradient. It optimizes the actual objective (IC), learns from exploration, handles bearish signals correctly, and learns selectivity naturally. Its weaknesses are speed (each step requires generating 4 completions), potential instability, and the need for careful reward function design.

### The Overfitting Mistake and Recovery

Midway through V7, a critical error was made that temporarily inflated results. The rejection sampling and DPO training data was initially built from the evaluation articles (the same 421 articles being tested on). This meant the model had been optimized against the returns of the articles it was being evaluated on.

The contaminated results showed V7b IC at 10 days of +0.171 and V7c direction accuracy of 61.1%. The error was caught by running a train/test split analysis, which revealed that there were zero true out of sample observations.

The fix was to rebuild all V7 training data using only the 184 training articles that had dates and return data. The evaluation articles were never touched. After this correction, the results were honest but more modest. GRPO still won, but with IC +0.16 instead of the inflated +0.17. The lesson is absolute: never use evaluation data for any part of the training pipeline. This includes reward computation, preference pair construction, and rejection sampling scoring.

### What the Signal Actually Looks Like

Several deep dive analyses revealed the nature of the model's predictive ability.

The model is primarily a news signal. Only news articles generate tradeable predictions. ArXiv papers produce zero signal (3 out of 144 passed the 0.5 threshold, and those were mostly wrong directionally). The model correctly learned to ignore academic papers.

Selectivity is the key driver of IC. The V4 model assigned 0.0 to 67% of ticker article pairs. A threshold sweep showed that filtering out predictions with absolute score below 0.5 improved IC from +0.112 to +0.190. Low conviction predictions are noise.

The signal is concentrated in specific tickers. RGTI (Rigetti) had the strongest IC at +0.228 (p=0.036). QBTS (D-Wave) was second at +0.182. The model is particularly good at predicting these two stocks.

Teacher opinion bearish labels are harmful. When bearish training examples based on what the teacher thought should be negative were added, the model's IC went negative at short horizons. The market disagreed with the teacher's bearish intuitions. Only GRPO (which learns from actual returns) produced useful bearish predictions.

#### Why Manus as the Teacher

The initial training data was generated by Manus, an autonomous AI agent. This is unusual. Most fine tuning projects use either human labels or a frontier API model (GPT-4, Claude) as the teacher. Manus was chosen for three reasons.

First, Manus can access the internet. When labeling a news article about "IonQ's partnership with KIST," Manus can look up what KIST is, check IonQ's recent stock price, and read the actual press release rather than guessing from a headline.

Second, Manus can spawn sub-agents. For arXiv papers, Manus dispatches a separate agent to read the paper, understand the methodology, and determine whether the results are genuinely novel.

Third, the empirical results validate the approach. When we tested GPT 5.5 as the teacher model (V8), the resulting student models performed significantly worse than the Manus trained students. The grounded, well researched labels produced by an agentic system proved more valuable for financial signal generation than the raw reasoning power of a closed book frontier model.

## Lessons Learned

### From Training Data Development

**Empirical grounding resolves ambiguity.** The teacher model was split 50/50 on whether Google's Willow breakthrough was bullish or bearish for RGTI. No amount of prompt engineering could resolve this theoretically. But the market data (RGTI +89% in 5 days) resolved it instantly. When the teacher is uncertain, check what actually happened.

**Adversarial review prevents costly mistakes.** Every implementation phase was preceded by an adversarial analysis. The most valuable catches: HON was initially slated for removal despite being the best predictor (IC=0.166); the minimum conviction threshold caused catastrophic over zeroing before refinement; and the chain of thought placeholder issue (53% of data) would have gone unnoticed without systematic quality checks.

**Source aware rules beat universal rules.** The minimum conviction threshold seemed universally correct ("don't guess when uncertain") but was catastrophic for news articles. The refined version distinguishes between news (should almost always score pure plays), arXiv (default to zero), and non quantum content (all zeros). Domain specific rules outperform general principles.

**The teacher's web research does not create a train test gap.** A concern was raised: the teacher browses the web to research articles, but the student model runs locally without internet access. Analysis showed this is not a problem because the article text is already in the user message, the system prompt contains all necessary company and technology knowledge, the student learns reasoning patterns rather than facts it needs to look up, and over 90% of the teacher's web browsing confirms what is already in the article.

**Thinking traces must drive scores, not decorate them.** V4's chain of thought field was often a post hoc rationalization (or worse, "REDACTED"). V5 forces the model to think first, then score. This ensures consistency between reasoning and output, and teaches the student model that reasoning is a prerequisite for scoring, not an afterthought.

**Directional balance requires intentional construction.** Left to its own devices, the teacher model produces 91% bullish labels on quantum computing news. This is natural: most quantum articles are about progress and breakthroughs. But a model trained on 91% bullish data becomes a hype machine that cannot predict downside. The fix required intentionally constructing bearish scenarios and feeding them to the teacher.

**Models are fragile in regimes they have not seen.** The most dangerous gap in the V5 base data was the absence of drawdown scenarios. The model had never seen market context showing stocks down 20 to 50%. Without this training, the model would likely produce random or inappropriately bullish signals during a real drawdown. Always ask: what market regime has the model never seen? That is where it will fail.

**Conflicting signals teach moderation.** Real world news is rarely purely bullish or bearish. "Great earnings but dilutive offering" requires the model to weigh competing factors and produce moderate scores. Without explicit training on mixed signals, the model learns to always go to extremes.

### From Fine Tuning

**The bearish examples paradox.** The bearish training examples were well motivated from a data quality perspective (the model needed to learn that negative scores are valid), but empirically harmful when used for supervised fine tuning. The teacher's opinions about what should be bearish did not match what the market actually did. This is what ultimately drove the pivot to GRPO, where the model learns from actual returns rather than teacher opinions. GRPO can produce useful bearish predictions because bearish signals are only reinforced when stocks actually decline.

**Selectivity matters more than coverage.** The V4 model's strength was not that it was right more often, but that it knew when to stay silent. Assigning 0.0 to 67% of ticker article pairs and only speaking with conviction on the remaining 33% produced a much higher IC than models that had an opinion on everything.

**Infrastructure determines what experiments are possible.** The 90x speedup from switching to vLLM batch inference was not just a cost optimization. It made the entire V7 experimental cycle possible. Generating 4 candidates per article for 184 training articles (736 total predictions) would have taken 11 hours with sequential inference. With vLLM, it took 8 minutes. This enabled rapid iteration on reward functions, preference pair construction, and GRPO hyperparameters.

**The student can outperform the teacher.** The GRPO fine tuned student (IC +0.151) dramatically outperforms the Manus teacher's direct predictions (IC +0.042). This is not a failure of the teacher. It is the expected outcome of knowledge distillation combined with reinforcement learning from ground truth. The teacher provides the initial capability (format, domain knowledge, reasoning patterns), and GRPO refines it against reality.

## Frontend Engineering: Deploying a Custom Trading Terminal on Hugging Face Spaces

The Alpha Signal Analysis platform needed a custom frontend. Not a Gradio widget grid, not a Streamlit dashboard, but a purpose built terminal style interface with interactive Plotly charts, tabbed navigation, a welcome overlay, concurrent inference feeds, and a FastAPI backend serving JSON endpoints. Deploying this on Hugging Face Spaces with ZeroGPU presented unique challenges.

### The Port Binding Problem

On Hugging Face Spaces with the Gradio SDK, the runtime occupies port 7860 before custom code even runs. The `spaces` package patches Gradio's launch method to handle GPU allocation and authentication. When attempting the standard pattern of mounting Gradio onto FastAPI using `gr.mount_gradio_app()` and then calling `demo.launch()`, the Spaces runtime had already claimed port 7860, causing the app to crash with an `OSError: Cannot find empty port` error.

The fundamental issue is that `gr.mount_gradio_app()` is designed for when the developer controls the server (running uvicorn directly). On Hugging Face Spaces, the runtime controls the server. Calling both `gr.mount_gradio_app()` and `demo.launch()` causes a conflict because `launch()` tries to start a second server.

### The Solution: gradio.Server Mode

Gradio 5.29 introduced `gradio.Server`, a class that inherits directly from FastAPI. Instead of creating a FastAPI app and trying to bolt Gradio onto it, developers can use Gradio's own FastAPI subclass that handles all the port management, queue infrastructure, and ZeroGPU integration internally.

This approach works because `Server.launch()` is aware of the Hugging Face Spaces runtime. It does not try to start a competing server. It registers itself with the existing infrastructure, and custom FastAPI routes (such as `/api/*`, `/`, `/static/*`) coexist with Gradio's internal endpoints on the same port. The key insight is that `gradio.Server` is not Gradio mounted on FastAPI; it is FastAPI, with Gradio's queue and GPU allocation built in. Custom routes are first class citizens.

For live inference on ZeroGPU hardware, the `@spaces.GPU` decorator needs to wrap the function that touches the GPU. This function is then routed through both a Gradio API endpoint (for proper queue and authentication handling) and a standard FastAPI POST endpoint (for the custom frontend to call). A critical deployment issue: the model repository was initially set to private, causing a generic 500 Internal Server Error. Making the model public resolved this. We also increased `max_new_tokens` to 10,000 because the GRPO model outputs detailed per-ticker reasoning that was getting truncated at lower limits.

### The Frontend Architecture

The frontend is pure HTML, CSS, and JavaScript with no build step. It consists of three files (`index.html`, `styles.css`, `app.js`) served as static files from the same origin as the API. This eliminates CORS issues, the need for a proxy, and framework overhead.

The visual language communicates "quantitative finance infrastructure" through a terminal inspired design system: pure black backgrounds (`#0c0c0c`, `#121212`, `#1a1a1a`) with no gradients, monospace typography (JetBrains Mono) for all text, green accent (`#00ff88`) for active states and interactive elements, minimal border radius (2px) for sharp technical edges, uppercase headers with letter spacing, and outlined buttons that invert to solid green on hover.

A full screen welcome overlay appears on every visit, requiring users to click through before accessing the dashboard. This serves as the product pitch: title and tagline, four stat cards (IC, p-value, ticker coverage, model iterations), description of what the signal does, and market context with the Citadel link. The CTA button is positioned near the top so it is immediately visible without scrolling.

### The Trading Terminal Interface

The tab order positions the product as a fine tuning evaluation tool for quant signal development:

**Tab 1: Evaluation Dashboard.** The landing page shows IC metrics across all 13 model iterations. Two context cards define IC and explain how to read the dashboard. The Signal Decay Curve plots IC at horizons of 1, 2, 5, 10, and 20 days with bright colored markers indicating statistical significance. Default state shows only 3 models checked on load (the best fine tuned model and both teacher models) to prevent visual overload while clearly showing the student beats teacher result. The IC Comparison Table uses explicit descriptive model names (e.g., "Nemotron-7B (SFT + GRPO, Manus Teacher)") that immediately communicate base model, training method, and teacher.

**Tab 2: Live Signal Debugging.** This tab supports concurrent, non-blocking analyses displayed as a scrollable feed. The input form stays at the top; clicking Analyze immediately adds a loading card to the feed and clears the input. Requests fire asynchronously so you can submit the next article without waiting. Results populate into each card as they return, with newest results at the top. Each feed card shows a model badge, source badge, timestamp, latency, article preview, inline signal bar chart, metadata, and expandable raw JSON and thinking trace sections. The feed handles multiple signal formats gracefully.

**Tab 3: Historical Prediction Analysis.** This tab loads precomputed predictions from all model iterations and displays them against realized market outcomes for 388+ articles from January to June 2026. Model filter checkboxes allow toggling which models appear. Three price charts provide context: Raw Price Movement with SPY overlay, Abnormal Returns versus Market (the actual metric IC is evaluated against), and Abnormal Returns versus Sector (equal weight quantum basket). During development, an article index alignment problem was discovered where the Manus Teacher predictions used different numbering; this was fixed by re-indexing based on article titles.

**Tab 4: Sector Map.** A reference tab showing the quantum computing sector structure, including signal weights by company, technology clusters, and signal propagation rules.

### Mobile Responsive Design

The terminal theme required careful mobile adaptation since monospace fonts and uppercase text consume more horizontal space. All responsive rules are CSS only (no JS changes), applied through 8 media query blocks. Tablets (768px and below) get stacked layouts, horizontal scroll on tab navigation, and reduced padding. Phones (480px and below) get a 2x2 stat grid on the welcome overlay, full width controls, and tighter chart containers. Key patterns include overflow-x auto with touch scrolling on chart containers, and the welcome overlay uses overflow-y auto so all content is accessible regardless of viewport height.

### Final Architecture and Deployment

The deployment workflow uses the `huggingface_hub` Python library to upload files and restart the Space. The Space is configured with the Gradio SDK, `zero-a10g` hardware, and Python 3.12.

The final architecture consists of a Hugging Face Space running `app.py` (using `gradio.Server`) which serves the static frontend files and provides JSON API endpoints for 13 models, events, predictions, cross-model comparison, evaluation metrics, sector data, and live GPU inference (max 10K tokens). The data layer includes 13 precomputed prediction JSONL files and market price parquet files.

### What We Learned

`gradio.Server` is the correct pattern for custom frontends on HF Spaces. ZeroGPU authentication flows through Gradio's internal mechanisms. Model repos must be public for the Space to download them. `max_new_tokens` matters for verbose models (the GRPO model needs 10,000 tokens for full per-ticker reasoning). Data alignment across model prediction files cannot be assumed; always join on a stable key. Default to showing fewer models and let users add more. Non-blocking concurrent requests are essential for debugging workflows. Terminal aesthetics communicate precision to the target audience. CSS-only responsive design carries zero risk of breaking desktop behavior.

## Limitations and Caveats

While the results demonstrate statistically significant predictive power, they should be interpreted with the following limitations:

- **Single Market Regime:** The evaluation period spans January to June 2026. This represents a single market regime, and performance may vary during broader sector selloffs or different macroeconomic conditions.
- **Sample Size:** While 421 articles is sufficient for top-level statistical significance, subset analyses (e.g., performance on specific event types or individual tickers) have limited statistical power.
- **Single-Factor Benchmark:** The CAR computation uses a single-factor CAPM model (SPY only). A multi-factor model (e.g., Fama-French plus a quantum sector factor) might explain some of the generated "alpha."
- **No Execution Simulation:** The signal has not been tested in a live paper-trading environment. Real-world implementation would need to account for execution costs, slippage, and market impact, especially given the lower liquidity of some quantum computing micro-caps.
- **Correlation Does Not Guarantee Profit:** A positive Information Coefficient indicates rank correlation with future returns, but translating this signal into a profitable, risk-managed portfolio requires additional quantitative engineering.

## Conclusion

The Alpha Signal Analysis project successfully demonstrated that a small language model (7 billion parameters), fine tuned specifically on quantum computing financial analysis using reinforcement learning (GRPO), can produce trading signals that outperform both larger general purpose models and the base model without fine tuning.

The journey revealed several critical insights. First, domain specific fine tuning requires high quality, empirically grounded training data; the market's reaction is the ultimate arbiter of truth, not the teacher model's intuition. Second, infrastructure choices matter immensely; moving from sequential generation to vLLM batch inference enabled a 90x speedup that made rapid experimentation possible. Third, the training methodology is as important as the data; supervised fine tuning on teacher opinions proved harmful when those opinions misaligned with the market, while reinforcement learning directly from market returns (GRPO) yielded consistent, statistically significant Information Coefficients of +0.15 to +0.16 across all horizons. Finally, deploying complex, custom interfaces on platforms like Hugging Face Spaces requires understanding the underlying server architecture, where `gradio.Server` provides the necessary control without sacrificing platform integration.

The resulting platform, accessible via a custom built trading terminal, serves as both a live inference engine and a comprehensive evaluation dashboard, proving the "Build Small" thesis in a highly technical and complex financial sector.

---

## References

[1] Lopez-Lira, A. and Tang, Y. (2024). "Can ChatGPT Forecast Stock Price Movements? Return Predictability and Large Language Models." *Journal of Finance*.

[2] Truong, C. (2025). "Industry-Specific Sentiment Analysis for Trading Signals." Working paper.

[3] Wang, Y. et al. (2023). "FinGPT: Open-Source Financial Large Language Models." arXiv.

[4] Mukherjee, S., Mitra, A., Jawahar, G., et al. (2023). "Orca: Progressive Learning from Complex Explanation Traces of GPT-4." arXiv:2306.02707.

[5] Wang, Y. et al. (2024). "GPT-Signal: Generative AI for Semi-Automated Feature Engineering in Quantitative Investment." Working paper.

[6] Ding, Y. et al. (2025). "TradExpert: Revolutionizing Trading with Mixture of Expert LLMs." Working paper.

[7] Grinold, R. and Kahn, R. "Active Portfolio Management." McGraw-Hill.

[8] Zhou, C., Liu, P., Xu, P., et al. (2023). "LIMA: Less Is More for Alignment." NeurIPS 2023. arXiv:2305.11206.

[9] Gunasekar, S. et al. (2023). "Textbooks Are All You Need." Microsoft Research.

[10] Chen, L., Li, S., Yan, J., et al. (2024). "AlpaGasus: Training A Better Alpaca with Fewer Data." ICLR 2024.

[11] Jansen, S. (2020). "Machine Learning for Algorithmic Trading." 2nd Edition, Packt Publishing.

[12] Hu, E. J., et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685.

[13] Kwon, W., et al. (2023). "vLLM: Efficient Memory Management for Large Language Model Serving with PagedAttention." arXiv:2309.06180.

[14] Wei, J., Wang, X., Schuurmans, D., et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS 2022.

[15] Hinton, G., Vinyals, O., Dean, J. (2015). "Distilling the Knowledge in a Neural Network." arXiv:1503.02531.
