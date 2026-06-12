# Quantum Alpha Intelligence: Building a Domain-Specific NLP Signal Generator for the Quantum Computing Sector

## The Hypothesis

Most retail investors and even many institutional analysts fundamentally misunderstand the quantum computing sector. When IonQ announces "35 algorithmic qubits," the market reaction depends entirely on whether participants understand the distinction between physical qubits, logical qubits, and algorithmic qubits, and whether they can contextualize that number against the company's stated roadmap. When a research group publishes a paper demonstrating error correction below the fault-tolerance threshold on a superconducting architecture, the implications for Rigetti versus IonQ versus D-Wave are non-obvious to anyone without a physics background.

This creates an information asymmetry that persists longer than in most sectors. In traditional equity markets, earnings surprises get priced in within minutes. In quantum computing, a technical milestone announcement can take days or weeks to be fully reflected in stock prices because the analyst community lacks the domain expertise to rapidly assess significance. Lopez-Lira and Tang (2024) documented this phenomenon empirically: GPT-4 sentiment scores predict next-day stock returns with a Sharpe ratio of 3.8, and crucially, predictability is strongest for smaller stocks and low-readability (highly complex) news. Quantum computing sits squarely in this sweet spot: small-cap stocks, highly technical announcements, and a market that struggles to parse the signal from the noise.

Our hypothesis is straightforward: a small language model (8 billion parameters), fine-tuned specifically on quantum computing financial analysis, can produce trading signals that outperform both larger general-purpose models and the base model without fine-tuning. The key insight is that domain-specific fine-tuning on a narrow task can compensate for raw model scale. This is the "Build Small" thesis: you do not need 200 billion parameters to extract alpha from quantum computing news if you have the right training data and the right evaluation framework.

## Why Quantum Computing Specifically

The quantum computing sector has several properties that make it uniquely suited to NLP-based alpha generation:

**Slow information diffusion.** Unlike earnings reports or macroeconomic data (which get priced in within seconds by algorithmic traders), quantum computing breakthroughs propagate slowly through the market. A paper published on arXiv about logical qubit error rates might take 3-10 trading days to be fully reflected in stock prices because most market participants cannot assess its significance without domain expertise. Truong (2025) documented this pattern explicitly: industry-specific sentiment reveals unique opportunities due to gradual information diffusion within specialized sectors.

**High technical complexity creates persistent mispricing.** The quantum computing sector has at least five distinct technological approaches (trapped-ion, superconducting, quantum annealing, topological, neutral atom), each with different strengths, timelines, and commercial viability. A breakthrough in one approach has asymmetric implications across the competitive landscape. For example, a superconducting error correction advance is bullish for Rigetti and IBM but bearish for IonQ and Honeywell (trapped-ion competitors). Most investors do not understand these competitive dynamics, creating persistent mispricing that a domain-expert model can exploit.

**Small universe with clear competitive structure.** The publicly traded quantum computing universe consists of approximately 10 companies, ranging from pure-play quantum firms (IonQ, Rigetti, D-Wave, Quantum Computing Inc., Quantinuum via Honeywell) to diversified technology companies with quantum divisions (IBM, Google, Microsoft, NVIDIA). This small, well-defined universe makes cross-sectional analysis tractable: every piece of news can be scored against every company simultaneously, producing a complete signal vector rather than a single-stock sentiment label.

**Multiple information sources with varying lead times.** Quantum computing information flows through a predictable pipeline: academic preprints on arXiv (earliest signal, often weeks before press coverage), company press releases, SEC filings, financial news articles, and social media commentary. A system that ingests at the source (arXiv) can capture signals before they reach mainstream financial media.

## The Academic Foundation

This project builds on several established research threads:

**LLMs for financial sentiment prediction.** Lopez-Lira and Tang (2024) demonstrated that large language models can predict stock returns from news headlines with statistical significance. Their key finding, that predictive power is strongest for complex, low-readability text, directly motivates our focus on technical quantum computing content. Basic models like BERT failed at this task, suggesting that the ability to translate technical language into financial implications is an emergent capability of larger models, or in our case, a capability that can be instilled through domain-specific fine-tuning.

**Teacher-student distillation for financial NLP.** The FinGPT project (Wang et al., 2023) established the paradigm of using instruction tuning with LoRA to adapt general-purpose LLMs to financial tasks. Their approach, which we follow, uses a larger "teacher" model to generate training labels and then trains a smaller "student" model to reproduce those outputs. The Orca paper (Mukherjee et al., 2023) showed that the quality of the reasoning traces in the training data matters more than the volume: a 13B model trained on rich explanations from GPT-4 achieved remarkable reasoning capabilities, outperforming models trained on shallow outputs.

**Cross-sectional signal generation.** The GPT-Signal paper (Wang et al., 2024) demonstrated that LLMs can generate novel return-predictive formulaic alphas across sectors (IT, Healthcare, Energy) that consistently outperform baseline signals over a 5-year backtest. TradExpert (Ding et al., 2025) proved that a Mixture of Experts approach, using specialized LLMs for different information types, outperforms single-model systems. Our architecture draws from both: we produce cross-sectional signals (scoring all companies simultaneously) and use specialized prompting for different source types (news, arXiv, SEC filings).

**Event study methodology for signal evaluation.** Rather than relying on simple directional accuracy (which can be misleading), we evaluate our signals using the Information Coefficient (Spearman rank correlation between predicted signals and realized abnormal returns), following the standard quantitative finance methodology described in Grinold and Kahn's "Active Portfolio Management." This approach controls for market movements using a factor model and measures whether the model's relative rankings of stocks correlate with their subsequent relative performance.

## System Architecture

The Quantum Alpha Intelligence platform consists of four major components:

### 1. Data Ingestion Pipeline

The system collects quantum computing-related content from multiple sources:

- **arXiv preprints** (quant-ph and cs.ET categories): These represent the earliest possible signal, often published weeks before press coverage. We monitor for papers mentioning quantum computing companies or relevant technical milestones.
- **Financial news** (via RSS feeds from Google News, Yahoo Finance): Real-time coverage of business events, partnerships, earnings, and analyst commentary.
- **SEC filings** (EDGAR): 10-K, 10-Q, and 8-K filings for the quantum computing companies, capturing regulatory disclosures and financial data.

Each article is timestamped, source-tagged, and cleaned (HTML stripped, URLs removed) before being passed to the signal generation model.

### 2. Signal Generation Model

The core of the system is a fine-tuned Qwen3-8B model that takes an article as input and produces a structured JSON output containing:

- **Cross-sectional signal vector**: A score for every company in the quantum universe (10 tickers), ranging from -2.0 (strongly bearish) to +2.0 (strongly bullish). Scores are scaled by each company's quantum revenue exposure (pure-play companies get full range; diversified companies like Google are capped at small magnitudes because quantum news has negligible impact on their stock price).
- **Event classification**: What type of event this represents (technical milestone, commercial partnership, government funding, competitive development, etc.).
- **Time horizon**: How long the signal is expected to persist before being fully priced in.
- **Technical translation**: A plain-language explanation of why this event matters commercially, translating physics jargon into investment-relevant language.
- **Chain of thought**: The model's reasoning process, explaining how it arrived at each score.

The signal vector is the key innovation. Rather than producing a single sentiment label for a single stock (which is what most financial NLP systems do), we produce a complete cross-section that captures the competitive dynamics: a trapped-ion breakthrough is simultaneously bullish for IonQ and bearish for Rigetti, with magnitudes determined by the specific nature of the advance.

### 3. Evaluation Framework

We evaluate the model using established quantitative finance methodology:

- **Abnormal Returns (AR)**: For each event, we compute the stock's return minus its expected return (estimated via OLS regression against the S&P 500 over a prior estimation window). This isolates the company-specific price movement from broader market trends.
- **Information Coefficient (IC)**: The Spearman rank correlation between the model's predicted signal scores and the realized abnormal returns. An IC above 0.05 is considered meaningful in quantitative finance; above 0.10 is exceptional.
- **Signal Decay Analysis**: We compute IC at multiple holding periods (1, 2, 5, 10, 20 trading days) to measure how quickly the signal's predictive power fades. This validates our hypothesis about slow information diffusion in the quantum sector.
- **Multi-model comparison**: We run the same evaluation on multiple models (fine-tuned vs. base, small vs. large) to demonstrate that domain-specific fine-tuning adds measurable value.

### 4. Interactive Dashboard

The platform is deployed as a Hugging Face Space with a custom trading terminal frontend (built with vanilla HTML/CSS/JS, not default Gradio components). The dashboard provides:

- **Live Analysis**: Paste any article and get real-time signal predictions from the fine-tuned model running on GPU.
- **Historical Predictions**: Browse all evaluation events and compare predictions across multiple models side-by-side, with actual price movement overlays.
- **Evaluation Dashboard**: Interactive IC comparison charts, signal decay curves, and statistical significance indicators across all models.
- **Sector Map**: Visualization of the quantum computing competitive landscape, technology clusters, and signal propagation rules.

## Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Base model | Qwen3-8B | Best zero-shot financial NLP performance in its size class |
| Fine-tuning | QLoRA (rank 64) via Unsloth | 2% of parameters trainable, fits on single A100 |
| Training compute | Modal (A100 GPUs) | Serverless GPU for training runs (~$3 per run) |
| Teacher model | Manus (agentic AI with web browsing) | Generates high-quality training labels with deep research |
| Inference | HF ZeroGPU | On-demand GPU allocation for live predictions |
| Evaluation | Custom (scipy, statsmodels, pandas) | Abnormal returns, IC, signal decay |
| Frontend | Custom HTML/CSS/JS + Plotly | Trading terminal UI (Off-Brand badge) |
| Backend | FastAPI + Gradio Server | API endpoints + GPU inference |
| Deployment | Hugging Face Spaces | Public demo for hackathon judges |
| Version control | GitHub (private) | Development workspace |

## The Training Data Strategy

The research literature is clear: for domain-specific fine-tuning, quality matters more than quantity, but you need a minimum threshold of examples to achieve robust performance. The LIMA paper (Zhou et al., 2023) showed that 1,000 carefully curated examples can match the performance of models trained on orders of magnitude more data. The Phi series (Gunasekar et al., 2023) proved that "textbook quality" data enables models as small as 2.7B parameters to match or outperform models ten times their size.

We adopted a multi-category training data strategy:

- **Real articles with deep research** (190 examples): Each article is processed by a Manus agent that browses the web, researches the companies, verifies claims, and produces a richly-reasoned signal vector. This is the highest-quality category.
- **Synthetic articles** (200 examples): The teacher generates realistic news articles for specific scenarios (technical milestones, business events, government funding) and then analyzes them. This covers scenarios that may not appear in our real data.
- **Paraphrased articles** (190 examples): The same real articles rewritten in different styles (press release, blog post, analyst note, tweet thread) with identical signal vectors. This teaches the model that the signal comes from content, not style.
- **Negative examples** (150 examples): Articles about non-quantum topics where all signal scores are zero. This prevents the model from hallucinating quantum relevance in unrelated content.
- **Edge cases** (100 examples): Ambiguous scenarios with conflicting implications, teaching the model to express measured uncertainty rather than defaulting to extreme scores.
- **Multi-turn follow-ups** (170 examples): Follow-up questions about the model's reasoning, teaching it to explain and defend its signal assignments.

The total dataset is approximately 1,000 examples, right at the sweet spot identified by the literature for complex instruction-following tasks.

## What Comes Next

The sections that follow document our iterative process: how we generated and refined the training data, the fine-tuning experiments we ran (comparing LoRA configurations, base models, and teacher quality), the prediction pipeline and its operational challenges, and the evaluation results that validate (or challenge) our initial hypothesis.

The core question we set out to answer: can a fine-tuned 8B model, armed with domain-specific training data about quantum computing, produce trading signals that outperform both larger general-purpose models and the base model without fine-tuning? The answer, as measured by Information Coefficient at a 5-day horizon, is yes, but with important caveats that we document transparently.
