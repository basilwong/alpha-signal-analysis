# Alpha Signal Analysis Platform: End-to-End System Design

## 1. System Overview

The platform is a domain-specific market intelligence system for the quantum computing sector. It ingests unstructured text (news, papers, filings), processes it through a language model to extract structured trading signals and contextual briefings, and presents the results through a professional dashboard interface.

The system has two operational modes that share a common architecture:

**V1 (Full Power)**: Uses `qwen3-max` via Alibaba Cloud API with persistent memory. Targets the Qwen Cloud Global Hackathon (Memory Agent track, deadline July 9).

**V2 (Small Model)**: Uses a fine-tuned `qwen3-8b` running on HF ZeroGPU. Targets the Build Small Hackathon (deadline June 15).

Both versions share the same frontend, data ingestion pipeline, and output schema. The only difference is the inference backend.

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA SOURCES                                    │
│                                                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │
│  │  arXiv   │  │   RSS    │  │   SEC    │  │  Yahoo   │  │  Reddit  │    │
│  │ quant-ph │  │  News    │  │  EDGAR   │  │ Finance  │  │  Posts   │    │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘    │
│       │              │              │              │              │          │
└───────┼──────────────┼──────────────┼──────────────┼──────────────┼──────────┘
        │              │              │              │              │
        └──────────────┴──────────────┴──────┬───────┴──────────────┘
                                             │
                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION PIPELINE                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  collect_articles.py                                                 │   │
│  │  * Fetches from all sources on schedule or on-demand                 │   │
│  │  * Deduplicates by title/URL                                         │   │
│  │  * Outputs: data/raw/articles.jsonl                                  │   │
│  │  * Format: {"text": "...", "source": "...", "title": "...", "date"}  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INFERENCE BACKEND (SWITCHABLE)                       │
│                                                                             │
│  ┌─────────────────────────────┐    ┌─────────────────────────────────┐    │
│  │  V1: qwen3-max (Teacher)    │    │  V2: Fine-tuned qwen3-8b        │    │
│  │                             │    │                                   │    │
│  │  Provider: Alibaba Cloud    │    │  Provider: HF ZeroGPU or Modal   │    │
│  │  Endpoint: Singapore API    │    │  Model: basilwong/alpha-signal  │    │
│  │  Memory: ChromaDB/Vector DB │    │  Memory: None (stateless)        │    │
│  │  Context: Persistent across │    │  Context: Per-request only       │    │
│  │           sessions          │    │                                   │    │
│  │  Cost: Free tier tokens     │    │  Cost: $0 (ZeroGPU)              │    │
│  └──────────────┬──────────────┘    └────────────────┬──────────────────┘    │
│                 │                                     │                      │
│                 └──────────────┬──────────────────────┘                      │
│                                │                                             │
└────────────────────────────────┼─────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIGNAL PROCESSING LAYER                              │
│                                                                             │
│  Input: Raw article text                                                    │
│  Output: Structured JSON signal                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  {                                                                   │   │
│  │    "sentiment": "bullish",                                           │   │
│  │    "confidence": 0.85,                                               │   │
│  │    "event_type": "logical_qubit_breakthrough",                       │   │
│  │    "affected_tickers": ["IONQ", "GOOGL"],                            │   │
│  │    "urgency": "high",                                                │   │
│  │    "technical_translation": "This means...",                          │   │
│  │    "key_facts": ["fact1", "fact2"],                                   │   │
│  │    "competitive_context": "Relative to competitors..."               │   │
│  │  }                                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         GRADIO SERVER (BACKEND API)                          │
│                                                                             │
│  Framework: gradio.Server (FastAPI)                                         │
│  Hosting: Hugging Face Spaces (build-small-hackathon org)                   │
│                                                                             │
│  Endpoints:                                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  @app.api("/analyze_news")     → Process single article → signal    │   │
│  │  @app.api("/get_signals")      → Retrieve latest signals for ticker │   │
│  │  @app.api("/get_briefing")     → Generate daily sector briefing     │   │
│  │  @app.api("/get_sector_overview") → Aggregated sector sentiment     │   │
│  │  @app.get("/")                 → Serve custom frontend              │   │
│  │  @app.get("/health")           → Health check                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────┬───────────────────────────────┘
                                              │
                                              ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CUSTOM FRONTEND (Trading Terminal)                   │
│                                                                             │
│  Technology: HTML5 / CSS3 / JavaScript (vanilla, no framework)              │
│  Connection: @gradio/client JS library (queued requests via SSE)            │
│  Theme: Dark professional trading terminal                                  │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌──────────────┐  ┌──────────────────────┐  ┌─────────────────┐  │   │
│  │  │ Sector Pulse │  │    Live Signals Feed  │  │  Ticker Heatmap │  │   │
│  │  │  (sentiment  │  │  (real-time entries   │  │  (color-coded   │  │   │
│  │  │   gauge)     │  │   with confidence)    │  │   by sentiment) │  │   │
│  │  └──────────────┘  └──────────────────────┘  └─────────────────┘  │   │
│  │  ┌────────────────────────────────────┐  ┌─────────────────────┐  │   │
│  │  │        Daily Briefing Panel        │  │   Analyze (Input)   │  │   │
│  │  │  (publication-grade narrative)     │  │  (paste article,    │  │   │
│  │  │                                    │  │   get signals back) │  │   │
│  │  └────────────────────────────────────┘  └─────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 3. Data Flow (End-to-End)

### Real-Time Signal Generation (Production)

```
User opens dashboard
    → Frontend calls /get_signals via @gradio/client
    → Backend fetches latest articles from RSS/news (or uses cached)
    → Each article sent to inference backend (V1 or V2)
    → Model returns structured JSON signal
    → Signals aggregated and returned to frontend
    → Frontend renders in signal feed, heatmap, and charts
```

### On-Demand Analysis (Interactive)

```
User pastes article into "Analyze" panel
    → Frontend calls /analyze_news via @gradio/client
    → Backend sends text to inference backend
    → Model returns structured JSON signal
    → Backend returns signal to frontend
    → Frontend displays sentiment, event type, translation, tickers
```

### Daily Briefing Generation

```
Scheduled or on-demand trigger
    → Backend collects top 5-10 signals from past 24 hours
    → Sends aggregated context to model with briefing prompt
    → Model generates publication-grade narrative
    → Stored and served via /get_briefing endpoint
```

## 4. Training Pipeline (Offline, One-Time)

This pipeline runs once to produce the fine-tuned V2 model.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TRAINING PIPELINE (OFFLINE)                          │
│                                                                             │
│  Step 1: Collect Raw Data                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  python scripts/collect_articles.py                                  │   │
│  │  Output: data/raw/articles.jsonl (200+ articles)                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  Step 2: Generate Labels (Teacher Model)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  python scripts/generate_training_data.py                            │   │
│  │  Teacher: qwen3-max (Alibaba Cloud Singapore, free tier)             │   │
│  │  Output: data/training/alpha_signal_train.jsonl                     │   │
│  │  Format: {"messages": [system, user, assistant]} per example         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  Step 3: Fine-Tune on Modal                                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  modal run scripts/modal_finetune.py                                 │   │
│  │  Base: unsloth/Qwen3-8B-Instruct                                    │   │
│  │  Method: QLoRA (4-bit, rank 64)                                      │   │
│  │  GPU: NVIDIA A100 80GB                                               │   │
│  │  Duration: ~30-60 min                                                │   │
│  │  Cost: ~$1.25-2.50 from $280 Modal credits                          │   │
│  │  Output: LoRA adapter weights → merged model                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                         │
│                                    ▼                                         │
│  Step 4: Publish to HF Hub                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Auto-pushed by training script                                      │   │
│  │  Destination: huggingface.co/basilwong/quantum-alpha-qwen3-8b        │   │
│  │  Satisfies: "Well-Tuned" badge (Build Small Hackathon)               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 5. Evaluation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION (Post-Training)                           │
│                                                                             │
│  Tier 1: Classification Metrics (Automated)                                 │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Hold out 10% of training data as validation set                     │   │
│  │  Run both base qwen3-8b AND fine-tuned model on same inputs          │   │
│  │  Compare: Sentiment Accuracy, Event F1, Ticker Jaccard, JSON pass %  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Tier 2: LLM-as-Judge (Automated)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Use qwen3-max to score technical_translation outputs                │   │
│  │  Rubric: Technical Accuracy (1-5), Commercial Relevance (1-5),       │   │
│  │          Readability (1-5)                                           │   │
│  │  Compare base vs fine-tuned on 30 held-out articles                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Tier 3: Information Coefficient (Historical Backtest)                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Collect 3-6 months of historical quantum news + price data          │   │
│  │  Run model on historical articles to generate sentiment scores       │   │
│  │  Calculate Spearman Rank IC vs next-day returns                      │   │
│  │  Measure signal decay over t+1, t+2, t+5 days                       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## 6. Deployment Architecture

### Build Small Hackathon (June 15)

```
┌─────────────────────────────────────────────────────────┐
│  Hugging Face Space: build-small-hackathon/              │
│                      alpha-signal-analysis          │
│                                                         │
│  Hardware: ZeroGPU (free, on-demand A100 allocation)    │
│  SDK: Gradio                                            │
│  Entry: app.py → gradio.Server                          │
│                                                         │
│  Model: basilwong/quantum-alpha-qwen3-8b (fine-tuned)   │
│  Loaded with @spaces.GPU decorator for on-demand GPU    │
│                                                         │
│  Frontend: Custom HTML/CSS/JS served from /             │
│  Backend: FastAPI endpoints via @app.api()              │
└─────────────────────────────────────────────────────────┘
```

### Qwen Cloud Hackathon (July 9)

```
┌─────────────────────────────────────────────────────────┐
│  Alibaba Cloud ECS Instance                             │
│                                                         │
│  ┌───────────────────────────────────────────────────┐ │
│  │  Gradio Server App (same codebase)                │ │
│  │  + ChromaDB (persistent vector memory)            │ │
│  │  + Qwen3-max API calls (Singapore endpoint)       │ │
│  └───────────────────────────────────────────────────┘ │
│                                                         │
│  Memory Agent Features:                                 │
│  * Accumulates sector knowledge across sessions         │
│  * Recalls relevant past events when analyzing new ones │
│  * Forgets outdated information (configurable TTL)      │
│  * Retrieves within limited context windows (RAG)       │
└─────────────────────────────────────────────────────────┘
```

## 7. Configuration Summary

### API Credentials

| Service | Credential | Purpose |
|---------|-----------|---------|
| Alibaba Cloud Model Studio (Singapore) | `sk-ws-H.IIMPYP...QR7QvVBi` | Teacher model (qwen3-max), free tier |
| Modal | Token ID: `ak-O0yCWJtr9WXovf2nFqg9mI` | Fine-tuning GPU compute ($280 credits) |
| Hugging Face | (need from user) | Model hosting, Space deployment |
| GitHub | `ghp_fzvi...LhxWL` | Code repository |

### Model Registry

| Model | Role | Location |
|-------|------|----------|
| qwen3-max | Teacher (data gen + V1 inference + judge) | Alibaba Cloud API |
| qwen3-8b (base) | Baseline comparison | Alibaba Cloud API / HF Hub |
| quantum-alpha-qwen3-8b (fine-tuned) | V2 production model | HF Hub (after training) |

### Quantum Computing Ticker Universe

| Category | Tickers |
|----------|---------|
| Pure-Play | IONQ, RGTI, QBTS, QUBT, INFQ |
| Adjacent | IBM, GOOGL, MSFT, HON, NVDA |
| ETFs | QTUM, WQTM, QNTM |

## 8. Implementation Order

| Step | Action | Dependency | Output |
|------|--------|------------|--------|
| 1 | Collect 200+ raw articles | None | `data/raw/articles.jsonl` |
| 2 | Generate training labels (teacher) | Step 1 + Qwen API key | `data/training/alpha_signal_train.jsonl` |
| 3 | Fine-tune on Modal | Step 2 + Modal auth | `basilwong/quantum-alpha-qwen3-8b` on HF Hub |
| 4 | Evaluate (Tier 1 + Tier 2) | Step 3 | Accuracy metrics + comparison report |
| 5 | Wire inference into Gradio app | Step 3 | Working `/analyze_news` endpoint |
| 6 | Connect data ingestion to app | Step 1 | Working `/get_signals` endpoint |
| 7 | Build daily briefing generator | Step 5 | Working `/get_briefing` endpoint |
| 8 | Polish frontend UI | Step 5-7 | Professional trading terminal |
| 9 | Record demo video | Step 8 | 2-min video for submission |
| 10 | Write Field Notes blog post | Step 4 | Published blog post |
| 11 | Submit Build Small (June 15) | Steps 1-10 | HF Space + video + social post |
| 12 | Add persistent memory (ChromaDB) | Step 11 | Memory-augmented V1 agent |
| 13 | Deploy on Alibaba Cloud ECS | Step 12 | Running backend on Alibaba infra |
| 14 | Submit Qwen Cloud (July 9) | Steps 12-13 | DevPost + video + repo + arch diagram |
