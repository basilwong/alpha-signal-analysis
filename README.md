# Alpha Signal Analysis Platform

**NLP-driven market intelligence for the quantum computing sector.**

Built for the [Build Small Hackathon](https://huggingface.co/build-small-hackathon) (Hugging Face + Gradio, June 5-15, 2026).

## Overview

The quantum computing sector is one of the most technically complex and misunderstood areas in public markets. Most investors lack the physics background to distinguish between genuine breakthroughs and incremental noise. This platform bridges that gap by using a fine-tuned small language model to translate dense technical announcements into actionable investment signals and strategic briefings.

## Architecture

```
DATA INGESTION          NLP PROCESSING              PRESENTATION
+--------------+       +-------------------+       +------------------+
| SEC EDGAR    |       |                   |       | Gradio Server    |
| arXiv API    | ----> | Fine-Tuned        | ----> | (FastAPI Backend) |
| Yahoo Finance|       | Qwen3-8B          |       |                  |
| RSS Feeds    |       | (on Modal)        |       | Custom HTML/JS   |
+--------------+       +-------------------+       | Trading Terminal  |
                                                   +------------------+
```

## Features

- **Signal Engine**: Real-time sentiment scoring, event classification, and catalyst detection across quantum computing tickers
- **Daily Briefings**: Publication-grade analyst narratives explaining why technical milestones matter commercially
- **Technical Translation**: Converts physics jargon into investment-relevant language
- **Custom Dashboard**: Professional trading terminal UI (targeting the "Off-Brand" badge)

## Target Badges

| Badge | Strategy |
|-------|----------|
| Well-Tuned | Fine-tuned Qwen3-8B published on HF Hub |
| Off-Brand | Custom HTML5/CSS3/JS frontend via `gradio.Server` |
| Field Notes | Engineering blog post documenting the build |

## Quantum Computing Universe

### Pure-Play Companies
- IonQ (IONQ) - Trapped Ion
- Rigetti Computing (RGTI) - Superconducting
- D-Wave Quantum (QBTS) - Quantum Annealing
- Quantum Computing Inc (QUBT)
- Infleqtion (INFQ) - Neutral Atom

### Adjacent / Large-Cap
- IBM (IBM) - Superconducting
- Alphabet/Google (GOOGL) - Superconducting (Willow)
- Microsoft (MSFT) - Topological
- Honeywell/Quantinuum (HON) - Trapped Ion
- NVIDIA (NVDA) - Quantum Simulation

## Project Structure

```
alpha-signal-analysis/
├── src/
│   ├── ingestion/       # Data collection scripts (SEC, arXiv, RSS, Yahoo)
│   ├── model/           # Fine-tuning scripts, model config, inference
│   ├── signals/         # Signal generation and scoring logic
│   └── api/             # Gradio Server backend endpoints
├── frontend/
│   ├── css/             # Custom dashboard styles
│   └── js/             # Frontend logic and chart rendering
├── data/
│   ├── raw/             # Raw ingested data
│   ├── processed/       # Cleaned and structured data
│   └── training/        # Fine-tuning dataset (instruction pairs)
├── notebooks/           # Exploration and prototyping notebooks
├── docs/                # Architecture docs and field notes
├── scripts/             # Utility scripts (deployment, data prep)
├── requirements.txt     # Python dependencies
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Base Model | Qwen3-8B-Instruct |
| Fine-Tuning | Unsloth + QLoRA |
| Training Compute | Modal (serverless GPU) |
| Inference | vLLM on Modal |
| Backend | Gradio Server (FastAPI) |
| Frontend | Custom HTML5/CSS3/JS + Chart.js |
| Data Sources | SEC EDGAR, arXiv, Yahoo Finance, RSS |
| Hosting | Hugging Face Spaces |

## Timeline

| Phase | Dates | Focus |
|-------|-------|-------|
| 1 | June 5-7 | Data ingestion pipeline + training dataset generation |
| 2 | June 8-10 | Model fine-tuning + deployment |
| 3 | June 11-13 | Dashboard frontend + backend integration |
| 4 | June 14-15 | Polish, demo video, blog post, submission |

## Getting Started

```bash
# Clone the repository
git clone https://github.com/basilwong/alpha-signal-analysis.git
cd alpha-signal-analysis

# Install dependencies
pip install -r requirements.txt

# Run the app locally (after model deployment)
python src/api/app.py
```

## License

MIT
