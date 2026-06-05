# Development Guide

This document covers how to set up the project locally, preview the frontend, and deploy to Hugging Face Spaces.

## Prerequisites

You will need Python 3.10 or higher installed on your machine. The project uses Gradio's Server mode, which requires Gradio 5.0+.

## Quick Start: Preview the Frontend

To view the trading terminal dashboard locally, follow these steps:

```bash
# 1. Clone the repository
git clone https://github.com/basilwong/quantum-alpha-intelligence.git
cd quantum-alpha-intelligence

# 2. Create a virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the minimal dependencies for frontend preview
pip install gradio fastapi uvicorn

# 4. Run the Gradio Server app
python src/api/app.py
```

The app will launch at `http://localhost:7860`. Open this URL in your browser to see the custom trading terminal dashboard.

The frontend is served as static HTML/CSS/JS through Gradio's Server mode (which is built on FastAPI). The backend API endpoints are also accessible at this URL.

## Project Layout

| Directory | Purpose |
|-----------|---------|
| `src/api/` | Gradio Server backend (FastAPI endpoints) |
| `src/ingestion/` | Data collection scripts (SEC EDGAR, arXiv, RSS, Yahoo Finance) |
| `src/model/` | Fine-tuning scripts, model config, inference wrappers |
| `src/signals/` | Signal generation, scoring, and aggregation logic |
| `frontend/` | Custom HTML5/CSS3/JS trading terminal UI |
| `data/` | Raw, processed, and training data (gitignored for large files) |
| `notebooks/` | Jupyter notebooks for exploration and prototyping |
| `scripts/` | Utility scripts for deployment, data prep, etc. |
| `docs/` | Architecture documentation and field notes |

## API Endpoints

Once the server is running, the following endpoints are available:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Serves the custom frontend dashboard |
| `/health` | GET | Health check (returns JSON status) |
| `/gradio_api/call/get_signals` | POST | Retrieve alpha signals for quantum tickers |
| `/gradio_api/call/get_briefing` | POST | Get the daily strategic briefing |
| `/gradio_api/call/analyze_news` | POST | Analyze a news article for signals |
| `/gradio_api/call/get_sector_overview` | POST | Get aggregated sector overview |

You can also view auto-generated API docs at `/gradio_api/info`.

## Installing Full Dependencies

For development work beyond frontend preview (data ingestion, model training, etc.):

```bash
pip install -r requirements.txt
```

Note: Some packages like `unsloth`, `vllm`, and `torch` with CUDA support require a GPU environment. For training, use Modal or Google Colab.

## Environment Variables

Create a `.env` file in the project root for API keys and configuration:

```bash
# .env (do not commit this file)
HF_TOKEN=your_hugging_face_token
MODAL_TOKEN_ID=your_modal_token_id
MODAL_TOKEN_SECRET=your_modal_token_secret
```

## Running Tests

```bash
# Run the health check
curl http://localhost:7860/health

# Test the analyze endpoint via Gradio client
python -c "
from gradio_client import Client
client = Client('http://localhost:7860')
result = client.predict('IonQ announces 35 logical qubits', 'news', api_name='/analyze_news')
print(result)
"
```

## Deployment to Hugging Face Spaces

The production app is deployed as a Hugging Face Space under the `build-small-hackathon` organization. See the `docs/` folder for deployment instructions.

The Space repo mirrors the structure of this repository but only includes the files needed for production (no training scripts or notebooks).
