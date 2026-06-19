# Alpha Signal Analysis — Memory Agent

**Qwen Cloud Global AI Hackathon 2026 — Memory Agent Track**

Memory-aware signal analysis and operational tracking for the quantum computing sector. The agent generates alpha trading signals for quantum computing stocks and gets smarter over time by accumulating knowledge from multiple data streams, tracking its own prediction accuracy, and connecting signals across time.

## Architecture

```
┌─────────────────────────────────────────────────┐
│  HF Space (Frontend UI)                          │
│  - Chat interface for interacting with agent     │
│  - Memory visualization                          │
│  - Signal timeline + evaluation dashboard        │
└──────────────────────┬──────────────────────────┘
                       │ HTTPS API calls
                       ▼
┌─────────────────────────────────────────────────┐
│  Alibaba Cloud ECS (Free Tier t5/t7)             │
│  - FastAPI backend (port 8000)                   │
│  - SQLite memory database (persistent)           │
│  - Memory retrieval engine                       │
│  - Forgetting/consolidation logic                │
│  - Agent orchestration loop                      │
│  - Scheduled data ingestion (cron)               │
└──────────┬──────────────────────┬───────────────┘
           │                      │
           ▼                      ▼
┌──────────────────┐   ┌──────────────────────────┐
│  DashScope API   │   │  Modal vLLM endpoint      │
│  qwen3-max       │   │  Fine-tuned Nemotron-7B   │
│  (memory-augmented│   │  (fast signal generation) │
│   reasoning)     │   │                           │
└──────────────────┘   └──────────────────────────┘
```

## Key Features

- **Persistent Memory**: SQLite-based memory store with sector knowledge, signal history, and user preferences
- **Memory-Augmented Reasoning**: Injects relevant past knowledge into LLM prompts for more informed predictions
- **Forgetting & Consolidation**: TTL-based expiry, relevance pruning, and contradiction resolution
- **Self-Evaluation**: Tracks prediction accuracy and adjusts confidence based on track record
- **Multi-Backend Inference**: Supports DashScope (Qwen Cloud) and Modal (fine-tuned models)
- **Scheduled Ingestion**: Automated news ingestion via RSS feeds
- **CLI Tracker**: Record trades, evals, and backtests via command line

## Tickers Covered

| Category | Tickers | Signal Range |
|----------|---------|--------------|
| Pure-Play Quantum | IONQ, RGTI, QBTS, QUBT, QNT | -2.0 to +2.0 |
| Diversified (Capped) | IBM, GOOGL, MSFT, HON, NVDA | Always 0.0 |

## Quick Start — Memory Agent Backend

```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export DASHSCOPE_API_KEY="your_key"
export DASHSCOPE_BASE_URL="https://ws-wuyspztgv1cyxvbr.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
export MEMORY_DB_PATH="./data/memory/memory.db"

# Seed the memory database
python -c "
from agent.seed_data import SEED_FACTS
from agent.memory import MemoryStore
m = MemoryStore('./data/memory/memory.db')
for f in SEED_FACTS:
    m.store_knowledge(f['ticker'], f['type'], f['content'], 'seed')
print(f'Seeded {len(SEED_FACTS)} facts. Stats: {m.get_memory_stats()}')
"

# Start the backend
uvicorn agent.server:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check + memory stats |
| GET | `/api/memory/stats` | Memory statistics |
| GET | `/api/memory/knowledge` | Retrieve stored knowledge |
| GET | `/api/memory/signals` | Retrieve signal history |
| POST | `/api/analyze` | Analyze article with memory-augmented reasoning |
| POST | `/api/memory/forget` | Trigger forgetting cycle |
| POST | `/api/memory/seed` | Seed initial knowledge |

### Example: Analyze an Article

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "IonQ announces 50 algorithmic qubits on latest trapped-ion processor", "source": "news"}'
```

## CLI Tracker

The CLI provides operational tracking independent of the agent:

```bash
# Track a trade
python3 app.py trade --strategy memory-agent-v1 --symbol IONQ --side buy --quantity 25 --price 42.50

# Track an eval
python3 app.py eval --name signal-json-schema --status pass --metrics '{"valid_rate": 0.98}'

# Track a backtest
python3 app.py backtest --strategy memory-agent-v1 --dataset quantum-news-2026 --start-date 2026-06-01 --end-date 2026-06-15

# List records
python3 app.py list trades --limit 10
```

## Deployment (Alibaba Cloud ECS)

1. Sign up for Alibaba Cloud free tier
2. Copy `infra/variables.tfvars.example` to `infra/variables.tfvars` and fill in credentials
3. Run `./deploy.sh`

See `infra/main.tf` for the full Terraform configuration.

## Memory System Design

### Three Memory Types

1. **Sector Knowledge**: Facts about companies, technologies, milestones (with TTL)
2. **Signal History**: Previous predictions and their outcomes (for self-evaluation)
3. **User Preferences**: Risk tolerance, focus areas (persistent)

### Forgetting Mechanisms

- **TTL Expiry**: Memories expire after configurable period (default 90 days)
- **Relevance Pruning**: Unused memories (access_count=0) pruned after 30 days
- **Contradiction Resolution**: New facts reduce confidence of contradicting old facts
- **Consolidation**: Old signals merged into weekly summaries

## Project Structure

```
alpha-signal-analysis/
├── agent/                     # Memory Agent backend (NEW)
│   ├── __init__.py
│   ├── config.py              # Configuration constants
│   ├── memory.py              # SQLite memory store
│   ├── retrieval.py           # Memory retrieval engine
│   ├── forgetting.py          # Forgetting/consolidation logic
│   ├── inference.py           # Model inference (DashScope + Modal)
│   ├── ingestion.py           # Data source connectors
│   ├── server.py              # FastAPI backend
│   └── seed_data.py           # Initial sector knowledge
├── infra/                     # Terraform IaC (NEW)
│   ├── main.tf
│   └── variables.tfvars.example
├── src/                       # Original CLI modules
│   ├── agent.py
│   ├── config.py
│   ├── memory.py
│   ├── prompts.py
│   ├── sector_data.py
│   └── tracker.py
├── frontend_v2/               # Static frontend assets
├── docs/
│   └── architecture_diagram.md
├── scripts/
├── tests/
├── app.py                     # CLI tracker
├── app_server_fixed.py        # Frontend server
├── deploy.sh                  # ECS deployment script
├── requirements.txt
└── README.md
```

## Frontend

The frontend stack is preserved in `frontend_v2/` and served by `app_server_fixed.py`:

```bash
python3 app_server_fixed.py
```

## Test

```bash
python3 -m unittest discover -s tests
python3 -m compileall app.py src agent scripts
```

## Security Notes

No API keys are committed in the current tree. Use environment variables or platform secrets for all credentials. If any previously committed key was real, rotate it because Git history may still contain it.

## Hackathon Submission

- **Track**: Memory Agent
- **Deadline**: July 9, 2026
- **Infrastructure**: Alibaba Cloud ECS (free tier) + DashScope API (Singapore)
- **Key Differentiator**: Self-improving trading signal agent that demonstrates memory accumulation, forgetting, and accuracy tracking over time
