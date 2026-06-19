# Alpha Signal Analysis

Memory-aware signal analysis and operational tracking for the quantum computing sector.

The active repo is intentionally small: no generated datasets and no one-off hackathon experiment scripts. The primary runtime is a CLI plus importable Python modules for recording trades, eval results, backtest summaries, memory records, and debug logs. A deployable frontend is retained separately for visualization and live signal debugging.

## Current Shape

```text
alpha-signal-analysis/
|-- app.py                 # CLI for tracking and optional agent analysis
|-- app_server_fixed.py    # Frontend server for deployment
|-- frontend_v2/           # Static frontend assets
|-- src/
|   |-- agent.py           # Optional Qwen Cloud wrapper
|   |-- config.py          # Environment-driven configuration
|   |-- memory.py          # Local JSONL memory store
|   |-- prompts.py         # Agent prompt
|   |-- sector_data.py     # Quantum ticker context
|   `-- tracker.py         # Trades/evals/backtests JSONL tracker
|-- scripts/
|   `-- import_memory.py   # Seed memory from JSONL
|-- data/
|   |-- memory/            # Local memory store, gitignored
|   |-- runs/              # Trades/evals/backtests, gitignored
|   `-- logs/              # Runtime logs, gitignored
|-- tests/
`-- docs/
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Set a Qwen key only if you use the `analyze` command:

```bash
export DASHSCOPE_API_KEY="your-qwen-cloud-key"
# or
export QWEN_API_KEY="your-qwen-cloud-key"
```

## Track Trades

```bash
python3 app.py trade \
  --strategy memory-agent-v1 \
  --symbol IONQ \
  --side buy \
  --quantity 25 \
  --price 42.50 \
  --source paper \
  --metadata '{"signal_score": 1.2}'
```

## Track Evals

```bash
python3 app.py eval \
  --name signal-json-schema \
  --status pass \
  --metrics '{"valid_rate": 0.98, "sample_size": 50}'
```

## Track Backtests

```bash
python3 app.py backtest \
  --strategy memory-agent-v1 \
  --dataset quantum-news-2026 \
  --start-date 2026-06-01 \
  --end-date 2026-06-15 \
  --metrics '{"sharpe": 1.1, "max_drawdown": -0.08}'
```

## Read Results

```bash
python3 app.py list trades --limit 10
python3 app.py list evals --limit 10
python3 app.py list backtests --limit 10
```

Records are stored under `data/runs/` by default. Logs go to `data/logs/app.log`.

## Frontend

The frontend stack is preserved in `frontend_v2/` and served by `app_server_fixed.py`:

```bash
python3 app_server_fixed.py
```

This server expects historical prediction/market files if you want populated dashboards, but it can still serve the static frontend without committing generated data. Deployment automation should use environment variables or platform secrets; do not commit tokens.

## Optional Agent Analysis

```bash
python3 app.py analyze --source news --file data/raw/article.txt
```

The agent can retrieve and write memory in `data/memory/events.jsonl`, but the core trade/eval/backtest tracker does not require a model call.

## Test

```bash
python3 -m unittest discover -s tests
python3 -m compileall app.py src scripts
```

## Security Notes

No API keys are committed in the current tree. If any previously committed key was real, rotate it because Git history may still contain it.
