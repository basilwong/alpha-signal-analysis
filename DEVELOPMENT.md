# Development Guide

## Local Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Qwen Cloud credentials are optional for tracking commands and required only for `analyze` or live frontend inference:

```bash
export DASHSCOPE_API_KEY="your-qwen-cloud-key"
# or
export QWEN_API_KEY="your-qwen-cloud-key"
```

## Commands

```bash
# Record operational artifacts
python3 app.py trade --strategy dev --symbol IONQ --side buy --quantity 1 --price 1
python3 app.py eval --name smoke --status pass --metrics '{"ok": 1}'
python3 app.py backtest --strategy dev --dataset smoke --start-date 2026-01-01 --end-date 2026-01-02 --metrics '{"return": 0.0}'

# Inspect records
python3 app.py list trades
python3 app.py list evals
python3 app.py list backtests

# Run the frontend server
python3 app_server_fixed.py

# Run tests
python3 -m unittest discover -s tests

# Compile-check Python files
python3 -m compileall app.py src scripts
```

## Data Policy

Generated data stays local:

- `data/raw/`
- `data/processed/`
- `data/memory/`
- `data/runs/`
- `data/logs/`
- any future `data/eval/`, `data/market/`, or `data/training/` directories

Commit only small fixtures required for tests or documentation.

## Adding Features

Keep the active path narrow:

1. Add tracking behavior in `src/tracker.py`.
2. Add durable memory behavior in `src/memory.py`.
3. Add model-call orchestration in `src/agent.py` only when a model call is actually required.
4. Add CLI surface in `app.py`.
5. Add focused tests under `tests/`.

Do not reintroduce UI frameworks, generated datasets, hardcoded API keys, or one-off experiment scripts.
