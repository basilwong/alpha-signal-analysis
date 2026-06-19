# Roadmap

## Goal

Build a minimal memory-aware trading research harness that can record trades, accumulate eval and backtest results, and preserve enough logging to debug agent behavior.

## Phase 1: Operational Ledger

- Keep one CLI entry point.
- Store trades, evals, and backtests as append-only JSONL.
- Keep generated records under gitignored `data/runs/`.
- Keep runtime logs under gitignored `data/logs/`.
- Preserve the frontend server as a separate deployable surface.

## Phase 2: Evaluation Discipline

- Define a stable eval result schema.
- Add structured validation for agent outputs.
- Add aggregate reports over `data/runs/evals.jsonl`.
- Tie each model or prompt change to an eval record.

## Phase 3: Backtesting Loop

- Standardize backtest inputs and result metrics.
- Connect backtest runs to strategy IDs and memory snapshots.
- Add comparison reports across strategies and time windows.
- Record trade decisions from paper/live adapters through `RunTracker`.

## Phase 3b: Frontend Deployment

- Keep the frontend assets under `frontend_v2/`.
- Serve them through `app_server_fixed.py`.
- Rebuild deployment automation without hardcoded tokens.
- Feed dashboard state from tracked eval/backtest outputs instead of committed generated datasets.

## Phase 4: Memory Quality

- Add memory freshness scoring.
- Add conflict handling for stale or contradictory company facts.
- Record which memory items influenced each decision.
- Compare one-shot versus memory-assisted results using tracked evals.
