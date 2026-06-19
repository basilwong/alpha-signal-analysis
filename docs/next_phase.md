# Next Phase: Tracking-First Research Harness

## Cleanup Boundary

The active repository should contain only the reusable foundation for the next phase:

- One CLI entry point.
- One optional Qwen Cloud agent wrapper.
- One local memory abstraction.
- One append-only tracker for trades, evals, and backtests.
- One preserved frontend server for visualization/deployment.
- Focused docs and tests.

Removed from the active tree:

- Prior hackathon submission docs and workflow stubs.
- Duplicate app and frontend implementations.
- Modal fine-tuning and batch-monitoring scripts.
- Generated training, evaluation, market, and log files.
- Current-tree files containing hardcoded API keys.

## Architecture

```text
trade adapter / eval runner / backtest runner
  -> RunTracker
  -> data/runs/{trades,evals,backtests}.jsonl
  -> data/logs/app.log
```

Optional model path:

```text
input text
  -> MemoryStore.search()
  -> Qwen Cloud chat completion
  -> structured signal JSON
  -> MemoryStore.add()
  -> RunTracker records eval/backtest/trade outcome separately
```

## Direction

The useful product surface is not a web app. It is the ledger:

1. Every trade decision can be recorded with strategy, source, and metadata.
2. Every eval can be recorded with status and metrics.
3. Every backtest can be recorded with dataset, date range, and metrics.
4. Debug logs are persisted locally and ignored by Git.

Once this is stable, add adapters and reports around the ledger instead of adding another UI layer.

## Frontend

The frontend is retained as a separate deployment surface:

```text
frontend_v2/
  -> app_server_fixed.py
```

Keep deployment credentials out of source control. The previous token-bearing deploy helpers should not be restored.
