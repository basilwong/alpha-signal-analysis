# TODO

## Next

- [ ] Define the trade adapter interface for paper/live fills.
- [ ] Add aggregate reporting over `data/runs/*.jsonl`.
- [ ] Add structured validation for `signal_vector` outputs.
- [ ] Add memory snapshot IDs to backtest records.
- [ ] Add a small non-sensitive fixture for tracker/report tests.
- [ ] Decide whether JSONL remains sufficient or needs SQLite once records grow.
- [ ] Rebuild frontend deployment automation using environment variables or platform secrets.
- [ ] Connect frontend dashboard data to `data/runs/` outputs.

## Done

- [x] Rename repo references to `alpha-signal-analysis`.
- [x] Remove prior hackathon generated data, logs, duplicate apps, old frontends, and submission docs.
- [x] Remove current-tree hardcoded API keys by deleting old experimental scripts.
- [x] Remove the UI framework from the active runtime.
- [x] Add append-only tracking for trades, evals, and backtests.
- [x] Restore the deployable frontend assets and server entry point.
