"""CLI entry point for Alpha Signal Analysis."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from src.agent import AgentConfigError, AlphaSignalMemoryAgent
from src.config import DEFAULT_LOG_PATH, DEFAULT_RUNS_DIR, SOURCE_TYPES
from src.tracker import RunTracker, parse_json_object


def configure_logging(log_path: Path = DEFAULT_LOG_PATH, verbose: bool = False) -> None:
    """Configure console and file logging."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Track trades, evals, and backtests.")
    parser.add_argument("--runs-dir", type=Path, default=DEFAULT_RUNS_DIR)
    parser.add_argument("--log-file", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--verbose", action="store_true")

    commands = parser.add_subparsers(dest="command", required=True)

    trade = commands.add_parser("trade", help="Record a trade fill or paper trade.")
    trade.add_argument("--strategy", required=True)
    trade.add_argument("--symbol", required=True)
    trade.add_argument("--side", required=True, choices=["buy", "sell", "short", "cover"])
    trade.add_argument("--quantity", required=True, type=float)
    trade.add_argument("--price", required=True, type=float)
    trade.add_argument("--source", default="manual")
    trade.add_argument("--metadata", default="{}")

    eval_cmd = commands.add_parser("eval", help="Record an evaluation result.")
    eval_cmd.add_argument("--name", required=True)
    eval_cmd.add_argument("--status", required=True, choices=["pass", "fail", "warn", "skip"])
    eval_cmd.add_argument("--metrics", default="{}")
    eval_cmd.add_argument("--metadata", default="{}")

    backtest = commands.add_parser("backtest", help="Record a backtest summary.")
    backtest.add_argument("--strategy", required=True)
    backtest.add_argument("--dataset", required=True)
    backtest.add_argument("--start-date", required=True)
    backtest.add_argument("--end-date", required=True)
    backtest.add_argument("--metrics", required=True)
    backtest.add_argument("--metadata", default="{}")

    list_cmd = commands.add_parser("list", help="List tracked records.")
    list_cmd.add_argument("collection", choices=["trades", "evals", "backtests"])
    list_cmd.add_argument("--limit", type=int, default=20)

    analyze = commands.add_parser("analyze", help="Run the Qwen memory agent on text.")
    analyze.add_argument("--source", choices=SOURCE_TYPES, default="news")
    analyze.add_argument("--text")
    analyze.add_argument("--file", type=Path)
    analyze.add_argument("--no-persist", action="store_true")

    return parser


def run(args: argparse.Namespace) -> dict | list:
    tracker = RunTracker(args.runs_dir)

    if args.command == "trade":
        return tracker.record_trade(
            strategy=args.strategy,
            symbol=args.symbol,
            side=args.side,
            quantity=args.quantity,
            price=args.price,
            source=args.source,
            metadata=parse_json_object(args.metadata, "metadata"),
        )

    if args.command == "eval":
        return tracker.record_eval(
            name=args.name,
            status=args.status,
            metrics=parse_json_object(args.metrics, "metrics"),
            metadata=parse_json_object(args.metadata, "metadata"),
        )

    if args.command == "backtest":
        return tracker.record_backtest(
            strategy=args.strategy,
            dataset=args.dataset,
            start_date=args.start_date,
            end_date=args.end_date,
            metrics=parse_json_object(args.metrics, "metrics"),
            metadata=parse_json_object(args.metadata, "metadata"),
        )

    if args.command == "list":
        return tracker.list_records(args.collection, limit=args.limit)

    if args.command == "analyze":
        if args.file:
            text = args.file.read_text(encoding="utf-8")
        elif args.text:
            text = args.text
        else:
            text = sys.stdin.read()

        agent = AlphaSignalMemoryAgent()
        result = agent.analyze(text=text, source=args.source, persist=not args.no_persist)
        return result

    raise ValueError(f"Unsupported command: {args.command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_file, args.verbose)

    try:
        result = run(args)
    except (AgentConfigError, ValueError) as exc:
        logging.getLogger(__name__).error("%s", exc)
        print(json.dumps({"error": str(exc)}, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
