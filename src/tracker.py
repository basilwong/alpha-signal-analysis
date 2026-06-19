"""Append-only tracking for trades, evals, and backtest results."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from src.config import DEFAULT_RUNS_DIR


COLLECTION_FILES = {
    "trades": "trades.jsonl",
    "evals": "evals.jsonl",
    "backtests": "backtests.jsonl",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def parse_json_object(raw: str, field_name: str) -> dict[str, Any]:
    """Parse an optional JSON object from a CLI argument."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return value


class RunTracker:
    """Persist operational records to JSONL files."""

    def __init__(self, base_dir: Path | str = DEFAULT_RUNS_DIR):
        self.base_dir = Path(base_dir)

    def record_trade(
        self,
        strategy: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        source: str = "manual",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a trade fill or paper trade."""
        normalized_side = side.lower()
        if normalized_side not in {"buy", "sell", "short", "cover"}:
            raise ValueError("side must be one of: buy, sell, short, cover")
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        if price < 0:
            raise ValueError("price must be non-negative")

        record = self._base_record("trade")
        record.update(
            {
                "strategy": strategy,
                "symbol": symbol.upper(),
                "side": normalized_side,
                "quantity": quantity,
                "price": price,
                "notional": quantity * price,
                "source": source,
                "metadata": metadata or {},
            }
        )
        return self._append("trades", record)

    def record_eval(
        self,
        name: str,
        status: str,
        metrics: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record an evaluation run or validation check."""
        normalized_status = status.lower()
        if normalized_status not in {"pass", "fail", "warn", "skip"}:
            raise ValueError("status must be one of: pass, fail, warn, skip")

        record = self._base_record("eval")
        record.update(
            {
                "name": name,
                "status": normalized_status,
                "metrics": metrics or {},
                "metadata": metadata or {},
            }
        )
        return self._append("evals", record)

    def record_backtest(
        self,
        strategy: str,
        dataset: str,
        start_date: str,
        end_date: str,
        metrics: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Record a backtest result summary."""
        if not metrics:
            raise ValueError("metrics must not be empty")

        record = self._base_record("backtest")
        record.update(
            {
                "strategy": strategy,
                "dataset": dataset,
                "start_date": start_date,
                "end_date": end_date,
                "metrics": metrics,
                "metadata": metadata or {},
            }
        )
        return self._append("backtests", record)

    def list_records(self, collection: str, limit: int | None = None) -> list[dict[str, Any]]:
        """Read records from newest to oldest."""
        path = self._path(collection)
        if not path.exists():
            return []

        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
        records.reverse()
        return records[:limit] if limit is not None else records

    def _base_record(self, kind: str) -> dict[str, Any]:
        return {"kind": kind, "created_at": _utc_now()}

    def _append(self, collection: str, record: dict[str, Any]) -> dict[str, Any]:
        payload = json.dumps(record, sort_keys=True)
        record["record_id"] = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]

        path = self._path(collection)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def _path(self, collection: str) -> Path:
        if collection not in COLLECTION_FILES:
            raise ValueError(f"Unknown collection: {collection}")
        return self.base_dir / COLLECTION_FILES[collection]
