"""Small JSONL memory store for the Qwen Cloud hackathon prototype."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from src.config import DEFAULT_MEMORY_PATH, OUTPUT_TICKERS


TOKEN_RE = re.compile(r"[A-Za-z0-9$._-]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _tokens(text: str) -> set[str]:
    return {token.lower().lstrip("$") for token in TOKEN_RE.findall(text)}


def _record_id(text: str, source: str, title: str, observed_at: str) -> str:
    payload = "\n".join([source, title, observed_at, text.strip()])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass
class MemoryRecord:
    """A persisted market-memory item."""

    text: str
    source: str = "manual_note"
    title: str = ""
    url: str = ""
    observed_at: str = field(default_factory=_utc_now)
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    record_id: str = ""

    def __post_init__(self) -> None:
        if not self.record_id:
            self.record_id = _record_id(self.text, self.source, self.title, self.observed_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "observed_at": self.observed_at,
            "source": self.source,
            "title": self.title,
            "url": self.url,
            "tags": self.tags,
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MemoryRecord":
        return cls(
            record_id=data.get("record_id", ""),
            observed_at=data.get("observed_at") or _utc_now(),
            source=data.get("source", "manual_note"),
            title=data.get("title", ""),
            url=data.get("url", ""),
            tags=list(data.get("tags", [])),
            text=data.get("text", ""),
            metadata=dict(data.get("metadata", {})),
        )


class MemoryStore:
    """Append-only JSONL memory with simple lexical retrieval."""

    def __init__(self, path: Path | str = DEFAULT_MEMORY_PATH):
        self.path = Path(path)

    def all_records(self) -> list[MemoryRecord]:
        if not self.path.exists():
            return []

        records = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                records.append(MemoryRecord.from_dict(json.loads(line)))
        return records

    def add(
        self,
        text: str,
        source: str = "manual_note",
        title: str = "",
        url: str = "",
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        observed_at: str | None = None,
    ) -> MemoryRecord:
        """Add a memory item, returning an existing duplicate when present."""
        record = MemoryRecord(
            text=text.strip(),
            source=source,
            title=title.strip(),
            url=url.strip(),
            tags=tags or infer_tags(text),
            metadata=metadata or {},
            observed_at=observed_at or _utc_now(),
        )

        existing = {item.record_id: item for item in self.all_records()}
        if record.record_id in existing:
            return existing[record.record_id]

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record.to_dict(), sort_keys=True) + "\n")
        return record

    def search(self, query: str, limit: int = 5) -> list[dict[str, Any]]:
        """Return the highest-overlap memories for a query."""
        query_terms = _tokens(query)
        if not query_terms:
            return []

        scored = []
        for record in self.all_records():
            haystack = " ".join([record.title, record.text, " ".join(record.tags)])
            terms = _tokens(haystack)
            overlap = query_terms.intersection(terms)
            ticker_hits = {ticker.lower() for ticker in OUTPUT_TICKERS}.intersection(overlap)
            score = len(overlap) + (3 * len(ticker_hits))
            if score > 0:
                scored.append((score, record.observed_at, record))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [record.to_dict() for _, _, record in scored[:limit]]


def infer_tags(text: str) -> list[str]:
    """Infer ticker tags from free text."""
    terms = _tokens(text)
    return sorted(ticker for ticker in OUTPUT_TICKERS if ticker.lower() in terms)
