#!/usr/bin/env python3
"""Import article or note JSONL into the local memory store."""

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.memory import MemoryStore  # noqa: E402


def import_jsonl(input_path: Path, store: MemoryStore) -> int:
    count = 0
    with input_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            item = json.loads(line)
            text = item.get("text") or item.get("content") or item.get("body")
            if not text:
                continue
            store.add(
                text=text,
                source=item.get("source", "manual_note"),
                title=item.get("title", ""),
                url=item.get("url", ""),
                observed_at=item.get("date") or item.get("observed_at"),
                metadata={key: value for key, value in item.items() if key not in {"text", "content", "body"}},
            )
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSONL file with text/content/body fields")
    parser.add_argument(
        "--memory-path",
        type=Path,
        default=None,
        help="Override the memory JSONL path",
    )
    args = parser.parse_args()

    store = MemoryStore(args.memory_path) if args.memory_path else MemoryStore()
    count = import_jsonl(args.input, store)
    print(f"Imported {count} records into {store.path}")


if __name__ == "__main__":
    main()
