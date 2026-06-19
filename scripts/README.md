# Scripts

Small operational helpers for the next phase.

## Import Memory

Seed the local memory store from a JSONL file:

```bash
python3 scripts/import_memory.py data/raw/articles.jsonl
```

Each JSON line should include one of `text`, `content`, or `body`. Optional fields such as `title`, `source`, `url`, and `date` are preserved as metadata.
