"""Project configuration without environment-specific secrets."""

import os
from pathlib import Path


PROJECT_NAME = "Alpha Signal Analysis"

DATA_DIR = Path(os.environ.get("ALPHA_DATA_DIR", "data"))
DEFAULT_MEMORY_PATH = Path(
    os.environ.get("ALPHA_MEMORY_PATH", str(DATA_DIR / "memory" / "events.jsonl"))
)
DEFAULT_RUNS_DIR = Path(os.environ.get("ALPHA_RUNS_DIR", str(DATA_DIR / "runs")))
DEFAULT_LOG_PATH = Path(os.environ.get("ALPHA_LOG_PATH", str(DATA_DIR / "logs" / "app.log")))

QWEN_BASE_URL = os.environ.get(
    "QWEN_BASE_URL",
    "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen-plus")
QWEN_TEMPERATURE = float(os.environ.get("QWEN_TEMPERATURE", "0.2"))
QWEN_MAX_TOKENS = int(os.environ.get("QWEN_MAX_TOKENS", "1600"))

SOURCE_TYPES = [
    "news",
    "research",
    "sec_filing",
    "press_release",
    "earnings_call",
    "analyst_note",
    "manual_note",
]

PURE_PLAY_TICKERS = {
    "IONQ": {"name": "IonQ", "technology": "trapped ion", "max_score": 2.0},
    "RGTI": {"name": "Rigetti Computing", "technology": "superconducting", "max_score": 2.0},
    "QBTS": {"name": "D-Wave Quantum", "technology": "quantum annealing", "max_score": 2.0},
    "QUBT": {"name": "Quantum Computing Inc.", "technology": "photonic/quantum optimization", "max_score": 2.0},
    "QNT": {"name": "Quantinuum", "technology": "trapped ion", "max_score": 2.0},
}

ADJACENT_TICKERS = {
    "IBM": {"name": "IBM", "technology": "superconducting", "max_score": 0.15},
    "HON": {"name": "Honeywell", "technology": "trapped ion exposure", "max_score": 0.30},
}

INACTIVE_TICKERS = {
    "MSFT": "Quantum exposure is too small relative to total business.",
    "GOOGL": "Quantum exposure is too small relative to total business.",
    "NVDA": "Moves primarily on AI/GPU demand, not quantum news.",
}

ACTIVE_TICKERS = list(PURE_PLAY_TICKERS) + list(ADJACENT_TICKERS)
OUTPUT_TICKERS = ACTIVE_TICKERS + list(INACTIVE_TICKERS)
