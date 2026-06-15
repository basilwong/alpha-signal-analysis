"""
Configuration for the Alpha Signal Analysis Platform.

Updated: fix/label-quality branch
Changes:
- Fix 1: MSFT/GOOGL/NVDA moved to INACTIVE_TICKERS, QNT added to PURE_PLAY_TICKERS
- Fix 7: LIQUIDITY_TIERS added
- Score ranges documented per ticker
"""

# ============================================================
# Quantum Computing Ticker Universe
# ============================================================

# Fix 1: Updated ticker universe based on label quality analysis
PURE_PLAY_TICKERS = {
    "IONQ": {"name": "IonQ", "technology": "Trapped Ion", "max_score": 2.0},
    "RGTI": {"name": "Rigetti Computing", "technology": "Superconducting", "max_score": 2.0},
    "QBTS": {"name": "D-Wave Quantum", "technology": "Quantum Annealing", "max_score": 2.0},
    "QUBT": {"name": "Quantum Computing Inc", "technology": "Neutral Atom", "max_score": 2.0},
    "QNT": {"name": "Quantinuum", "technology": "Trapped Ion", "max_score": 2.0,
            "note": "IPO'd June 4, 2026 on NASDAQ. Spun off from Honeywell. Direct competitor to IONQ."},
}

ADJACENT_TICKERS = {
    "IBM": {"name": "IBM", "technology": "Superconducting", "quantum_revenue_pct": 2.0,
            "max_score": 0.15, "division": "IBM Quantum"},
    "HON": {"name": "Honeywell", "technology": "Trapped Ion", "quantum_revenue_pct": 1.0,
            "max_score": 0.3, "division": "Former Quantinuum parent",
            "note": "Post-Quantinuum spinoff (June 2026). Retains minority stake. Reduced but non-zero exposure."},
}

# Fix 1: Tickers removed from active scoring due to label quality analysis findings
INACTIVE_TICKERS = {
    "MSFT": {
        "name": "Microsoft", "technology": "Topological", "division": "Azure Quantum",
        "reason": "Quantum revenue <0.1%. IC=-0.033 (p=0.53), indistinguishable from noise.",
        "score": 0.0,
        "reasoning": "Inactive: quantum revenue exposure too low for meaningful signal.",
    },
    "GOOGL": {
        "name": "Alphabet/Google", "technology": "Superconducting", "division": "Google Quantum AI",
        "reason": "Quantum revenue <0.1%. IC=-0.023 (p=0.67), indistinguishable from noise.",
        "score": 0.0,
        "reasoning": "Inactive: quantum revenue exposure too low for meaningful signal.",
    },
    "NVDA": {
        "name": "NVIDIA", "technology": "Quantum Simulation", "division": "cuQuantum",
        "reason": "Anti-predictive (IC=-0.175, p=0.0008). Moves on AI/GPU demand, not quantum news.",
        "score": 0.0,
        "reasoning": "Inactive: anti-predictive, moves on AI/GPU demand not quantum news.",
    },
}

# Combined views
ACTIVE_TICKERS = list(PURE_PLAY_TICKERS.keys()) + list(ADJACENT_TICKERS.keys())
ALL_TICKERS = {**PURE_PLAY_TICKERS, **ADJACENT_TICKERS, **INACTIVE_TICKERS}
OUTPUT_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON", "MSFT", "GOOGL", "NVDA"]

# ============================================================
# Fix 7: Liquidity Tiers (verified against Yahoo Finance, June 2026)
# ============================================================

LIQUIDITY_TIERS = {
    "IONQ": {"avg_daily_volume_usd": 180_000_000, "tier": "high"},
    "RGTI": {"avg_daily_volume_usd": 95_000_000, "tier": "high"},
    "QBTS": {"avg_daily_volume_usd": 70_000_000, "tier": "medium"},
    "QUBT": {"avg_daily_volume_usd": 45_000_000, "tier": "medium"},
    "QNT": {"avg_daily_volume_usd": 150_000_000, "tier": "high",
            "note": "Estimated from IPO week. Will stabilize."},
    "IBM": {"avg_daily_volume_usd": 800_000_000, "tier": "very_high"},
    "HON": {"avg_daily_volume_usd": 600_000_000, "tier": "very_high"},
}

# ============================================================
# Quantum Computing ETFs
# ============================================================

ETFS = {
    "QTUM": "Defiance Quantum ETF",  # Fix 6: Used as sector benchmark
    "WQTM": "WisdomTree Quantum Computing Fund",
    "QNTM": "VanEck Quantum ETF",
}

# ============================================================
# Event Categories for Classification
# ============================================================

EVENT_TYPES = [
    "physical_qubit_milestone",
    "logical_qubit_breakthrough",
    "error_correction_advance",
    "quantum_volume_increase",
    "government_funding",
    "commercial_partnership",
    "revenue_earnings",
    "executive_change",
    "patent_grant",
    "academic_publication",
    "product_launch",
    "competitive_development",
    "regulatory_filing",
    "analyst_rating_change",
    "sector_macro_event",
]

# Sentiment Labels
SENTIMENT_LABELS = ["strongly_bearish", "bearish", "neutral", "bullish", "strongly_bullish"]

# ============================================================
# Data Source Configuration
# ============================================================

ARXIV_CATEGORIES = ["quant-ph", "cs.ET"]
ARXIV_KEYWORDS = [
    "quantum computing",
    "quantum error correction",
    "logical qubit",
    "fault tolerant quantum",
    "quantum advantage",
    "quantum supremacy",
    "superconducting qubit",
    "trapped ion",
    "quantum annealing",
    "quantum volume",
    "quantum processor",
    "quantum algorithm",
    "quantum machine learning",
    "quantum cryptography",
]

# SEC Filing Types to Monitor
SEC_FILING_TYPES = ["10-K", "10-Q", "8-K", "S-1", "DEF 14A"]

# ============================================================
# Model Configuration
# ============================================================

MODEL_CONFIG = {
    "base_model": "Qwen/Qwen3-8B-Instruct",
    "max_seq_length": 4096,
    "lora_rank": 64,
    "lora_alpha": 16,
    "learning_rate": 5e-5,
    "num_epochs": 4,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
}

# Modal Configuration
MODAL_CONFIG = {
    "workspace_id": "ac-PGYLNihy2INHkVQupXFTUV",
    "app_name": "alpha-signal-finetune",
    "gpu": "A100",
}

# Qwen Cloud (DashScope) Configuration
QWEN_CLOUD_CONFIG = {
    "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
    "teacher_model": "qwen3.7-max",
    "temperature": 0.3,
    "max_tokens": 1024,
}

# ============================================================
# Signal Thresholds
# ============================================================

SIGNAL_CONFIG = {
    "high_urgency_threshold": 0.8,
    "medium_urgency_threshold": 0.5,
    "minimum_confidence": 0.6,
}

# Score ranges per ticker type (for validation)
SCORE_RANGES = {
    "pure_play": (-2.0, 2.0),      # IONQ, RGTI, QBTS, QUBT, QNT
    "adjacent_ibm": (-0.15, 0.15),  # IBM
    "adjacent_hon": (-0.3, 0.3),    # HON
    "inactive": (0.0, 0.0),         # MSFT, GOOGL, NVDA
}
