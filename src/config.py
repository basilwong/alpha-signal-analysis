"""
Configuration for the Quantum Alpha Intelligence Platform.
"""

# Quantum Computing Ticker Universe
PURE_PLAY_TICKERS = {
    "IONQ": {"name": "IonQ", "technology": "Trapped Ion"},
    "RGTI": {"name": "Rigetti Computing", "technology": "Superconducting"},
    "QBTS": {"name": "D-Wave Quantum", "technology": "Quantum Annealing"},
    "QUBT": {"name": "Quantum Computing Inc", "technology": "Photonic"},
    "INFQ": {"name": "Infleqtion", "technology": "Neutral Atom"},
}

ADJACENT_TICKERS = {
    "IBM": {"name": "IBM", "technology": "Superconducting", "division": "IBM Quantum"},
    "GOOGL": {"name": "Alphabet/Google", "technology": "Superconducting", "division": "Google Quantum AI"},
    "MSFT": {"name": "Microsoft", "technology": "Topological", "division": "Azure Quantum"},
    "HON": {"name": "Honeywell/Quantinuum", "technology": "Trapped Ion", "division": "Quantinuum"},
    "NVDA": {"name": "NVIDIA", "technology": "Quantum Simulation", "division": "cuQuantum"},
}

ALL_TICKERS = {**PURE_PLAY_TICKERS, **ADJACENT_TICKERS}

# Quantum Computing ETFs
ETFS = {
    "QTUM": "Defiance Quantum ETF",
    "WQTM": "WisdomTree Quantum Computing Fund",
    "QNTM": "VanEck Quantum ETF",
}

# Event Categories for Classification
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
]

# Sentiment Labels
SENTIMENT_LABELS = ["strongly_bearish", "bearish", "neutral", "bullish", "strongly_bullish"]

# Data Source Configuration
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

# Model Configuration
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

# Signal Thresholds
SIGNAL_CONFIG = {
    "high_urgency_threshold": 0.8,
    "medium_urgency_threshold": 0.5,
    "minimum_confidence": 0.6,
}
