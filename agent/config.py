import os

# DashScope (Qwen Cloud)
DASHSCOPE_API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-ws-H.IIMPYP.OVEd.MEYCIQCgnJiyfu3TI7aOMuMio4dSrWTf5zbFNrCpKP-NTyUGagIhAJQ6AGEG4uC8C9LmDEqJCLQGSUnilOLV6lQ1QR7QvVBi")
DASHSCOPE_BASE_URL = os.environ.get("DASHSCOPE_BASE_URL", "https://ws-wuyspztgv1cyxvbr.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1")
REASONING_MODEL = "qwen3-max"  # For memory-augmented reasoning
EMBEDDING_MODEL = "text-embedding-v3"  # For memory retrieval (if available)

# Modal (fine-tuned model)
MODAL_ENDPOINT = os.environ.get("MODAL_ENDPOINT", "")  # Set when Modal endpoint is deployed
INFERENCE_BACKEND = os.environ.get("INFERENCE_BACKEND", "dashscope")  # "dashscope" or "modal"

# Memory
MEMORY_DB_PATH = os.environ.get("MEMORY_DB_PATH", "/opt/alpha-signal-analysis/data/memory.db")
MAX_MEMORY_CONTEXT_TOKENS = 4000  # How many tokens of memory to inject into prompt
MEMORY_DECAY_DAYS = 90  # Memories older than this get consolidated
FORGETTING_THRESHOLD = 0.3  # Relevance score below this gets pruned

# Tickers
QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]
PURE_PLAY_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT"]  # Full signal range
DIVERSIFIED_TICKERS = ["IBM", "GOOGL", "MSFT", "HON", "NVDA"]  # Capped signals
