"""
FastAPI backend for the Memory Agent.
Deployed on Alibaba Cloud ECS (free tier).
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json
import time
from datetime import datetime
from typing import List, Optional
from pathlib import Path

from .memory import MemoryStore
from .retrieval import MemoryRetriever
from .forgetting import ForgettingEngine
from .inference import generate_signal
from .config import MEMORY_DB_PATH

app = FastAPI(title="Alpha Signal Analysis - Memory Agent")

# Serve frontend static files
FRONTEND_DIR = Path(__file__).parent.parent / "frontend_v2"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize memory
memory = MemoryStore(MEMORY_DB_PATH)
retriever = MemoryRetriever(memory)
forgetting = ForgettingEngine(memory)


class AnalyzeRequest(BaseModel):
    text: str
    source: str = "news"
    enable_thinking: bool = True


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class SeedFact(BaseModel):
    ticker: str = ""
    type: str = "general"
    content: str = ""


@app.get("/api/health")
async def health():
    return {"status": "running", "memory_stats": memory.get_memory_stats()}


@app.get("/api/memory/stats")
async def memory_stats():
    return memory.get_memory_stats()


@app.get("/api/memory/knowledge")
async def get_knowledge(ticker: str = None, limit: int = 20):
    rows = memory.retrieve_knowledge(ticker=ticker, limit=limit)
    return {"knowledge": [{"id": r[0], "ticker": r[1], "fact_type": r[2], "content": r[3], "source": r[4], "confidence": r[5], "created_at": r[6]} for r in rows]}


@app.get("/api/memory/signals")
async def get_signals(ticker: str = None, limit: int = 20):
    rows = memory.retrieve_signal_history(ticker=ticker, limit=limit)
    return {"signals": [{"id": r[0], "date": r[1], "title": r[2], "source": r[3], "signal_vector": json.loads(r[4]) if r[4] else {}, "model": r[9]} for r in rows]}


@app.post("/api/analyze")
async def analyze(req: AnalyzeRequest):
    """Analyze an article with memory-augmented reasoning."""
    start = time.time()

    # 1. Retrieve relevant memories
    memory_context = retriever.retrieve_context(req.text)

    # 2. Generate signal with memory context
    result = generate_signal(req.text, req.source, memory_context, req.enable_thinking)

    # 3. Parse the signal
    content = result["content"]
    try:
        s = content.find("{")
        e = content.rfind("}") + 1
        signal = json.loads(content[s:e]) if s != -1 else {}
    except:
        signal = {"error": "Failed to parse", "raw": content[:500]}

    # 4. Store in memory — handle multiple LLM response formats
    signal_vector = signal.get("signal_vector", signal.get("signals", {}))
    if not signal_vector:
        # LLM may return tickers at top level: {"IONQ": 1.8, "RGTI": -0.4, ...}
        from .config import QUANTUM_TICKERS
        flat_signals = {k: v for k, v in signal.items() if k in QUANTUM_TICKERS and isinstance(v, (int, float))}
        if flat_signals:
            signal_vector = flat_signals
    if signal_vector:
        memory.store_signal(
            article_date=datetime.utcnow().strftime("%Y-%m-%d"),
            article_title=req.text[:100],
            article_source=req.source,
            signal_vector=signal_vector,
            reasoning=signal.get("chain_of_thought", ""),
            model_used="qwen3-max"
        )

        # 5. Extract and store new knowledge
        cot = signal.get("chain_of_thought", "")
        if cot and len(cot) > 50:
            # Store key facts mentioned in reasoning
            for ticker, data in signal_vector.items():
                if isinstance(data, dict):
                    score = abs(data.get("score", 0))
                    reasoning_text = data.get("reasoning", "")[:200]
                else:
                    score = abs(float(data)) if data is not None else 0
                    reasoning_text = ""
                if score > 0.3:
                    memory.store_knowledge(
                        ticker=ticker,
                        fact_type="signal_context",
                        content=f"On {datetime.utcnow().strftime('%Y-%m-%d')}: score={data} {reasoning_text}",
                        source=req.source
                    )

    latency = int((time.time() - start) * 1000)

    return {
        "signal": signal,
        "thinking": result["thinking"],
        "memory_context_used": memory_context[:500],
        "latency_ms": latency,
        "memory_stats": memory.get_memory_stats()
    }


@app.post("/api/memory/forget")
async def trigger_forgetting():
    """Manually trigger a forgetting cycle."""
    result = forgetting.run_forgetting_cycle()
    return {"forgetting_result": result, "memory_stats": memory.get_memory_stats()}


@app.get("/")
async def serve_index():
    """Serve the custom frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Alpha Signal Analysis Memory Agent API. Frontend not found."}


@app.post("/api/memory/seed")
async def seed_knowledge(facts: List[SeedFact]):
    """Seed the memory with initial sector knowledge."""
    for fact in facts:
        memory.store_knowledge(
            ticker=fact.ticker,
            fact_type=fact.type,
            content=fact.content,
            source="seed",
            confidence=0.9
        )
    return {"seeded": len(facts)}
