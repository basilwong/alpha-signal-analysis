"""
Quantum Alpha Intelligence Platform — Gradio Server Mode

Uses gradio.Server (inherits from FastAPI) to serve both custom API endpoints
and the custom frontend. This avoids port binding conflicts on HF Spaces because
gradio.Server handles port management internally, just like gr.Blocks.launch().

For ZeroGPU support, we use @app.api() which integrates with the Gradio queue
and supports the @spaces.GPU decorator pattern.
"""

import json
import time
import os
from pathlib import Path
from gradio import Server
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
EVAL_DIR = DATA_DIR / "eval"
MARKET_DIR = DATA_DIR / "market"
FRONTEND_DIR = BASE_DIR / "frontend_v2"

# Quantum universe
QUANTUM_TICKERS = ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]

# Model prediction files (historical comparison)
MODEL_FILES = {
    # Fine-tuned models (Manus Teacher)
    "Nemotron-7B (SFT + GRPO, Manus Teacher)": EVAL_DIR / "predictions_v7d_grpo_clean.jsonl",
    "Nemotron-7B (Best-of-4 SFT, Manus Teacher)": EVAL_DIR / "predictions_v7b_clean.jsonl",
    "Nemotron-7B (SFT + DPO, Manus Teacher)": EVAL_DIR / "predictions_v7c_clean.jsonl",
    "Nemotron-7B (SFT + Thinking, Manus Teacher)": EVAL_DIR / "predictions_openreasoning7b_v7a.jsonl",
    "Nemotron-7B (SFT + Bearish, Manus Teacher)": EVAL_DIR / "predictions_openreasoning7b_v6.jsonl",
    "Nemotron-7B (SFT, Manus Teacher)": EVAL_DIR / "predictions_openreasoning7b_v4.jsonl",
    # Fine-tuned models (GPT-5.5 Teacher)
    "Nemotron-7B (SFT, GPT-5.5 Teacher)": EVAL_DIR / "predictions_v8_sft_fixed.jsonl",
    "Nemotron-7B (SFT + GRPO, GPT-5.5 Teacher)": EVAL_DIR / "predictions_v8_grpo.jsonl",
    # Teacher models
    "Manus (Teacher, Direct)": EVAL_DIR / "predictions_manus_teacher_v2.jsonl",
    "GPT-5.5 (Teacher, Direct)": EVAL_DIR / "predictions_codex_teacher.jsonl",
    # Base models
    "Nemotron-7B (Base, No Fine-Tuning)": EVAL_DIR / "predictions_base_7b_fixed.jsonl",
    "Nemotron-14B (Base, No Fine-Tuning)": EVAL_DIR / "predictions_base_14b_fixed.jsonl",
    "Nemotron-32B (Base, No Fine-Tuning)": EVAL_DIR / "predictions_base_32b_fixed.jsonl",
}

# Models available for live inference (only fine-tuned models)
LIVE_MODELS = {
    "Nemotron-7B (SFT + GRPO, Manus Teacher)": "basilwong/quantum-alpha-openreasoning-7b-grpo",
    "Nemotron-7B (SFT, Manus Teacher)": "build-small-hackathon/quantum-alpha-qwen3-8b",
}

MODEL_ID = "basilwong/quantum-alpha-openreasoning-7b-grpo"


def load_predictions(path):
    """Load predictions from a JSONL file."""
    predictions = []
    if path.exists():
        with open(path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    if p.get("status") == "success" or p.get("success") == True:
                        predictions.append(p)
    return sorted(predictions, key=lambda x: x.get("date", ""))


def load_all_models():
    """Load predictions for all models."""
    all_preds = {}
    for name, path in MODEL_FILES.items():
        preds = load_predictions(path)
        if preds:
            all_preds[name] = preds
    return all_preds


def load_eval_results():
    """Load evaluation metrics."""
    path = EVAL_DIR / "results_multi_model.json"
    if path.exists():
        with open(path) as f:
            return json.loads(f.read())
    return {}


def load_market_data():
    """Load market data as JSON-serializable dict."""
    import pandas as pd
    import numpy as np
    prices = {}
    for ticker in QUANTUM_TICKERS + ["SPY"]:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            series = df[close_col].dropna()
            if hasattr(series, 'iloc') and len(series.shape) > 1:
                series = series.iloc[:, 0]
            values = series.values.flatten() if hasattr(series.values, 'flatten') else series.values
            prices[ticker] = {
                "dates": [d.strftime("%Y-%m-%d") for d in series.index],
                "values": [round(float(v), 2) for v in values],
            }
    return prices


# Load data at startup
print("Loading data...")
ALL_PREDICTIONS = load_all_models()
EVAL_RESULTS = load_eval_results()
MARKET_DATA = load_market_data()
print(f"Models loaded: {', '.join(f'{k} ({len(v)})' for k, v in ALL_PREDICTIONS.items())}")

# Sector data
SECTOR_DATA = {
    "tickers": {
        "IONQ": {"name": "IonQ", "tech": "Trapped Ion", "signal_weight": 1.0, "cluster": "Trapped Ion"},
        "RGTI": {"name": "Rigetti", "tech": "Superconducting", "signal_weight": 1.0, "cluster": "Superconducting"},
        "QBTS": {"name": "D-Wave", "tech": "Annealing", "signal_weight": 1.0, "cluster": "Annealing"},
        "QUBT": {"name": "QCi", "tech": "Neutral Atom", "signal_weight": 1.0, "cluster": "Neutral Atom"},
        "QNT": {"name": "Quantinuum", "tech": "Trapped Ion", "signal_weight": 1.0, "cluster": "Trapped Ion"},
        "IBM": {"name": "IBM", "tech": "Superconducting", "signal_weight": 0.15, "cluster": "Superconducting"},
        "GOOGL": {"name": "Google", "tech": "Superconducting", "signal_weight": 0.0, "cluster": "Superconducting"},
        "MSFT": {"name": "Microsoft", "tech": "Topological", "signal_weight": 0.0, "cluster": "Topological"},
        "HON": {"name": "Honeywell", "tech": "Trapped Ion", "signal_weight": 0.30, "cluster": "Trapped Ion"},
        "NVDA": {"name": "NVIDIA", "tech": "Adjacent", "signal_weight": 0.0, "cluster": "Adjacent"},
    },
    "clusters": {
        "Trapped Ion": ["IONQ", "QNT", "HON"],
        "Superconducting": ["RGTI", "IBM", "GOOGL"],
        "Annealing": ["QBTS"],
        "Topological": ["MSFT"],
        "Neutral Atom": ["QUBT"],
        "Adjacent": ["NVDA"],
    },
    "dynamics": [
        {"trigger": "Trapped-ion breakthrough", "bullish": ["IONQ", "HON"], "bearish": ["RGTI", "IBM", "GOOGL"]},
        {"trigger": "Superconducting breakthrough", "bullish": ["RGTI", "IBM", "GOOGL"], "bearish": ["IONQ", "HON"]},
        {"trigger": "Error correction advance", "bullish": ["IONQ", "RGTI", "HON", "IBM", "GOOGL", "MSFT"], "bearish": []},
        {"trigger": "Government funding", "bullish": ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "HON"], "bearish": []},
    ],
}


# ============================================================
# GRADIO SERVER APP
# ============================================================

app = Server(title="Quantum Alpha Intelligence API")


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/api/models")
async def get_models():
    """List all available models with prediction counts."""
    models = []
    for name, preds in ALL_PREDICTIONS.items():
        models.append({
            "name": name,
            "predictions": len(preds),
            "live_inference": name in LIVE_MODELS,
        })
    return JSONResponse({"models": models})


@app.get("/api/events")
async def get_events(model: str = "Nemotron-7B (SFT, Manus Teacher)"):
    """List all events for a given model."""
    preds = ALL_PREDICTIONS.get(model, [])
    events = []
    for i, p in enumerate(preds):
        events.append({
            "idx": i,
            "article_idx": p.get("article_idx"),
            "date": p.get("date", ""),
            "title": p.get("title", "Untitled"),
            "source": p.get("source", "news"),
        })
    return JSONResponse({"model": model, "events": events})


@app.get("/api/prediction")
async def get_prediction(model: str, idx: int):
    """Get a specific prediction by model and index."""
    preds = ALL_PREDICTIONS.get(model, [])
    if idx < 0 or idx >= len(preds):
        return JSONResponse({"error": "Index out of range"}, status_code=404)

    pred = preds[idx]
    signal = pred.get("signal", {})

    # Get price data for the event date
    event_date = pred.get("date", "")
    price_data = {}
    benchmark_data = {}  # SPY as market benchmark
    if event_date and MARKET_DATA:
        # Get SPY benchmark data
        if "SPY" in MARKET_DATA:
            spy_dates = MARKET_DATA["SPY"]["dates"]
            spy_values = MARKET_DATA["SPY"]["values"]
            try:
                spy_start = next(i for i, d in enumerate(spy_dates) if d >= event_date)
                spy_end = min(spy_start + 21, len(spy_dates))
                benchmark_data["SPY"] = {
                    "dates": spy_dates[spy_start:spy_end],
                    "values": spy_values[spy_start:spy_end],
                }
            except StopIteration:
                pass

        # Get quantum ticker data
        for ticker in QUANTUM_TICKERS:
            if ticker in MARKET_DATA:
                dates = MARKET_DATA[ticker]["dates"]
                values = MARKET_DATA[ticker]["values"]
                try:
                    start_idx = next(i for i, d in enumerate(dates) if d >= event_date)
                    end_idx = min(start_idx + 21, len(dates))
                    price_data[ticker] = {
                        "dates": dates[start_idx:end_idx],
                        "values": values[start_idx:end_idx],
                    }
                except StopIteration:
                    pass

    return JSONResponse({
        "model": model,
        "idx": idx,
        "prediction": {
            "article_idx": pred.get("article_idx"),
            "date": pred.get("date"),
            "title": pred.get("title"),
            "source": pred.get("source"),
            "signal": signal,
            "time_seconds": pred.get("time_seconds") or pred.get("time_ms", 0) / 1000,
        },
        "price_data": price_data,
        "benchmark_data": benchmark_data,
    })


@app.get("/api/prediction_comparison")
async def get_prediction_comparison(article_idx: int):
    """Get predictions from ALL models for a specific article (by article_idx)."""
    results = {}
    for model_name, preds in ALL_PREDICTIONS.items():
        for p in preds:
            if p.get("article_idx") == article_idx:
                results[model_name] = {
                    "signal": p.get("signal", {}),
                    "time_seconds": p.get("time_seconds") or p.get("time_ms", 0) / 1000,
                }
                break
    return JSONResponse({"article_idx": article_idx, "models": results})


@app.get("/api/eval_metrics")
async def get_eval_metrics():
    """Get evaluation metrics for all models."""
    return JSONResponse(EVAL_RESULTS)


@app.get("/api/sector_data")
async def get_sector_data():
    """Get sector map data."""
    return JSONResponse(SECTOR_DATA)


@app.get("/api/market_data")
async def get_market_data(ticker: str, start: str = "", end: str = ""):
    """Get market data for a specific ticker."""
    if ticker not in MARKET_DATA:
        return JSONResponse({"error": f"Ticker {ticker} not found"}, status_code=404)
    data = MARKET_DATA[ticker]
    if start:
        dates = data["dates"]
        values = data["values"]
        filtered = [(d, v) for d, v in zip(dates, values) if d >= start and (not end or d <= end)]
        if filtered:
            dates, values = zip(*filtered)
            return JSONResponse({"ticker": ticker, "dates": list(dates), "values": list(values)})
    return JSONResponse({"ticker": ticker, **data})


# ============================================================
# LIVE INFERENCE (via Gradio API for ZeroGPU support)
# ============================================================

SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, produce a signal vector scoring ALL 9 tickers simultaneously.

Tickers: IONQ, RGTI, QBTS, QUBT, IBM, GOOGL, MSFT, HON, NVDA
Score range: -2.0 to +2.0 (scaled by signal weight for diversified companies)
Output ONLY valid JSON matching the signal vector schema."""


def _do_inference(text: str, source: str, model_name: str, enable_thinking: bool) -> str:
    """Run inference using the fine-tuned model. Called within GPU context."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = LIVE_MODELS.get(model_name, MODEL_ID)

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Analyze: {text}"},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=True, enable_thinking=enable_thinking
    ).to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=10000, temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated = outputs[0][inputs["input_ids"].shape[-1]:]
    raw = tokenizer.decode(generated, skip_special_tokens=True)
    latency = time.time() - start

    # Parse
    thinking = ""
    content = raw
    if "<think>" in raw:
        parts = raw.split("</think>")
        if len(parts) > 1:
            thinking = parts[0].replace("<think>", "").strip()
            content = parts[-1].strip()

    try:
        s = content.find("{")
        e = content.rfind("}") + 1
        signal = json.loads(content[s:e]) if s != -1 else json.loads(content)
    except Exception:
        # Try to salvage partial JSON by finding the outermost complete object
        try:
            # Sometimes the model outputs a wrapper like {"signal_vector": ...}
            # that gets truncated. Try to parse what we have.
            partial = content[s:e] if s != -1 else content
            # Add closing braces if truncated
            open_braces = partial.count("{") - partial.count("}")
            if open_braces > 0:
                partial += "}" * open_braces
            signal = json.loads(partial)
        except Exception:
            signal = {"error": "Failed to parse JSON (output may have been truncated)", "raw": content[:500]}

    return json.dumps({
        "signal": signal,
        "thinking": thinking,
        "latency_ms": int(latency * 1000),
        "model": model_name,
    })


# Wrap with spaces.GPU for ZeroGPU support
try:
    import spaces

    @spaces.GPU
    def gpu_inference(text: str, source: str, model_name: str, enable_thinking: bool) -> str:
        return _do_inference(text, source, model_name, enable_thinking)
except ImportError:
    def gpu_inference(text: str, source: str, model_name: str, enable_thinking: bool) -> str:
        return _do_inference(text, source, model_name, enable_thinking)


# Register the inference function as a Gradio API endpoint for ZeroGPU queue support
@app.api(name="analyze")
def analyze_via_gradio(text: str, source: str, model_name: str, enable_thinking: bool) -> str:
    """Run live inference via Gradio API (supports ZeroGPU)."""
    return gpu_inference(text, source, model_name, enable_thinking)


# Also expose as a standard FastAPI POST endpoint for the frontend
from fastapi import Request


@app.post("/api/analyze")
async def analyze(request: Request):
    """Run live inference on a new article."""
    body = await request.json()
    text = body.get("text", "")
    source = body.get("source", "news")
    model_name = body.get("model", "Nemotron-7B (SFT + GRPO, Manus Teacher)")
    enable_thinking = body.get("enable_thinking", False)

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    result = gpu_inference(text, source, model_name, enable_thinking)
    return JSONResponse(json.loads(result))


# ============================================================
# SERVE FRONTEND
# ============================================================

@app.get("/")
async def serve_index():
    """Serve the custom frontend."""
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse({"error": "Frontend not found. Place index.html in frontend_v2/"})


# Mount static files (CSS, JS)
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, ssr_mode=False)
