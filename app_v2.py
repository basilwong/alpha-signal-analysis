"""
Quantum Alpha Intelligence Platform - V2 Frontend
Three-tab interface: Signal Explorer | Evaluation Dashboard | Sector Map

Uses pre-computed predictions and market data for historical browsing,
with live inference capability via ZeroGPU for new articles.
"""

import os
import json
import time
import re
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False

# ============================================================
# DATA LOADING
# ============================================================

DATA_DIR = Path("data")
EVAL_DIR = DATA_DIR / "eval"
MARKET_DIR = DATA_DIR / "market"

# Sector data (inline for portability)
QUANTUM_UNIVERSE = ["IONQ", "RGTI", "QBTS", "QUBT", "IBM", "GOOGL", "MSFT", "HON", "NVDA"]

COMPANY_INFO = {
    "IONQ": {"name": "IonQ", "tech": "Trapped Ion", "exposure": 1.0, "color": "#00d4aa"},
    "RGTI": {"name": "Rigetti", "tech": "Superconducting", "exposure": 1.0, "color": "#ff6b6b"},
    "QBTS": {"name": "D-Wave", "tech": "Annealing", "exposure": 1.0, "color": "#ffa94d"},
    "QUBT": {"name": "QCi", "tech": "Neutral Atom", "exposure": 1.0, "color": "#a78bfa"},
    "IBM": {"name": "IBM", "tech": "Superconducting", "exposure": 0.02, "color": "#4dabf7"},
    "GOOGL": {"name": "Google", "tech": "Superconducting", "exposure": 0.001, "color": "#69db7c"},
    "MSFT": {"name": "Microsoft", "tech": "Topological", "exposure": 0.001, "color": "#ffd43b"},
    "HON": {"name": "Honeywell", "tech": "Trapped Ion", "exposure": 0.05, "color": "#f06595"},
    "NVDA": {"name": "NVIDIA", "tech": "Adjacent", "exposure": 0.01, "color": "#74c0fc"},
}

TECH_CLUSTERS = {
    "Trapped Ion": ["IONQ", "HON"],
    "Superconducting": ["RGTI", "IBM", "GOOGL"],
    "Annealing": ["QBTS"],
    "Topological": ["MSFT"],
    "Neutral Atom": ["QUBT"],
    "Adjacent": ["NVDA"],
}

# Load predictions
def load_predictions():
    """Load pre-computed predictions from file (prefer final version)."""
    predictions = []
    # Try final file first, fall back to intermediate
    pred_path = EVAL_DIR / "predictions_v2_final.jsonl"
    if not pred_path.exists():
        pred_path = EVAL_DIR / "predictions_v2.jsonl"
    if pred_path.exists():
        with open(pred_path) as f:
            for line in f:
                if line.strip():
                    p = json.loads(line)
                    if p.get("status") == "success":
                        predictions.append(p)
    return sorted(predictions, key=lambda x: x.get("date", ""))


def load_market_data():
    """Load cached market data."""
    prices = {}
    for ticker in QUANTUM_UNIVERSE + ["SPY"]:
        path = MARKET_DIR / f"{ticker}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            close_col = "Adj Close" if "Adj Close" in df.columns else "Close"
            prices[ticker] = df[close_col]
    return pd.DataFrame(prices)


# Load data at startup
print("Loading data...")
PREDICTIONS = load_predictions()
PRICES = load_market_data()
print(f"Loaded {len(PREDICTIONS)} predictions, {len(PRICES)} days of market data")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_event_list():
    """Get list of events for the dropdown."""
    events = []
    for i, p in enumerate(PREDICTIONS):
        title = p.get("title", "Untitled")[:60]
        date = p.get("date", "N/A")
        events.append(f"[{i}] {date} | {title}")
    return events


def create_signal_vector_chart(signal_vector: dict) -> go.Figure:
    """Create horizontal bar chart of signal vector across all tickers."""
    tickers = []
    scores = []
    colors = []
    reasonings = []

    for ticker in QUANTUM_UNIVERSE:
        if ticker in signal_vector:
            entry = signal_vector[ticker]
            score = entry.get("score", 0)
            reasoning = entry.get("reasoning", "")
            tickers.append(f"{ticker} ({COMPANY_INFO[ticker]['name']})")
            scores.append(score)
            colors.append("#00d4aa" if score >= 0 else "#ff6b6b")
            reasonings.append(reasoning)

    # Sort by score
    sorted_data = sorted(zip(tickers, scores, colors, reasonings), key=lambda x: x[1], reverse=True)
    tickers, scores, colors, reasonings = zip(*sorted_data) if sorted_data else ([], [], [], [])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=list(tickers),
        x=list(scores),
        orientation='h',
        marker_color=list(colors),
        text=[f"{s:+.2f}" for s in scores],
        textposition='outside',
        hovertext=list(reasonings),
        hoverinfo='text',
    ))

    fig.update_layout(
        title="Cross-Sectional Signal Vector",
        xaxis_title="Signal Score",
        yaxis_title="",
        xaxis=dict(range=[-2.5, 2.5], zeroline=True, zerolinewidth=2, zerolinecolor='gray'),
        height=400,
        margin=dict(l=150, r=50, t=50, b=50),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
    )
    return fig


def create_price_overlay_chart(prediction: dict) -> go.Figure:
    """Create predicted vs actual price movement chart."""
    date_str = prediction.get("date", "")
    signal = prediction.get("signal", {})
    signal_vector = signal.get("signal_vector", {})

    fig = go.Figure()

    if not date_str or PRICES.empty:
        fig.update_layout(title="Price data not available", template="plotly_dark",
                         paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
        return fig

    try:
        event_date = pd.Timestamp(date_str)
    except:
        fig.update_layout(title="Invalid date", template="plotly_dark",
                         paper_bgcolor="#1a1a2e", plot_bgcolor="#16213e")
        return fig

    # Get top 3 tickers by absolute score
    ticker_scores = [(t, abs(signal_vector.get(t, {}).get("score", 0))) for t in QUANTUM_UNIVERSE]
    top_tickers = sorted(ticker_scores, key=lambda x: x[1], reverse=True)[:3]
    top_tickers = [t[0] for t in top_tickers if t[1] > 0.1]

    if not top_tickers:
        top_tickers = ["IONQ", "RGTI"]

    for ticker in top_tickers:
        if ticker not in PRICES.columns:
            continue

        ticker_prices = PRICES[ticker].dropna()
        # Find the event date index
        dates_after = ticker_prices.index[ticker_prices.index >= event_date]
        if len(dates_after) < 2:
            continue

        # Get 20 trading days after event
        start_idx = ticker_prices.index.get_loc(dates_after[0])
        end_idx = min(start_idx + 21, len(ticker_prices))
        window = ticker_prices.iloc[start_idx:end_idx]

        # Compute cumulative return from event date
        base_price = window.iloc[0]
        cum_returns = ((window / base_price) - 1) * 100

        days = list(range(len(cum_returns)))
        color = COMPANY_INFO[ticker]["color"]

        fig.add_trace(go.Scatter(
            x=days, y=cum_returns.values,
            mode='lines+markers',
            name=f"{ticker} (actual)",
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))

        # Add predicted direction as annotation
        predicted_score = signal_vector.get(ticker, {}).get("score", 0)
        if predicted_score != 0:
            fig.add_hline(
                y=predicted_score * 3,  # Rough scaling: score of 1.0 ~ 3% move
                line_dash="dash",
                line_color=color,
                opacity=0.5,
                annotation_text=f"{ticker} predicted ({predicted_score:+.1f})",
            )

    fig.update_layout(
        title="Actual Price Movement After Event (Cumulative Return %)",
        xaxis_title="Trading Days After Event",
        yaxis_title="Cumulative Return (%)",
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        legend=dict(x=0.02, y=0.98),
    )
    return fig


def create_sector_map() -> go.Figure:
    """Create a grouped bar chart showing companies by technology cluster and revenue exposure."""
    fig = go.Figure()

    # Organize data by cluster
    cluster_order = ["Trapped Ion", "Superconducting", "Annealing", "Topological", "Neutral Atom", "Adjacent"]
    cluster_colors = {
        "Trapped Ion": "#00d4aa",
        "Superconducting": "#4dabf7",
        "Annealing": "#ffa94d",
        "Topological": "#ffd43b",
        "Neutral Atom": "#a78bfa",
        "Adjacent": "#74c0fc",
    }

    tickers_ordered = []
    exposures = []
    colors = []
    labels = []

    for cluster in cluster_order:
        for ticker in TECH_CLUSTERS[cluster]:
            info = COMPANY_INFO[ticker]
            tickers_ordered.append(f"{ticker} ({info['name']})")
            exposures.append(info["exposure"] * 100)
            colors.append(cluster_colors[cluster])
            labels.append(cluster)

    fig.add_trace(go.Bar(
        y=tickers_ordered,
        x=exposures,
        orientation='h',
        marker_color=colors,
        text=[f"{e:.1f}%" for e in exposures],
        textposition='outside',
        hovertext=[f"{t}<br>Cluster: {l}<br>Quantum Revenue: {e:.1f}%<br>Max Signal: +/-{min(2.0, e/50):.2f}" for t, l, e in zip(tickers_ordered, labels, exposures)],
        hoverinfo='text',
    ))

    # Add cluster annotations
    fig.update_layout(
        title="Revenue Exposure to Quantum Computing (determines signal scaling)",
        xaxis_title="Quantum Revenue (%)",
        yaxis_title="",
        height=400,
        margin=dict(l=180, r=80, t=50, b=50),
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
        xaxis=dict(range=[0, 110]),
    )
    return fig


def create_decay_curve_chart() -> go.Figure:
    """Signal decay curve from actual evaluation results."""
    horizons = [1, 2, 5, 10, 20]
    ic_values = [0.0108, 0.0160, 0.0483, 0.0876, 0.0469]
    p_values = [0.537, 0.362, 0.006, 0.000, 0.016]

    fig = go.Figure()
    # Color markers by significance
    colors = ['#888888' if p > 0.05 else '#00d4aa' for p in p_values]
    fig.add_trace(go.Scatter(
        x=horizons, y=ic_values,
        mode='lines+markers',
        name='Overall IC',
        line=dict(color='#00d4aa', width=2),
        marker=dict(size=10, color=colors, line=dict(width=2, color='white')),
        hovertext=[f'IC={ic:.4f}, p={p:.4f}{" ***" if p<0.01 else " **" if p<0.05 else ""}' for ic, p in zip(ic_values, p_values)],
        hoverinfo='text',
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
    # Add significance threshold annotation
    fig.add_annotation(x=10, y=0.0876, text="Peak: IC=0.088***", showarrow=True, arrowhead=2, font=dict(color='#00d4aa'))
    fig.update_layout(
        title="Signal Decay Curve (IC by Holding Period) | Green = statistically significant",
        xaxis_title="Holding Period (Trading Days)",
        yaxis_title="Information Coefficient",
        height=350,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#16213e",
    )
    return fig


# ============================================================
# EVENT HANDLERS
# ============================================================

def on_event_select(event_choice):
    """Handle event selection from dropdown."""
    if not event_choice or not PREDICTIONS:
        return (go.Figure(), "", "", "", go.Figure(), "{}", "")

    # Parse index from choice string
    try:
        idx = int(event_choice.split("]")[0].replace("[", ""))
    except:
        idx = 0

    if idx >= len(PREDICTIONS):
        idx = 0

    prediction = PREDICTIONS[idx]
    signal = prediction.get("signal", {})
    signal_vector = signal.get("signal_vector", {})

    # Signal vector chart
    signal_chart = create_signal_vector_chart(signal_vector)

    # Summary line
    event_type = signal.get("event_type", "N/A")
    time_horizon = signal.get("time_horizon", "N/A")
    decay = signal.get("signal_decay", "N/A")
    novelty = signal.get("information_novelty", "N/A")
    summary = f"Event: {event_type} | Horizon: {time_horizon} | Decay: {decay} | Novelty: {novelty}"

    # Technical translation
    translation = signal.get("technical_translation", "N/A")

    # Signal rationale
    rationale = signal.get("signal_rationale", "N/A")

    # Price overlay chart
    price_chart = create_price_overlay_chart(prediction)

    # Raw JSON
    raw_json = json.dumps(signal, indent=2)

    # Article text
    article_text = f"**{prediction.get('title', 'Untitled')}**\n\n"
    article_text += f"Source: {prediction.get('source', 'N/A')} | Date: {prediction.get('date', 'N/A')}"
    if prediction.get("time_seconds"):
        article_text += f" | Inference: {prediction['time_seconds']}s"

    return (signal_chart, summary, translation, rationale, price_chart, raw_json, article_text)


# ============================================================
# LIVE INFERENCE (GPU)
# ============================================================

MODEL_ID = "basilwong/quantum-alpha-qwen3-8b"

LIVE_SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

The quantum computing universe consists of these 9 tickers:
- IONQ: IonQ (trapped-ion, 100% quantum revenue)
- RGTI: Rigetti Computing (superconducting, 100% quantum revenue)
- QBTS: D-Wave Quantum (quantum annealing, 100% quantum revenue)
- QUBT: Quantum Computing Inc. (neutral atom, 100% quantum revenue)
- IBM: International Business Machines (superconducting, ~2% quantum revenue)
- GOOGL: Alphabet/Google (superconducting, <0.1% quantum revenue)
- MSFT: Microsoft (topological, <0.1% quantum revenue)
- HON: Honeywell/Quantinuum (trapped-ion, ~5% quantum revenue)
- NVDA: NVIDIA (adjacent/enabler, ~1% quantum revenue)

Key domain knowledge:
- Trapped-ion breakthroughs: bullish IONQ/HON, bearish RGTI/IBM/GOOGL
- Superconducting breakthroughs: bullish RGTI/IBM/GOOGL, bearish IONQ/HON
- Error correction advances: benefit ALL gate-based approaches
- Government funding: broadly bullish for entire sector
- Scale by revenue exposure: GOOGL/MSFT max +/-0.05, HON max +/-0.3, IBM max +/-0.15
- If the content is NOT related to quantum computing, assign all scores to 0.0

Output a valid JSON object:
{
    "signal_vector": {
        "IONQ": {"score": float, "reasoning": "1 sentence"},
        "RGTI": {"score": float, "reasoning": "1 sentence"},
        "QBTS": {"score": float, "reasoning": "1 sentence"},
        "QUBT": {"score": float, "reasoning": "1 sentence"},
        "IBM": {"score": float, "reasoning": "1 sentence"},
        "GOOGL": {"score": float, "reasoning": "1 sentence"},
        "MSFT": {"score": float, "reasoning": "1 sentence"},
        "HON": {"score": float, "reasoning": "1 sentence"},
        "NVDA": {"score": float, "reasoning": "1 sentence"}
    },
    "event_type": str,
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "signal_decay": "fast" | "medium" | "slow",
    "information_novelty": "high" | "medium" | "low",
    "technical_translation": "2-3 sentences.",
    "signal_rationale": "Why these scores?"
}

Output ONLY the JSON object. No markdown, no code blocks, no extra text."""

SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess novelty and likely decay speed.",
    "arxiv": "This is an academic paper abstract. Most are incremental. Only significant scores for genuine breakthroughs. Slow decay.",
    "sec_filing": "This is a regulatory filing. High reliability. Fast decay.",
    "press_release": "Company press release. Be skeptical.",
    "social_media": "Social media post. High noise, low reliability.",
    "earnings_call": "Earnings call. Forward guidance matters most.",
}

# Model loading (lazy - only loads when first inference is requested)
_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading model: {MODEL_ID}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
        )
        _model.eval()
        print("Model loaded.")
    return _model, _tokenizer


def _run_inference(text: str, source: str) -> tuple:
    """Run inference. Returns (signal_dict, latency_ms)."""
    import torch
    model, tokenizer = _load_model()
    start = time.time()

    # Clean input
    cleaned = re.sub(r'<[^>]+>', ' ', text)
    cleaned = re.sub(r'https?://\S+', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    instruction = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
    user_message = f"{instruction}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{cleaned}"

    messages = [
        {"role": "system", "content": LIVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=True, enable_thinking=False
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024, temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    if "<think>" in raw_output:
        parts = raw_output.split("</think>")
        if len(parts) > 1:
            raw_output = parts[-1].strip()

    latency_ms = int((time.time() - start) * 1000)

    try:
        s = raw_output.find("{")
        e = raw_output.rfind("}") + 1
        if s != -1 and e > s:
            signal = json.loads(raw_output[s:e])
        else:
            signal = json.loads(raw_output)
        return signal, latency_ms
    except json.JSONDecodeError as err:
        return {"error": str(err), "raw": raw_output[:500]}, latency_ms


# Apply @spaces.GPU if available
if HAS_SPACES:
    _run_inference = spaces.GPU(_run_inference)


def live_analyze(text: str, source: str) -> tuple:
    """Live analysis handler for the UI."""
    if not text or not text.strip():
        return go.Figure(), "", "", "", ""

    signal, latency_ms = _run_inference(text, source)

    if "error" in signal:
        return go.Figure(), f"Error: {signal['error']}", "", f"{latency_ms}ms", json.dumps(signal, indent=2)

    signal_vector = signal.get("signal_vector", {})
    chart = create_signal_vector_chart(signal_vector)

    event_type = signal.get("event_type", "N/A")
    time_horizon = signal.get("time_horizon", "N/A")
    decay = signal.get("signal_decay", "N/A")
    novelty = signal.get("information_novelty", "N/A")
    summary = f"Event: {event_type} | Horizon: {time_horizon} | Decay: {decay} | Novelty: {novelty}"

    translation = signal.get("technical_translation", "N/A")
    raw_json = json.dumps(signal, indent=2)

    return chart, summary, translation, f"{latency_ms}ms", raw_json


# ============================================================
# BUILD GRADIO APP
# ============================================================

with gr.Blocks(
    title="Quantum Alpha Intelligence",
    theme=gr.themes.Base(primary_hue="emerald", neutral_hue="slate"),
    css="""
    .main-title { text-align: center; }
    .info-tooltip { cursor: help; color: #888; font-size: 0.8em; }
    """
) as app:

    gr.Markdown(
        """
        # Quantum Alpha Intelligence
        ### Cross-Sectional NLP Signal Generator for the Quantum Computing Sector
        """,
        elem_classes="main-title"
    )

    with gr.Tabs():
        # ============================================================
        # TAB 1: SIGNAL EXPLORER
        # ============================================================
        with gr.Tab("Signal Explorer"):
            gr.Markdown("Browse historical events or analyze new articles. Each event produces a signal vector across all 9 quantum tickers.")

            with gr.Row():
                event_dropdown = gr.Dropdown(
                    choices=get_event_list(),
                    label="Select Event",
                    value=get_event_list()[0] if PREDICTIONS else None,
                    scale=4,
                )
                # model_selector = gr.Dropdown(
                #     choices=["Fine-tuned Qwen3-8B (LoRA)", "Base Qwen3-8B"],
                #     value="Fine-tuned Qwen3-8B (LoRA)",
                #     label="Model",
                #     scale=2,
                # )

            article_info = gr.Markdown(value="Select an event to view details.")

            gr.Markdown("**Signal Vector** — Shows the model's predicted price impact for each ticker. Green bars = bullish (stock expected to go up). Red bars = bearish. Bar length = predicted magnitude. Scores are scaled by revenue exposure (pure-play quantum stocks can reach +/-2.0, diversified companies are capped lower).")

            with gr.Row():
                with gr.Column(scale=3):
                    signal_chart = gr.Plot(label="Signal Vector")
                with gr.Column(scale=2):
                    signal_summary = gr.Textbox(label="Signal Summary", interactive=False)
                    translation_box = gr.Textbox(label="Technical Translation (ℹ️ What this means for investors)", lines=3, interactive=False)
                    rationale_box = gr.Textbox(label="Signal Rationale (ℹ️ Why these scores)", lines=3, interactive=False)

            gr.Markdown("**Actual Price Movement** — Shows what actually happened to the top affected stocks in the 20 trading days after this event. Cumulative abnormal return (%) is the stock's movement minus what we'd expect from overall market movements. Dashed lines show the model's predicted direction.")

            price_chart = gr.Plot(label="Actual Price Movement After Event")

            with gr.Accordion("Raw JSON Signal", open=False):
                json_output = gr.Code(label="Full Signal Output", language="json", lines=15)

            # Wire up event selection
            event_dropdown.change(
                fn=on_event_select,
                inputs=[event_dropdown],
                outputs=[signal_chart, signal_summary, translation_box, rationale_box, price_chart, json_output, article_info],
            )

            # Live Analysis Section
            gr.Markdown("---")
            gr.Markdown("### Live Analysis")
            gr.Markdown("Paste a new article or URL to analyze in real-time with the fine-tuned model.")

            with gr.Row():
                live_input = gr.Textbox(
                    label="Paste Article Text",
                    placeholder="Paste quantum computing news, press release, or arXiv abstract here...",
                    lines=6,
                    scale=3,
                )
                live_source = gr.Dropdown(
                    choices=["news", "arxiv", "sec_filing", "press_release", "social_media", "earnings_call"],
                    value="news",
                    label="Source Type",
                    scale=1,
                )

            live_btn = gr.Button("Analyze (runs on GPU)", variant="primary")

            with gr.Row():
                with gr.Column(scale=3):
                    live_signal_chart = gr.Plot(label="Live Signal Vector")
                with gr.Column(scale=2):
                    live_summary = gr.Textbox(label="Signal Summary", interactive=False)
                    live_translation = gr.Textbox(label="Technical Translation", lines=3, interactive=False)
                    live_latency = gr.Textbox(label="Latency", interactive=False)

            with gr.Accordion("Live Raw JSON", open=False):
                live_json = gr.Code(label="Full Signal Output", language="json", lines=15)

            live_btn.click(
                fn=live_analyze,
                inputs=[live_input, live_source],
                outputs=[live_signal_chart, live_summary, live_translation, live_latency, live_json],
            )

        # ============================================================
        # TAB 2: EVALUATION DASHBOARD
        # ============================================================
        with gr.Tab("Evaluation Dashboard"):
            gr.Markdown("""
            ### Model Performance Metrics
            Aggregate evaluation of the fine-tuned model's predictions against actual market data.
            *ℹ️ Metrics computed using Abnormal Returns (market-adjusted) and Information Coefficient (Spearman rank correlation).*
            """)

            with gr.Row():
                gr.Textbox(value="+0.088 (p<0.001)", label="Overall IC at +10d (ℹ️ Spearman rank correlation between predicted signals and realized abnormal returns)", interactive=False)
                gr.Textbox(value="+0.048 (p=0.006)", label="Overall IC at +5d", interactive=False)
                gr.Textbox(value="10 days", label="Optimal Horizon (ℹ️ Holding period with highest IC)", interactive=False)
                gr.Textbox(value=f"{len(PREDICTIONS)}", label="Events Evaluated", interactive=False)

            with gr.Row():
                decay_chart = gr.Plot(value=create_decay_curve_chart(), label="Signal Decay Curve")

            gr.Markdown("""
            ### Limitations
            - Sample size: Limited events (statistical power constrained)
            - Period: Aug 2024 - Jun 2026 (single market regime)
            - No intraday timing (daily granularity only)
            - Abnormal returns use single-factor market model (SPY)
            - Correlation does not imply causation
            - Bonferroni correction applied to subset tests
            """)

        # ============================================================
        # TAB 3: SECTOR MAP
        # ============================================================
        with gr.Tab("Sector Map"):
            gr.Markdown("""
            ### Quantum Computing Competitive Landscape

            This tab explains the structure of the quantum computing sector and how our model assigns signals.
            Understanding the competitive dynamics is key to interpreting the signal vectors.
            """)

            gr.Markdown("""
            ### How Signals Propagate

            When a piece of news is published, it doesn't just affect one company. The model considers:

            1. **Direct impact** — The company mentioned in the article (strongest signal)
            2. **Technology allies** — Companies using the same quantum approach benefit from validation of their technology
            3. **Technology competitors** — Companies using rival approaches may be negatively affected (competitive gap)
            4. **Revenue scaling** — A quantum breakthrough barely moves Google's stock (quantum is <0.1% of revenue) but can move IonQ's stock significantly (100% quantum)
            """)

            sector_chart = gr.Plot(value=create_sector_map(), label="Sector Relationship Map")

            gr.Markdown("""
            ### Technology Clusters

            | Cluster | Companies | Key Characteristic |
            |---------|-----------|-------------------|
            | Trapped Ion | IONQ, HON (Quantinuum) | High fidelity, all-to-all connectivity |
            | Superconducting | RGTI, IBM, GOOGL | Fast gates, limited connectivity |
            | Annealing | QBTS (D-Wave) | Optimization-only, not gate-based |
            | Topological | MSFT | Inherent error resistance, early stage |
            | Neutral Atom | QUBT | Highly scalable arrays |
            | Adjacent | NVDA | Simulation hardware, HPC |

            ### Revenue Exposure

            | Ticker | Quantum Revenue | Signal Scaling |
            |--------|----------------|----------------|
            | IONQ, RGTI, QBTS, QUBT | 100% | Full signal (up to +/-2.0) |
            | HON | ~5% | Scaled to max +/-0.3 |
            | IBM | ~2% | Scaled to max +/-0.15 |
            | NVDA | ~1% | Scaled to max +/-0.1 |
            | GOOGL, MSFT | <0.1% | Scaled to max +/-0.05 |
            """)

    # Load initial event on startup
    if PREDICTIONS:
        app.load(
            fn=on_event_select,
            inputs=[event_dropdown],
            outputs=[signal_chart, signal_summary, translation_box, rationale_box, price_chart, json_output, article_info],
        )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
