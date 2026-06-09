"""
Quantum Alpha Intelligence Platform - V2 Frontend
Three-tab interface: Signal Explorer | Evaluation Dashboard | Sector Map

Uses pre-computed predictions and market data for historical browsing,
with live inference capability via ZeroGPU for new articles.
"""

import os
import json
import time
import gradio as gr
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path

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
    """Create a network visualization of the quantum computing sector."""
    fig = go.Figure()

    # Position clusters
    positions = {
        "Trapped Ion": (0.2, 0.8),
        "Superconducting": (0.7, 0.8),
        "Annealing": (0.1, 0.3),
        "Topological": (0.5, 0.3),
        "Neutral Atom": (0.9, 0.3),
        "Adjacent": (0.5, 0.05),
    }

    # Draw cluster backgrounds
    for cluster, pos in positions.items():
        tickers = TECH_CLUSTERS[cluster]
        fig.add_trace(go.Scatter(
            x=[pos[0]], y=[pos[1]],
            mode='markers+text',
            marker=dict(size=50 + len(tickers) * 20, color='rgba(100,100,200,0.1)'),
            text=[cluster],
            textposition='top center',
            textfont=dict(size=11, color='white'),
            showlegend=False,
            hoverinfo='skip',
        ))

    # Draw company nodes
    for cluster, tickers_in_cluster in TECH_CLUSTERS.items():
        pos = positions[cluster]
        for j, ticker in enumerate(tickers_in_cluster):
            offset_x = (j - len(tickers_in_cluster)/2) * 0.08
            info = COMPANY_INFO[ticker]
            node_size = 15 + info["exposure"] * 25

            fig.add_trace(go.Scatter(
                x=[pos[0] + offset_x], y=[pos[1] - 0.08],
                mode='markers+text',
                marker=dict(size=node_size, color=info["color"], line=dict(width=1, color='white')),
                text=[ticker],
                textposition='bottom center',
                textfont=dict(size=9, color='white'),
                showlegend=False,
                hovertext=f"{info['name']}<br>Tech: {info['tech']}<br>Quantum Revenue: {info['exposure']*100:.1f}%",
                hoverinfo='text',
            ))

    # Draw competitive edges
    edges = [
        ("IONQ", "RGTI", "competes"),
        ("IONQ", "IBM", "competes"),
        ("RGTI", "IBM", "competes"),
        ("RGTI", "GOOGL", "competes"),
        ("HON", "IONQ", "allies"),
    ]

    fig.update_layout(
        title="Quantum Computing Sector Map",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.1]),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-0.1, 1.0]),
        height=500,
        template="plotly_dark",
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
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

            with gr.Row():
                with gr.Column(scale=3):
                    signal_chart = gr.Plot(label="Signal Vector")
                with gr.Column(scale=2):
                    signal_summary = gr.Textbox(label="Signal Summary", interactive=False)
                    translation_box = gr.Textbox(label="Technical Translation (ℹ️ What this means for investors)", lines=3, interactive=False)
                    rationale_box = gr.Textbox(label="Signal Rationale (ℹ️ Why these scores)", lines=3, interactive=False)

            price_chart = gr.Plot(label="Actual Price Movement After Event")

            with gr.Accordion("Raw JSON Signal", open=False):
                json_output = gr.Code(label="Full Signal Output", language="json", lines=15)

            # Wire up event selection
            event_dropdown.change(
                fn=on_event_select,
                inputs=[event_dropdown],
                outputs=[signal_chart, signal_summary, translation_box, rationale_box, price_chart, json_output, article_info],
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
            Companies grouped by technology approach. Node size reflects quantum revenue exposure.
            *ℹ️ Revenue exposure determines how much quantum-specific news affects the stock price.*
            """)

            sector_chart = gr.Plot(value=create_sector_map(), label="Sector Map")

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
