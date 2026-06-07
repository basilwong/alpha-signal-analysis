"""
Quantum Alpha Intelligence Platform
NLP-driven alpha signal generator for quantitative trading systems.
Designed for integration into automated trading pipelines.
"""

import os
import json
import time
import torch
import spaces
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration
MODEL_ID = "build-small-hackathon/quantum-alpha-qwen3-8b"

# Quant-focused system prompt with improved output schema
SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. Your job is to extract actionable trading signals from text that can be consumed by an automated execution system.

You must output a valid JSON object with the following fields:

{
    "sentiment": "strongly_bearish" | "bearish" | "neutral" | "bullish" | "strongly_bullish",
    "expected_move_magnitude": "negligible" | "moderate" | "significant" | "major",
    "expected_move_pct_range": [lower_bound, upper_bound],
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "signal_decay": "fast" | "medium" | "slow",
    "information_novelty": "high" | "medium" | "low",
    "event_type": one of ["physical_qubit_milestone", "logical_qubit_breakthrough", "error_correction_advance", "quantum_volume_increase", "government_funding", "commercial_partnership", "revenue_earnings", "executive_change", "patent_grant", "academic_publication", "product_launch", "competitive_development", "regulatory_filing", "analyst_rating_change"],
    "primary_ticker": "IONQ" | "RGTI" | "QBTS" | "QUBT" | "IBM" | "GOOGL" | "MSFT" | "HON" | "NVDA",
    "cross_asset_signals": [
        {"ticker": "XXXX", "direction": "bullish" | "bearish", "magnitude": "negligible" | "moderate" | "significant", "reason": "brief explanation"}
    ],
    "technical_translation": "2-3 sentences explaining the commercial significance for a portfolio manager who does not have a physics background.",
    "key_facts": ["fact1", "fact2", "fact3"],
    "signal_rationale": "Why this specific magnitude and time horizon? What historical precedent or sector dynamics justify this estimate?"
}

Guidelines:
- expected_move_pct_range: Estimate the likely stock price move as [low%, high%]. Use negative values for bearish. Example: [3.0, 8.0] or [-12.0, -5.0].
- signal_decay: "fast" = priced in same session, "medium" = 2-3 days, "slow" = gradual diffusion over 1-2 weeks (typical for highly technical announcements that analysts need time to interpret).
- information_novelty: "high" = first to report or pre-publication leak, "medium" = same-day coverage, "low" = widely reported for 24h+.
- cross_asset_signals: Identify second-order effects on competitors or adjacent companies. A breakthrough by one company is often bearish for direct competitors.
- Logical qubit milestones and error correction advances are typically "significant" to "major" with "slow" decay because most market participants don't understand the technical implications.
- Revenue misses are typically "fast" decay (priced in immediately).
- Government funding is "moderate" magnitude with "medium" decay across the entire sector.

Output ONLY the JSON object. No additional text."""

# Load model and tokenizer at startup
print(f"Loading tokenizer for {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
print("Tokenizer loaded.")

print(f"Loading model {MODEL_ID}...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print("Model loaded successfully!")


@spaces.GPU
def run_inference(text: str, source: str = "news", enable_thinking: bool = False) -> tuple:
    """Run model inference. Returns (raw_output, thinking_text, latency_ms)."""
    start_time = time.time()

    user_message = f"Analyze the following {source} content and generate a quantitative trading signal:\n\n{text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # Apply chat template with thinking toggle
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        add_generation_prompt=True,
        return_dict=True,
        enable_thinking=enable_thinking,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048 if enable_thinking else 1024,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    latency_ms = int((time.time() - start_time) * 1000)

    # Separate thinking from output
    thinking_text = ""
    signal_text = raw_output

    if "<think>" in raw_output:
        parts = raw_output.split("</think>")
        if len(parts) > 1:
            thinking_text = parts[0].replace("<think>", "").strip()
            signal_text = parts[-1].strip()

    return signal_text, thinking_text, latency_ms


def analyze_article(text: str, source: str, enable_thinking: bool) -> tuple:
    """
    Analyze a quantum computing article and return structured signals.
    Returns (json_output, sentiment, magnitude, time_horizon, tickers, translation, thinking, api_response)
    """
    if not text or not text.strip():
        return "Please enter text to analyze.", "", "", "", "", "", ""

    signal_text, thinking_text, latency_ms = run_inference(text, source, enable_thinking)

    # Parse JSON from output
    try:
        start = signal_text.find("{")
        end = signal_text.rfind("}") + 1
        if start != -1 and end > start:
            signal = json.loads(signal_text[start:end])
        else:
            signal = json.loads(signal_text)

        # Extract display fields
        formatted_json = json.dumps(signal, indent=2)
        sentiment = signal.get("sentiment", "unknown")
        magnitude = signal.get("expected_move_magnitude", "unknown")
        pct_range = signal.get("expected_move_pct_range", [0, 0])
        pct_str = f"{pct_range[0]:+.1f}% to {pct_range[1]:+.1f}%" if isinstance(pct_range, list) and len(pct_range) == 2 else str(pct_range)
        time_horizon = signal.get("time_horizon", "unknown")
        decay = signal.get("signal_decay", "unknown")
        novelty = signal.get("information_novelty", "unknown")

        primary = signal.get("primary_ticker", "")
        cross = signal.get("cross_asset_signals", [])
        cross_str = ", ".join([f"{c['ticker']} ({c['direction']})" for c in cross]) if cross else "None"
        tickers_display = f"{primary} (primary) | Cross-asset: {cross_str}"

        translation = signal.get("technical_translation", "")

        # Build the API response mock
        api_response = json.dumps({
            "status": "success",
            "latency_ms": latency_ms,
            "model": MODEL_ID,
            "thinking_enabled": enable_thinking,
            "signal": signal,
            "metadata": {
                "source_type": source,
                "input_length_chars": len(text),
                "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
        }, indent=2)

        # Build summary line with latency
        summary = f"{sentiment.upper()} | {magnitude} ({pct_str}) | Horizon: {time_horizon} | Decay: {decay} | Novelty: {novelty} | Latency: {latency_ms}ms"

        return formatted_json, summary, tickers_display, translation, thinking_text, api_response, f"{latency_ms}ms"

    except (json.JSONDecodeError, Exception) as e:
        error_api = json.dumps({
            "status": "error",
            "latency_ms": latency_ms,
            "error": str(e),
            "raw_output": signal_text[:500],
        }, indent=2)
        return f"Parse error: {e}\n\nRaw:\n{signal_text}", "ERROR", "", "", thinking_text, error_api, f"{latency_ms}ms"


# Build the Gradio interface
with gr.Blocks(
    title="Quantum Alpha Intelligence",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
    ),
) as app:

    gr.Markdown(
        """
        # Quantum Alpha Intelligence
        ### Quantitative NLP Signal Generator for the Quantum Computing Sector

        Generates structured trading signals from unstructured text. Designed for integration
        into automated quantitative trading pipelines. Powered by a fine-tuned Qwen3-8B model.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            input_text = gr.Textbox(
                label="Input Text (Article / Press Release / Abstract)",
                placeholder="Paste quantum computing news, press release, arXiv abstract, or earnings excerpt...",
                lines=8,
            )
            with gr.Row():
                source_type = gr.Dropdown(
                    choices=["news", "arxiv", "sec_filing", "press_release", "social_media", "earnings_call"],
                    value="news",
                    label="Source Type",
                    scale=2,
                )
                thinking_toggle = gr.Checkbox(
                    label="Enable Thinking (slower, shows reasoning)",
                    value=False,
                    scale=2,
                )
            analyze_btn = gr.Button("Generate Signal", variant="primary", size="lg")

        with gr.Column(scale=2):
            with gr.Row():
                signal_summary = gr.Textbox(label="Signal Summary", interactive=False, scale=4)
                latency_output = gr.Textbox(label="Latency", interactive=False, scale=1)
            tickers_output = gr.Textbox(label="Affected Tickers & Cross-Asset Signals", interactive=False)
            translation_output = gr.Textbox(label="Technical Translation (For Portfolio Managers)", lines=3, interactive=False)

    with gr.Row():
        with gr.Column():
            json_output = gr.Code(label="Structured Signal (JSON)", language="json", lines=18)
        with gr.Column():
            thinking_output = gr.Textbox(label="Model Reasoning (when thinking enabled)", lines=18, interactive=False)

    with gr.Accordion("API Response (for system integration)", open=False):
        api_output = gr.Code(label="Mock API Response", language="json", lines=20)

    # Examples
    gr.Examples(
        examples=[
            ["IonQ announced today that its latest trapped-ion quantum processor has achieved 35 algorithmic qubits, representing a significant improvement in the number of high-fidelity qubits available for running quantum algorithms. The company stated that this milestone brings them closer to achieving quantum advantage for commercially relevant problems in optimization and machine learning. IonQ's stock rose 8% in pre-market trading following the announcement.", "news", False],
            ["Rigetti Computing reported Q1 2026 revenue of $4.2 million, missing analyst expectations of $5.1 million. The company cited delays in its 84-qubit Ankaa-3 system deployment as the primary factor. CEO Subodh Kulkarni noted that while hardware development is on track, customer adoption has been slower than anticipated.", "news", False],
            ["Title: Demonstration of fault-tolerant quantum computation with 48 logical qubits\n\nAbstract: We demonstrate fault-tolerant quantum computation using 48 logical qubits encoded in a surface code architecture. Our system achieves a logical error rate of 10^-6 per round of error correction, representing a 100x improvement over the physical error rate. This result establishes a clear path toward scalable, fault-tolerant quantum computing.", "arxiv", True],
        ],
        inputs=[input_text, source_type, thinking_toggle],
        label="Example Inputs",
    )

    analyze_btn.click(
        fn=analyze_article,
        inputs=[input_text, source_type, thinking_toggle],
        outputs=[json_output, signal_summary, tickers_output, translation_output, thinking_output, api_output, latency_output],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
