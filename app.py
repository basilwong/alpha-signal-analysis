"""
Quantum Alpha Intelligence Platform
Entry point for Hugging Face Spaces deployment with ZeroGPU support.

This app uses the fine-tuned Qwen3-8B model to analyze quantum computing
news and generate structured trading signals.
"""

import os
import json
import torch
import spaces
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# Model configuration
MODEL_ID = "build-small-hackathon/quantum-alpha-qwen3-8b"

# System prompt (same as training)
SYSTEM_PROMPT = """You are an expert quantum computing financial analyst with deep knowledge of both quantum physics and capital markets. Your role is to analyze news articles, press releases, academic papers, and regulatory filings related to the quantum computing sector and produce structured intelligence reports.

You must output a valid JSON object with the following fields:

{
    "sentiment": "strongly_bearish" | "bearish" | "neutral" | "bullish" | "strongly_bullish",
    "confidence": 0.0 to 1.0,
    "event_type": one of ["physical_qubit_milestone", "logical_qubit_breakthrough", "error_correction_advance", "quantum_volume_increase", "government_funding", "commercial_partnership", "revenue_earnings", "executive_change", "patent_grant", "academic_publication", "product_launch", "competitive_development", "regulatory_filing", "analyst_rating_change"],
    "affected_tickers": ["IONQ", "RGTI", etc.],
    "urgency": "low" | "medium" | "high",
    "technical_translation": "A 2-3 sentence explanation of what this means commercially, written for an investor who does not have a physics background. Explain WHY this matters for the company's competitive position and valuation.",
    "key_facts": ["fact1", "fact2", "fact3"],
    "competitive_context": "How does this development position the company relative to competitors in the quantum computing space?"
}

Guidelines for your analysis:
- Sentiment should reflect the impact on the specific company's stock, not general market sentiment.
- For academic papers, assess whether the research has near-term commercial implications or is purely theoretical.
- Distinguish between physical qubits (less significant individually) and logical qubits (highly significant).
- Error correction advances are typically more significant than raw qubit count increases.
- Government funding announcements are bullish for the entire sector, not just the recipient.
- Be skeptical of press releases that announce "quantum advantage" without peer-reviewed validation.
- Consider the competitive dynamics: a breakthrough by one company may be bearish for competitors.

Output ONLY the JSON object. No additional text, no markdown formatting, no code blocks."""

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
def run_inference(text: str, source: str = "news") -> str:
    """Run model inference with GPU allocation via ZeroGPU."""
    user_message = f"Analyze the following {source} content about the quantum computing sector and provide a structured intelligence report:\n\n{text}"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True, return_dict=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)

    # Strip thinking tags if present
    if "<think>" in raw_output:
        parts = raw_output.split("</think>")
        if len(parts) > 1:
            raw_output = parts[-1].strip()

    return raw_output


def analyze_article(text: str, source: str = "news") -> tuple:
    """
    Analyze a quantum computing article and return structured signals.
    Returns (formatted_json, sentiment_label, confidence, tickers, translation)
    """
    if not text or not text.strip():
        return "Please enter an article to analyze.", "", "", "", ""

    raw_output = run_inference(text, source)

    # Parse JSON from output
    try:
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            signal = json.loads(raw_output[start:end])
        else:
            signal = json.loads(raw_output)

        formatted = json.dumps(signal, indent=2)
        sentiment = signal.get("sentiment", "unknown")
        confidence = f"{signal.get('confidence', 0):.0%}"
        tickers = ", ".join(signal.get("affected_tickers", []))
        translation = signal.get("technical_translation", "")

        return formatted, sentiment, confidence, tickers, translation

    except (json.JSONDecodeError, Exception) as e:
        return f"Parse error: {e}\n\nRaw output:\n{raw_output}", "error", "0%", "", ""


# Build the Gradio interface
with gr.Blocks(
    title="Quantum Alpha Intelligence",
    theme=gr.themes.Base(
        primary_hue="emerald",
        neutral_hue="slate",
    ),
    css="""
    .main-header { text-align: center; margin-bottom: 1rem; }
    .signal-box { border: 1px solid #333; border-radius: 8px; padding: 1rem; }
    """
) as app:

    gr.Markdown(
        """
        # Quantum Alpha Intelligence Platform
        ### NLP-Driven Alpha Signal Generator for the Quantum Computing Sector

        Powered by a fine-tuned Qwen3-8B model trained on quantum computing financial data.
        Paste any quantum computing news article, press release, or research abstract below.
        """,
        elem_classes="main-header"
    )

    with gr.Row():
        with gr.Column(scale=2):
            input_text = gr.Textbox(
                label="Article / News Text",
                placeholder="Paste a quantum computing news article, press release, or arXiv abstract here...",
                lines=10,
            )
            source_type = gr.Dropdown(
                choices=["news", "arxiv", "sec_filing", "press_release", "social_media", "earnings_call"],
                value="news",
                label="Source Type",
            )
            analyze_btn = gr.Button("Analyze", variant="primary", size="lg")

        with gr.Column(scale=2):
            sentiment_output = gr.Textbox(label="Sentiment", interactive=False)
            confidence_output = gr.Textbox(label="Confidence", interactive=False)
            tickers_output = gr.Textbox(label="Affected Tickers", interactive=False)
            translation_output = gr.Textbox(label="Technical Translation (For Investors)", lines=3, interactive=False)

    with gr.Row():
        json_output = gr.Code(label="Full Structured Signal (JSON)", language="json", lines=20)

    # Example articles for quick testing
    gr.Examples(
        examples=[
            ["IonQ announced today that its latest trapped-ion quantum processor has achieved 35 algorithmic qubits, representing a significant improvement in the number of high-fidelity qubits available for running quantum algorithms. The company stated that this milestone brings them closer to achieving quantum advantage for commercially relevant problems in optimization and machine learning. IonQ's stock rose 8% in pre-market trading following the announcement.", "news"],
            ["Rigetti Computing reported Q1 2026 revenue of $4.2 million, missing analyst expectations of $5.1 million. The company cited delays in its 84-qubit Ankaa-3 system deployment as the primary factor. CEO Subodh Kulkarni noted that while hardware development is on track, customer adoption has been slower than anticipated.", "news"],
            ["Title: Demonstration of fault-tolerant quantum computation with 48 logical qubits\n\nAbstract: We demonstrate fault-tolerant quantum computation using 48 logical qubits encoded in a surface code architecture. Our system achieves a logical error rate of 10^-6 per round of error correction, representing a 100x improvement over the physical error rate.", "arxiv"],
        ],
        inputs=[input_text, source_type],
        label="Example Articles (click to load)",
    )

    analyze_btn.click(
        fn=analyze_article,
        inputs=[input_text, source_type],
        outputs=[json_output, sentiment_output, confidence_output, tickers_output, translation_output],
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)
