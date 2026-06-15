"""
Inference module for the Alpha Signal Analysis Platform.

Loads the fine-tuned Qwen3-8B model and provides a simple interface
for generating structured signals from raw text input.

For HF Spaces deployment, use the @spaces.GPU decorator for on-demand GPU allocation.
For local testing, the model loads on whatever hardware is available.
"""

import json
import os
from typing import Optional

# Model configuration
MODEL_ID = "basilwong/quantum-alpha-qwen3-8b"
BASE_MODEL_ID = "Qwen/Qwen3-8B"
MAX_NEW_TOKENS = 1024

# System prompt (same as used during training)
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


class QuantumAlphaModel:
    """Wrapper for the fine-tuned Alpha Signal model."""

    def __init__(self, model_id: str = MODEL_ID, use_base: bool = False):
        """
        Initialize the model.

        Args:
            model_id: HF Hub model ID for the fine-tuned model
            use_base: If True, load the base model instead (for comparison)
        """
        self.model_id = model_id if not use_base else BASE_MODEL_ID
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load the model and tokenizer."""
        if self._loaded:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading model: {self.model_id}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        self.model.eval()
        self._loaded = True
        print(f"Model loaded successfully: {self.model_id}")

    def analyze(self, text: str, source: str = "news") -> dict:
        """
        Analyze a piece of text and return structured signals.

        Args:
            text: The raw article/news text to analyze
            source: The source type (news, arxiv, sec_filing, etc.)

        Returns:
            A dictionary containing the structured signal, or an error dict
        """
        if not self._loaded:
            self.load()

        import torch

        user_message = f"Analyze the following {source} content about the quantum computing sector and provide a structured intelligence report:\n\n{text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )

        # Decode only the generated tokens (not the input)
        generated_ids = outputs[0][input_ids.shape[-1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Strip thinking tags if present (Qwen3 sometimes uses <think>...</think>)
        if "<think>" in raw_output:
            # Extract content after </think>
            parts = raw_output.split("</think>")
            if len(parts) > 1:
                raw_output = parts[-1].strip()

        # Parse JSON
        try:
            # Handle potential markdown code blocks
            if raw_output.startswith("```"):
                raw_output = raw_output.split("```")[1]
                if raw_output.startswith("json"):
                    raw_output = raw_output[4:]
                raw_output = raw_output.strip()

            signal = json.loads(raw_output)
            return signal
        except json.JSONDecodeError:
            # Try to find JSON within the output
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            if start != -1 and end > start:
                try:
                    signal = json.loads(raw_output[start:end])
                    return signal
                except json.JSONDecodeError:
                    pass

            return {
                "error": "Failed to parse model output as JSON",
                "raw_output": raw_output[:500],
                "sentiment": "neutral",
                "confidence": 0.0,
                "event_type": None,
                "affected_tickers": [],
                "urgency": "low",
                "technical_translation": None,
            }


# Singleton instance for the app
_model_instance: Optional[QuantumAlphaModel] = None


def get_model(use_base: bool = False) -> QuantumAlphaModel:
    """Get or create the model singleton."""
    global _model_instance
    if _model_instance is None:
        _model_instance = QuantumAlphaModel(use_base=use_base)
    return _model_instance


def analyze_text(text: str, source: str = "news") -> dict:
    """Convenience function to analyze text using the default model."""
    model = get_model()
    return model.analyze(text, source)
