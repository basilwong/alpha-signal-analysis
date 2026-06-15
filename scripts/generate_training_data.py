"""
Training Data Generation Pipeline for Alpha Signal Analysis Platform

Uses Qwen3.7-Max (via Qwen Cloud / DashScope API) as the teacher model
to generate high-quality labeled training examples for fine-tuning Qwen3-8B.

The pipeline:
1. Loads raw quantum computing news/articles from data/raw/
2. Sends each article to Qwen3.7-Max with a structured prompt
3. Receives structured JSON output (sentiment, event type, tickers, translation)
4. Saves the instruction-tuning pairs to data/training/

Usage:
    export DASHSCOPE_API_KEY="your-key-here"
    python scripts/generate_training_data.py --input data/raw/articles.jsonl --output data/training/alpha_signal_train.jsonl
"""

import os
import json
import argparse
import time
from pathlib import Path
from openai import OpenAI

# Alibaba Cloud Model Studio (Singapore region) configuration
DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY",
    "sk-ws-H.IIMPYP.OVEd.MEYCIQCgnJiyfu3TI7aOMuMio4dSrWTf5zbFNrCpKP-NTyUGagIhAJQ6AGEG4uC8C9LmDEqJCLQGSUnilOLV6lQ1QR7QvVBi"
)
DASHSCOPE_BASE_URL = "https://ws-wuyspztgv1cyxvbr.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
TEACHER_MODEL = "qwen3-max"

# System prompt that defines the teacher's role and output format
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


def create_client():
    """Create the Qwen Cloud API client."""
    if not DASHSCOPE_API_KEY:
        raise ValueError(
            "DASHSCOPE_API_KEY environment variable not set. "
            "Get your key at https://home.qwencloud.com/api-keys"
        )
    return OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url=DASHSCOPE_BASE_URL,
    )


def generate_label(client: OpenAI, article_text: str, source: str = "unknown") -> dict:
    """
    Send an article to the teacher model and get a structured label back.

    Args:
        client: The OpenAI-compatible API client
        article_text: The raw text of the article/news item
        source: The source of the article (e.g., "reuters", "arxiv", "sec_filing")

    Returns:
        A dictionary containing the structured label from the teacher model
    """
    user_prompt = f"""Analyze the following {source} content about the quantum computing sector:

---
{article_text}
---

Produce your structured JSON analysis."""

    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Low temperature for consistent, factual outputs
            max_tokens=1024,
        )

        raw_output = response.choices[0].message.content.strip()

        # Parse the JSON output
        # Handle potential markdown code blocks
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
            raw_output = raw_output.strip()

        label = json.loads(raw_output)
        return label

    except json.JSONDecodeError as e:
        print(f"  WARNING: Failed to parse JSON from teacher model: {e}")
        print(f"  Raw output: {raw_output[:200]}...")
        return None
    except Exception as e:
        print(f"  ERROR: API call failed: {e}")
        return None


def format_training_example(article_text: str, source: str, label: dict) -> dict:
    """
    Format a single training example in the chat/messages format
    expected by the fine-tuning script.

    Returns a dictionary with a "messages" field containing the conversation.
    """
    user_message = f"Analyze the following {source} content about the quantum computing sector and provide a structured intelligence report:\n\n{article_text}"

    assistant_message = json.dumps(label, indent=2)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": assistant_message},
        ]
    }


def process_articles(input_path: str, output_path: str, limit: int = None):
    """
    Process a JSONL file of articles and generate training data.

    Expected input format (one JSON per line):
    {"text": "article content...", "source": "reuters", "title": "..."}
    """
    client = create_client()

    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load input articles
    articles = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    if limit:
        articles = articles[:limit]

    print(f"Processing {len(articles)} articles...")
    print(f"Teacher model: {TEACHER_MODEL}")
    print(f"Output: {output_path}")
    print("=" * 60)

    successful = 0
    failed = 0

    with open(output_file, "w") as f_out:
        for i, article in enumerate(articles):
            text = article.get("text", "")
            source = article.get("source", "news")
            title = article.get("title", "Untitled")

            print(f"[{i+1}/{len(articles)}] Processing: {title[:60]}...")

            # Generate the label from the teacher model
            label = generate_label(client, text, source)

            if label:
                # Format as training example
                example = format_training_example(text, source, label)
                f_out.write(json.dumps(example) + "\n")
                successful += 1
                print(f"  -> {label['sentiment']} | {label['event_type']} | {label['affected_tickers']}")
            else:
                failed += 1
                print(f"  -> FAILED")

            # Rate limiting (be respectful of free tier)
            time.sleep(1)

    print("=" * 60)
    print(f"Complete! {successful} successful, {failed} failed")
    print(f"Training data saved to: {output_path}")

    return {"successful": successful, "failed": failed, "output": output_path}


def main():
    parser = argparse.ArgumentParser(description="Generate training data using teacher model")
    parser.add_argument(
        "--input", type=str, default="data/raw/articles.jsonl",
        help="Path to input JSONL file with raw articles"
    )
    parser.add_argument(
        "--output", type=str, default="data/training/alpha_signal_train.jsonl",
        help="Path to output JSONL file for training data"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit the number of articles to process (for testing)"
    )

    args = parser.parse_args()
    process_articles(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
