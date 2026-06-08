"""
Training Data Generation V2: Cross-Sectional Signal Vector Schema

Uses Qwen3-max to generate a full signal vector across all 9 quantum tickers
for each article. This is the updated schema that produces cross-sectional
signals suitable for evaluation with Alphalens.

Usage:
    python scripts/generate_training_data_v2.py --input data/raw/articles.jsonl --output data/training/quantum_alpha_train_v2.jsonl --limit 5
"""

import os
import json
import argparse
import time
from pathlib import Path
from openai import OpenAI

# Alibaba Cloud Model Studio (Singapore region)
DASHSCOPE_API_KEY = os.environ.get(
    "DASHSCOPE_API_KEY",
    "sk-ws-H.IIMPYP.OVEd.MEYCIQCgnJiyfu3TI7aOMuMio4dSrWTf5zbFNrCpKP-NTyUGagIhAJQ6AGEG4uC8C9LmDEqJCLQGSUnilOLV6lQ1QR7QvVBi"
)
DASHSCOPE_BASE_URL = "https://ws-wuyspztgv1cyxvbr.ap-southeast-1.maas.aliyuncs.com/compatible-mode/v1"
TEACHER_MODEL = "qwen3-max"

# New system prompt with cross-sectional signal vector
SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

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

Key domain knowledge for signal generation:
- Trapped-ion breakthroughs are bullish for IONQ and HON, bearish for superconducting competitors (RGTI, IBM, GOOGL)
- Superconducting breakthroughs are bullish for RGTI, IBM, GOOGL, bearish for trapped-ion (IONQ, HON)
- Error correction advances benefit ALL gate-based approaches (not annealing)
- Government funding is broadly bullish for the entire sector
- Revenue exposure matters: a quantum event moves IONQ (100% quantum) much more than GOOGL (<0.1% quantum)
- Scale signals by revenue exposure: GOOGL score should rarely exceed +/-0.1 for quantum-specific news
- Logical qubit milestones are more significant than physical qubit counts
- Academic papers typically have slow signal decay (market is slow to digest technical content)
- Earnings misses have fast decay (priced in immediately)

Output a valid JSON object with this exact structure:

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
    "event_type": "physical_qubit_milestone" | "logical_qubit_breakthrough" | "error_correction_advance" | "government_funding" | "commercial_partnership" | "revenue_earnings" | "executive_change" | "academic_publication" | "product_launch" | "competitive_development" | "regulatory_filing" | "analyst_rating_change" | "sector_macro_event",
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "signal_decay": "fast" | "medium" | "slow",
    "information_novelty": "high" | "medium" | "low",
    "technical_translation": "2-3 sentences explaining the commercial significance for a portfolio manager without physics background.",
    "signal_rationale": "Why these specific scores? What competitive dynamics and revenue exposures justify this distribution?"
}

Score ranges:
- strongly_bullish: +1.5 to +2.0
- bullish: +0.5 to +1.5
- slightly_bullish: +0.1 to +0.5
- neutral: -0.1 to +0.1
- slightly_bearish: -0.5 to -0.1
- bearish: -1.5 to -0.5
- strongly_bearish: -2.0 to -1.5

IMPORTANT: Scale scores by revenue exposure. Pure-play quantum companies (IONQ, RGTI, QBTS, QUBT) can have scores up to +/-2.0. Diversified companies must be scaled: HON max +/-0.3, IBM max +/-0.15, GOOGL/MSFT max +/-0.05, NVDA max +/-0.1.

Output ONLY the JSON object. No additional text."""

# Source-specific instructions
SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess whether it's already widely reported (low novelty) or breaking (high novelty). News typically has fast decay unless highly technical.",
    "arxiv": "This is an academic paper abstract. Most papers are incremental (negligible signal). Only assign significant scores for genuine breakthroughs. Academic signals have slow decay because analysts need time to digest.",
    "sec_filing": "This is a regulatory filing. Extract factual data. High reliability. Fast decay (priced in immediately).",
    "press_release": "This is a company press release. Be skeptical of marketing language. Look for concrete numbers. Discount unvalidated claims.",
    "social_media": "This is a social media post. High noise, low reliability. Assign low scores unless the source is clearly an insider or domain expert.",
    "earnings_call": "This is from an earnings call. Forward guidance matters more than backward results. Look for timeline changes and tone shifts.",
}


def create_client():
    """Create the Qwen Cloud API client."""
    return OpenAI(api_key=DASHSCOPE_API_KEY, base_url=DASHSCOPE_BASE_URL)


def generate_signal_vector(client: OpenAI, article_text: str, source: str = "news") -> dict:
    """Generate a full signal vector for an article using the teacher model."""
    source_instruction = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
    user_prompt = f"{source_instruction}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{article_text}"

    try:
        response = client.chat.completions.create(
            model=TEACHER_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=1500,
        )

        raw_output = response.choices[0].message.content.strip()

        # Parse JSON (handle markdown code blocks)
        if raw_output.startswith("```"):
            raw_output = raw_output.split("```")[1]
            if raw_output.startswith("json"):
                raw_output = raw_output[4:]
            raw_output = raw_output.strip()

        # Find JSON in output
        start = raw_output.find("{")
        end = raw_output.rfind("}") + 1
        if start != -1 and end > start:
            signal = json.loads(raw_output[start:end])
        else:
            signal = json.loads(raw_output)

        # Validate structure
        if "signal_vector" not in signal:
            return None
        if not all(t in signal["signal_vector"] for t in ["IONQ", "RGTI", "QBTS"]):
            return None

        return signal

    except (json.JSONDecodeError, Exception) as e:
        print(f"    ERROR: {e}")
        return None


def format_training_example(article_text: str, source: str, signal: dict) -> dict:
    """Format as an instruction-tuning conversation."""
    source_instruction = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
    user_message = f"{source_instruction}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{article_text}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
            {"role": "assistant", "content": json.dumps(signal, indent=2)},
        ]
    }


def process_articles(input_path: str, output_path: str, limit: int = None):
    """Process articles and generate training data with new schema."""
    client = create_client()

    input_file = Path(input_path)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Load articles
    articles = []
    with open(input_file, "r") as f:
        for line in f:
            if line.strip():
                articles.append(json.loads(line))

    if limit:
        articles = articles[:limit]

    print(f"Processing {len(articles)} articles (V2 signal vector schema)...")
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

            print(f"[{i+1}/{len(articles)}] {title[:70]}...")

            signal = generate_signal_vector(client, text, source)

            if signal:
                example = format_training_example(text, source, signal)
                f_out.write(json.dumps(example) + "\n")
                successful += 1

                # Print summary
                sv = signal["signal_vector"]
                top = max(sv.items(), key=lambda x: x[1]["score"])
                bot = min(sv.items(), key=lambda x: x[1]["score"])
                print(f"    Top: {top[0]} ({top[1]['score']:+.2f}) | Bot: {bot[0]} ({bot[1]['score']:+.2f}) | {signal.get('event_type', 'N/A')}")
            else:
                failed += 1
                print(f"    FAILED")

            time.sleep(1)  # Rate limiting

    print("=" * 60)
    print(f"Complete! {successful} successful, {failed} failed")
    print(f"Training data saved to: {output_path}")
    return {"successful": successful, "failed": failed}


def main():
    parser = argparse.ArgumentParser(description="Generate V2 training data (signal vector schema)")
    parser.add_argument("--input", type=str, default="data/raw/articles.jsonl")
    parser.add_argument("--output", type=str, default="data/training/quantum_alpha_train_v2.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    process_articles(args.input, args.output, args.limit)


if __name__ == "__main__":
    main()
