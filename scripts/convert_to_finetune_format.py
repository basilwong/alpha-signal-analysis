"""
Convert training data to fine-tuning chat format (system/user/assistant).
Produces JSONL compatible with Qwen3-8B fine-tuning via Unsloth/TRL.

Incorporates all fixes:
- Updated system prompt (Fixes 2, 14, ticker universe)
- Market context prepended to user message (Fix 3a)
- No signal_decay in output (Fix 5)
- 10 tickers including QNT (Fix 1)
- chain_of_thought repaired (Fix 16)

Usage:
    python scripts/convert_to_finetune_format.py
"""

import json
import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_TRAINING = PROJECT_ROOT / "data" / "training"
OUTPUT_FILE = DATA_TRAINING / "quantum_alpha_train_v4.jsonl"

# ============================================================
# System Prompt (incorporates all fixes)
# ============================================================

SYSTEM_PROMPT = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

The quantum computing universe consists of these 10 tickers:

**Active (scored):**
- IONQ: IonQ (trapped-ion, 100% quantum revenue, pure-play)
- RGTI: Rigetti Computing (superconducting, 100% quantum revenue, pure-play)
- QBTS: D-Wave Quantum (quantum annealing, 100% quantum revenue, pure-play)
- QUBT: Quantum Computing Inc. (neutral atom, 100% quantum revenue, pure-play)
- QNT: Quantinuum (trapped-ion, 100% quantum revenue, pure-play, IPO'd June 2026)
- IBM: International Business Machines (superconducting, ~2% quantum revenue)
- HON: Honeywell (trapped-ion, ~1% quantum revenue post-Quantinuum spinoff)

**Inactive (always 0.0):**
- MSFT: Microsoft — quantum revenue <0.1%, signal is noise
- GOOGL: Alphabet/Google — quantum revenue <0.1%, signal is noise
- NVDA: NVIDIA — moves on AI/GPU demand, not quantum news

Key domain knowledge:
- Trapped-ion breakthroughs: bullish IONQ/QNT/HON, bearish RGTI/IBM
- Superconducting breakthroughs: bullish RGTI/IBM, bearish IONQ/QNT/HON
- Error correction advances: benefit ALL gate-based approaches
- Government funding: broadly bullish for entire sector
- IONQ and QNT are direct competitors (both trapped-ion pure-play)
- Company-specific events: IONQ and QNT may move opposite (zero-sum)
- Sector-wide events: IONQ and QNT move together

Score ranges (MUST respect):
- Pure-play (IONQ, RGTI, QBTS, QUBT, QNT): [-2.0, +2.0]
- HON: [-0.3, +0.3]
- IBM: [-0.15, +0.15]
- MSFT, GOOGL, NVDA: always 0.0

Minimum conviction rule:
- If you have no specific reason to believe this news will move a stock, assign 0.0
- Do not guess directional scores when you lack conviction
- "No opinion" (0.0) is valid and often correct
- Most news does not meaningfully move stocks
- When in doubt, 0.0 is better than a small guess

ArXiv paper rules:
- Default maximum absolute score: 0.5
- Exception: company-authored hardware papers with measured metrics → up to 1.0
- Pure theory or unrelated quantum physics → all scores 0.0
- Most academic papers do NOT move stocks

Market context awareness:
- Consider recent price action when assigning scores
- If a stock is already up significantly, bullish news may be priced in
- Be more conservative on low-liquidity names
- In high-volatility environments, signals decay faster

Output a valid JSON object with this exact structure:
{
    "signal_vector": {
        "IONQ": {"score": float, "reasoning": "1-2 sentences"},
        "RGTI": {"score": float, "reasoning": "1-2 sentences"},
        "QBTS": {"score": float, "reasoning": "1-2 sentences"},
        "QUBT": {"score": float, "reasoning": "1-2 sentences"},
        "QNT": {"score": float, "reasoning": "1-2 sentences"},
        "IBM": {"score": float, "reasoning": "1-2 sentences"},
        "HON": {"score": float, "reasoning": "1-2 sentences"},
        "MSFT": {"score": 0.0, "reasoning": "Inactive: quantum revenue exposure too low for meaningful signal."},
        "GOOGL": {"score": 0.0, "reasoning": "Inactive: quantum revenue exposure too low for meaningful signal."},
        "NVDA": {"score": 0.0, "reasoning": "Inactive: anti-predictive, moves on AI/GPU demand not quantum news."}
    },
    "event_type": "descriptive event category",
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "information_novelty": "high" | "medium" | "low",
    "technical_translation": "2-3 sentences explaining commercial significance for a portfolio manager.",
    "signal_rationale": "Why these specific scores? What competitive dynamics justify this distribution?",
    "chain_of_thought": "Step-by-step reasoning: what the article says, which technology it relates to, significance, market pricing speed, second-order effects."
}

Output ONLY the JSON object. No additional text, no markdown, no code blocks."""

# Source-specific instructions
SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess whether it's already widely reported (low novelty) or breaking (high novelty).",
    "arxiv": "This is an academic paper. Most papers are incremental (score 0.0). Only assign significant scores for genuine breakthroughs with commercial implications.",
    "sec_filing": "This is a regulatory filing. High reliability. Fast decay.",
    "press_release": "Company press release. Be skeptical of marketing language.",
    "social_media": "Social media post. High noise, low reliability.",
    "earnings_call": "Earnings call. Forward guidance matters most.",
}


def build_user_message(record: dict) -> str:
    """Build the user message from a training record."""
    parts = []
    
    # Market context (if available)
    market_context = record.get("market_context", "")
    if market_context:
        parts.append(market_context)
    
    # Source instruction
    source = record.get("source", "news")
    instruction = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
    parts.append(instruction)
    
    # Article/scenario content
    parts.append("\nAnalyze the following content and generate a cross-sectional signal vector:\n")
    
    if record.get("title"):
        parts.append(f"Title: {record['title']}")
    if record.get("date"):
        parts.append(f"Date: {record['date']}")
    if record.get("text"):
        parts.append(f"\n{record['text']}")
    elif record.get("scenario"):
        parts.append(f"\nScenario: {record['scenario']}")
    
    return "\n".join(parts)


def build_assistant_message(signal: dict) -> str:
    """Build the assistant response (clean JSON output)."""
    # Ensure correct field order and no extra fields
    output = {
        "signal_vector": signal.get("signal_vector", {}),
        "event_type": signal.get("event_type", ""),
        "time_horizon": signal.get("time_horizon", ""),
        "information_novelty": signal.get("information_novelty", ""),
        "technical_translation": signal.get("technical_translation", ""),
        "signal_rationale": signal.get("signal_rationale", ""),
        "chain_of_thought": signal.get("chain_of_thought", ""),
    }
    
    # Remove signal_decay if somehow present
    output.pop("signal_decay", None)
    
    return json.dumps(output, indent=2)


def convert_record(record: dict) -> dict:
    """Convert a single training record to chat format."""
    user_msg = build_user_message(record)
    assistant_msg = build_assistant_message(record["signal"])
    
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ]
    }


def main():
    print("Converting training data to fine-tuning format...")
    print(f"Output: {OUTPUT_FILE}")
    print()
    
    # Load all successful training data
    all_records = []
    
    # Combined (original 1000)
    combined_file = DATA_TRAINING / "manus_teacher_combined.jsonl"
    with open(combined_file) as f:
        combined = [json.loads(l) for l in f if l.strip()]
    successful_combined = [r for r in combined if r.get("success") and r.get("signal")]
    all_records.extend(successful_combined)
    print(f"  Combined: {len(successful_combined)} successful examples")
    
    # ArXiv rebalance
    arxiv_file = DATA_TRAINING / "manus_arxiv_rebalance.jsonl"
    if arxiv_file.exists():
        with open(arxiv_file) as f:
            arxiv = [json.loads(l) for l in f if l.strip()]
        successful_arxiv = [r for r in arxiv if r.get("success") and r.get("signal")]
        all_records.extend(successful_arxiv)
        print(f"  ArXiv: {len(successful_arxiv)} successful examples")
    
    # QNT examples
    qnt_file = DATA_TRAINING / "manus_qnt_examples.jsonl"
    if qnt_file.exists():
        with open(qnt_file) as f:
            qnt = [json.loads(l) for l in f if l.strip()]
        successful_qnt = [r for r in qnt if r.get("success") and r.get("signal")]
        all_records.extend(successful_qnt)
        print(f"  QNT: {len(successful_qnt)} successful examples")
    
    print(f"\n  Total records to convert: {len(all_records)}")
    
    # Convert
    converted = []
    skipped = 0
    issues = []
    
    for i, record in enumerate(all_records):
        signal = record["signal"]
        
        # Quality gate: skip if chain_of_thought is still broken
        cot = signal.get("chain_of_thought", "")
        if len(cot) < 30:
            skipped += 1
            issues.append(f"idx={record.get('article_idx')}: CoT too short ({len(cot)} chars)")
            continue
        
        # Quality gate: skip if signal_vector is incomplete
        sv = signal.get("signal_vector", {})
        if len(sv) < 9:  # At least 9 tickers (some old ones might not have QNT)
            skipped += 1
            issues.append(f"idx={record.get('article_idx')}: Only {len(sv)} tickers in signal_vector")
            continue
        
        # Convert
        try:
            chat_record = convert_record(record)
            converted.append(chat_record)
        except Exception as e:
            skipped += 1
            issues.append(f"idx={record.get('article_idx')}: Conversion error: {e}")
    
    print(f"\n  Converted: {len(converted)}")
    print(f"  Skipped: {skipped}")
    
    if issues:
        print(f"\n  Issues ({len(issues)}):")
        for issue in issues[:10]:
            print(f"    {issue}")
        if len(issues) > 10:
            print(f"    ... and {len(issues) - 10} more")
    
    # Write output
    with open(OUTPUT_FILE, "w") as f:
        for record in converted:
            f.write(json.dumps(record) + "\n")
    
    print(f"\n  Written to: {OUTPUT_FILE}")
    print(f"  File size: {OUTPUT_FILE.stat().st_size / 1024 / 1024:.1f} MB")
    
    # Sanity checks on output
    print("\n--- Output Sanity Checks ---")
    
    # Token length estimates
    lengths = []
    for record in converted:
        total_chars = sum(len(msg["content"]) for msg in record["messages"])
        lengths.append(total_chars)
    
    avg_tokens = sum(lengths) / len(lengths) / 4
    max_tokens = max(lengths) / 4
    over_limit = sum(1 for l in lengths if l / 4 > 4096)
    
    print(f"  Avg estimated tokens: {avg_tokens:.0f}")
    print(f"  Max estimated tokens: {max_tokens:.0f}")
    print(f"  Over 4096 limit: {over_limit}")
    
    if max_tokens > 4096:
        print(f"  ⚠️  WARNING: {over_limit} examples may exceed model context window!")
    else:
        print(f"  ✓ All examples fit within 4096 token limit")
    
    # Category distribution
    cats = Counter()
    for record in all_records[:len(converted)]:
        cats[record.get("category", "unknown")] += 1
    
    print(f"\n  Category distribution:")
    for cat, count in cats.most_common():
        print(f"    {cat}: {count} ({count/len(converted)*100:.0f}%)")
    
    # Source distribution (for the arxiv rebalancing check)
    sources = Counter()
    for record in all_records[:len(converted)]:
        sources[record.get("source", record.get("arxiv_tier", "other"))] += 1
    
    print(f"\n  Source distribution:")
    for src, count in sources.most_common():
        print(f"    {src}: {count} ({count/len(converted)*100:.0f}%)")
    
    # Sample output
    print(f"\n--- Sample Output (first example) ---")
    sample = converted[0]
    print(f"  System prompt: {len(sample['messages'][0]['content'])} chars")
    print(f"  User message: {len(sample['messages'][1]['content'])} chars")
    print(f"  Assistant response: {len(sample['messages'][2]['content'])} chars")
    print(f"  User preview: {sample['messages'][1]['content'][:150]}...")
    
    # Verify JSON parseable
    parse_errors = 0
    for record in converted:
        try:
            json.loads(record["messages"][2]["content"])
        except json.JSONDecodeError:
            parse_errors += 1
    
    print(f"\n  Assistant JSON parse errors: {parse_errors}")
    if parse_errors == 0:
        print(f"  ✓ All assistant messages are valid JSON")
    else:
        print(f"  ⚠️  {parse_errors} assistant messages have JSON parse errors!")


if __name__ == "__main__":
    main()
