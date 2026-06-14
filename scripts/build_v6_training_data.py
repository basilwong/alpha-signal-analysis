"""
Build V6 training data by:
1. Taking the base V5 data (881 examples, already in messages format)
2. Converting the bearish/robustness supplements from flat format to messages format
3. Combining into a single quantum_alpha_train_v6.jsonl

Usage:
    python scripts/build_v6_training_data.py
"""

import json
from pathlib import Path

BASE_FILE = "data/training/quantum_alpha_train_v5.jsonl"
SUPPLEMENT_FILES = [
    "data/training/quantum_alpha_train_v5_bearish.jsonl",
    "data/training/quantum_alpha_train_v5_bearish_b2.jsonl",
    "data/training/quantum_alpha_train_v5_robustness.jsonl",
]
OUTPUT_FILE = "data/training/quantum_alpha_train_v6.jsonl"

# Extract system prompt from the base V5 data
with open(BASE_FILE) as f:
    first_example = json.loads(f.readline())
    SYSTEM_PROMPT = first_example["messages"][0]["content"]


def convert_supplement_example(raw: dict) -> dict:
    """
    Convert a flat-format supplement example into OpenAI messages format.
    
    Input format:
        {
            "thinking": "...",
            "signal": {"signal_vector": {...}, "event_type": ..., ...},
            "scenario": "...",
            "market_context": "...",
            ...
        }
    
    Output format:
        {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "[MARKET CONTEXT]\n\n[ARTICLE]\n..."},
                {"role": "assistant", "content": "<think>\n...\n</think>\n{JSON}"}
            ]
        }
    """
    thinking = raw.get("thinking", "")
    signal = raw.get("signal", {})
    scenario = raw.get("scenario", "")
    market_context = raw.get("market_context", "")
    category = raw.get("category", "news")

    # Build user message
    user_parts = []
    if market_context:
        user_parts.append(market_context)

    # Use the scenario as the article content
    source_instructions = {
        "earnings": "This is an earnings report. Forward guidance matters most.",
        "competitive": "This is a competitive dynamics event. Assess impact on all players.",
        "technical_setback": "This is a technical setback report. Assess severity and duration.",
        "dilution": "This is a dilution/funding event. Assess shareholder impact.",
        "executive": "This is an executive change. Assess leadership impact.",
        "macro": "This is a macro/sector event. Assess broad impact.",
        "news": "This is a financial news article. Assess novelty and likely decay speed.",
    }
    source_inst = source_instructions.get(category, source_instructions["news"])
    user_parts.append(source_inst)
    user_parts.append(f"\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{scenario}")

    user_content = "\n\n".join(user_parts)

    # Build assistant message: <think>...</think> + JSON
    assistant_content = f"<think>\n{thinking}\n</think>\n{json.dumps(signal)}"

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content},
        ]
    }


def main():
    examples = []

    # Load base V5 (already in correct format)
    print(f"Loading base V5: {BASE_FILE}")
    with open(BASE_FILE) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    print(f"  Loaded {len(examples)} base examples")

    # Convert and add supplements
    for supp_file in SUPPLEMENT_FILES:
        path = Path(supp_file)
        if not path.exists():
            print(f"  SKIP (not found): {supp_file}")
            continue

        converted = 0
        skipped = 0
        with open(supp_file) as f:
            for line in f:
                if not line.strip():
                    continue
                raw = json.loads(line)

                # Skip failed examples
                if not raw.get("success", False):
                    skipped += 1
                    continue

                # Skip if no signal
                if not raw.get("signal") or not raw.get("signal", {}).get("signal_vector"):
                    skipped += 1
                    continue

                example = convert_supplement_example(raw)
                examples.append(example)
                converted += 1

        print(f"  {path.name}: converted {converted}, skipped {skipped}")

    # Write combined V6
    print(f"\nWriting {OUTPUT_FILE} ({len(examples)} total examples)")
    with open(OUTPUT_FILE, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    # Quick stats
    bullish = 0
    bearish = 0
    zero = 0
    for ex in examples:
        assistant = ex["messages"][2]["content"]
        if "</think>" in assistant:
            json_part = assistant[assistant.index("</think>") + 8:].strip()
        else:
            json_part = assistant.strip()

        start_j = json_part.find("{")
        end_j = json_part.rfind("}") + 1
        if start_j >= 0 and end_j > start_j:
            try:
                signal = json.loads(json_part[start_j:end_j])
                sv = signal.get("signal_vector", {})
                for ticker in ["IONQ", "RGTI", "QBTS", "QUBT", "QNT", "IBM", "HON"]:
                    if ticker in sv:
                        score = sv[ticker].get("score", 0)
                        if isinstance(score, (int, float)):
                            if score > 0.01:
                                bullish += 1
                            elif score < -0.01:
                                bearish += 1
                            else:
                                zero += 1
            except:
                pass

    total_scored = bullish + bearish + zero
    print(f"\nDirectional balance:")
    print(f"  Bullish: {bullish} ({bullish/total_scored*100:.1f}%)")
    print(f"  Bearish: {bearish} ({bearish/total_scored*100:.1f}%)")
    print(f"  Zero:    {zero} ({zero/total_scored*100:.1f}%)")
    print(f"\nDone!")


if __name__ == "__main__":
    main()
