"""
Sanity check: Run 5 articles through the base OpenReasoning-Nemotron-7B
and print the raw output to understand what it's generating.

Then test with a stronger JSON-forcing prompt.

Usage:
    modal run scripts/sanity_check_base_nemotron.py
"""

import modal
import json
import re
import os

app = modal.App("quantum-alpha-sanity-check-base")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "vllm>=0.12.0",
        "huggingface_hub",
        "transformers",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

BASE_MODEL = "nvidia/OpenReasoning-Nemotron-7B"
EVAL_FILE = "/outputs/articles_eval.jsonl"

# Original prompt (same as fine-tuned run)
SYSTEM_PROMPT_V1 = """You are a quantitative NLP signal generator for the quantum computing sector. For every piece of news or research, you must produce a signal vector that scores ALL companies in the quantum computing universe simultaneously.

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

Score ranges (MUST respect):
- Pure-play (IONQ, RGTI, QBTS, QUBT, QNT): [-2.0, +2.0]
- HON: [-0.3, +0.3]
- IBM: [-0.15, +0.15]
- MSFT, GOOGL, NVDA: always 0.0

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
        "MSFT": {"score": 0.0, "reasoning": "Inactive"},
        "GOOGL": {"score": 0.0, "reasoning": "Inactive"},
        "NVDA": {"score": 0.0, "reasoning": "Inactive"}
    },
    "event_type": "descriptive event category",
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "information_novelty": "high" | "medium" | "low",
    "technical_translation": "2-3 sentences.",
    "signal_rationale": "Why these scores?",
    "chain_of_thought": "Step-by-step reasoning."
}

Output ONLY the JSON object. No additional text, no markdown, no code blocks."""

# Stronger prompt that explicitly tells the model not to think/reason first
SYSTEM_PROMPT_V2 = """You are a JSON API that outputs quantitative trading signals for quantum computing stocks.

IMPORTANT: You must respond with ONLY a raw JSON object. No thinking, no explanation, no markdown code blocks. Just the JSON.

Tickers and score ranges:
- IONQ, RGTI, QBTS, QUBT, QNT: [-2.0, +2.0] (pure-play quantum)
- IBM: [-0.15, +0.15]
- HON: [-0.3, +0.3]
- MSFT, GOOGL, NVDA: always 0.0

Respond with exactly this JSON structure:
{"signal_vector": {"IONQ": {"score": 0.0, "reasoning": ""}, "RGTI": {"score": 0.0, "reasoning": ""}, "QBTS": {"score": 0.0, "reasoning": ""}, "QUBT": {"score": 0.0, "reasoning": ""}, "QNT": {"score": 0.0, "reasoning": ""}, "IBM": {"score": 0.0, "reasoning": ""}, "HON": {"score": 0.0, "reasoning": ""}, "MSFT": {"score": 0.0, "reasoning": "Inactive"}, "GOOGL": {"score": 0.0, "reasoning": "Inactive"}, "NVDA": {"score": 0.0, "reasoning": "Inactive"}}, "event_type": "", "time_horizon": "", "information_novelty": "", "signal_rationale": "", "chain_of_thought": ""}"""


def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.function(
    image=image,
    gpu="A10G",
    timeout=1800,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def sanity_check():
    """Test 5 articles with both prompt versions and show raw output."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    print("Loading base model...")
    llm = LLM(
        model=BASE_MODEL,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    print("Ready!\n")

    # Load first 5 articles
    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            if line.strip():
                articles.append(json.loads(line))

    sampling_params = SamplingParams(temperature=0.3, top_p=0.9, max_tokens=1500)

    # Test both prompts
    for prompt_name, system_prompt in [("V1 (original)", SYSTEM_PROMPT_V1), ("V2 (JSON-forcing)", SYSTEM_PROMPT_V2)]:
        print("=" * 70)
        print(f"TESTING PROMPT: {prompt_name}")
        print("=" * 70)

        prompts = []
        for article in articles:
            text = clean_text(article.get("text", ""))[:2000]
            user_msg = f"Analyze this quantum computing news and generate a signal vector:\n\n{text}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_msg},
            ]
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        outputs = llm.generate(prompts, sampling_params)

        success = 0
        for i, (article, output) in enumerate(zip(articles, outputs)):
            raw = output.outputs[0].text
            title = article.get("title", "")[:50]

            # Try to parse JSON
            has_json = False
            start_j = raw.find("{")
            end_j = raw.rfind("}") + 1
            if start_j >= 0 and end_j > start_j:
                try:
                    json_str = re.sub(r',\s*([}\]])', r'\1', raw[start_j:end_j])
                    parsed = json.loads(json_str)
                    if "signal_vector" in parsed:
                        has_json = True
                        success += 1
                except:
                    pass

            status = "JSON OK" if has_json else "NO JSON"
            print(f"\n  Article {i}: {title}")
            print(f"  Status: {status}")
            print(f"  Raw output (first 500 chars):")
            print(f"  {raw[:500]}")
            print()

        print(f"\n  RESULT: {success}/5 valid JSON outputs with {prompt_name}")
        print()

    return "done"


@app.local_entrypoint()
def main():
    sanity_check.remote()
