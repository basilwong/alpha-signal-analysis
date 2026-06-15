"""
Re-run the 33 failed predictions with improved JSON repair.
Loads the raw vLLM outputs from a re-generation of just the failed articles.

But first: let's just try to repair the existing outputs without re-running inference.
The raw text IS in the error entries (we just need to extract it from the vLLM output).

Actually, we don't have the raw text saved. So we need to re-run inference on just
the 33 failed articles with better JSON parsing.

Usage:
    modal run scripts/retry_failures_nemotron7b.py
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-nemotron7b-retry")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "vllm>=0.12.0",
        "datasets",
        "huggingface_hub",
        "transformers",
        "accelerate",
        "peft",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

MERGED_OUTPUT = "/outputs/openreasoning-7b-merged"
EVAL_FILE = "/outputs/articles_eval.jsonl"
PREDICTIONS_FILE = "/outputs/predictions_openreasoning7b_v4.jsonl"

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

SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess novelty and likely decay speed.",
    "arxiv": "This is an academic paper abstract. Most are incremental. Only significant scores for genuine breakthroughs.",
    "sec_filing": "This is a regulatory filing. High reliability.",
    "press_release": "Company press release. Be skeptical.",
    "social_media": "Social media post. High noise, low reliability.",
    "earnings_call": "Earnings call. Forward guidance matters most.",
}


def clean_text(text):
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&\w+;', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def robust_json_parse(raw: str) -> dict:
    """Aggressively repair and parse JSON from model output."""
    # Strip thinking blocks
    if "<think>" in raw:
        parts = raw.split("</think>")
        raw = parts[-1].strip() if len(parts) > 1 else raw

    # Find JSON boundaries
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start < 0 or end <= start:
        raise ValueError("No JSON object found")

    json_str = raw[start:end]

    # Attempt 1: direct parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Attempt 2: remove trailing commas
    fixed = re.sub(r',\s*([}\]])', r'\1', json_str)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 3: fix unescaped newlines inside string values
    # Replace literal newlines between quotes with \\n
    fixed = re.sub(r'(?<=": ")(.*?)(?="[,\s*}])', 
                   lambda m: m.group(0).replace('\n', '\\n').replace('\r', ''),
                   fixed, flags=re.DOTALL)
    try:
        return json.loads(fixed)
    except json.JSONDecodeError:
        pass

    # Attempt 4: extract just the signal_vector portion (minimum viable output)
    sv_match = re.search(r'"signal_vector"\s*:\s*\{', fixed)
    if sv_match:
        # Find the matching closing brace for signal_vector
        depth = 0
        sv_start = sv_match.start()
        brace_start = fixed.index('{', sv_match.end() - 1)
        for i in range(brace_start, len(fixed)):
            if fixed[i] == '{':
                depth += 1
            elif fixed[i] == '}':
                depth -= 1
                if depth == 0:
                    sv_end = i + 1
                    break
        else:
            raise ValueError("Could not find signal_vector closing brace")

        sv_json = fixed[brace_start:sv_end]
        # Remove trailing commas in the extracted portion
        sv_json = re.sub(r',\s*([}\]])', r'\1', sv_json)
        try:
            sv = json.loads(sv_json)
            return {"signal_vector": sv, "event_type": "unknown", "parse_note": "partial_recovery"}
        except json.JSONDecodeError:
            pass

    # Attempt 5: line-by-line repair
    lines = fixed.split('\n')
    repaired_lines = []
    for line in lines:
        # Fix unescaped quotes within string values (common in chain_of_thought)
        # Pattern: a line that has content but breaks JSON
        repaired_lines.append(line)

    fixed = '\n'.join(repaired_lines)
    # Try removing the chain_of_thought field entirely if it's causing issues
    fixed_no_cot = re.sub(
        r'"chain_of_thought"\s*:\s*"[^"]*(?:"[^"]*)*"',
        '"chain_of_thought": "See signal_rationale"',
        fixed
    )
    try:
        return json.loads(fixed_no_cot)
    except json.JSONDecodeError:
        pass

    raise ValueError(f"All repair attempts failed")


@app.function(
    image=image,
    gpu="A10G",
    timeout=3600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def retry_failures():
    """Re-run just the failed articles with better JSON parsing."""
    import torch
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # Load existing predictions to find failures
    failed_indices = set()
    existing_results = {}
    with open(PREDICTIONS_FILE) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                existing_results[r.get("article_idx")] = r
                if r.get("status") == "error":
                    failed_indices.add(r.get("article_idx"))

    print(f"Found {len(failed_indices)} failed articles to retry")

    # Load evaluation articles
    articles = {}
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                if i in failed_indices:
                    articles[i] = article

    print(f"Loaded {len(articles)} articles to retry")

    # Load vLLM
    print("Loading vLLM...")
    llm = LLM(
        model=MERGED_OUTPUT,
        trust_remote_code=True,
        max_model_len=2048,
        gpu_memory_utilization=0.90,
        dtype="bfloat16",
    )

    tokenizer = AutoTokenizer.from_pretrained(MERGED_OUTPUT, trust_remote_code=True)
    print("Ready!")

    # Prepare prompts for failed articles
    prompts = []
    meta_list = []
    for idx in sorted(failed_indices):
        article = articles[idx]
        text = clean_text(article.get("text", ""))
        source = article.get("source", "news")
        source_inst = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        user_msg = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text[:2500]}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
        meta_list.append({"idx": idx, "date": article.get("date", ""),
                          "title": article.get("title", ""), "source": source})

    # Generate with slightly different params (lower temp for more deterministic JSON)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.95, max_tokens=1500)
    print(f"Generating {len(prompts)} predictions...")
    outputs = llm.generate(prompts, sampling_params)

    # Parse with robust repair
    recovered = 0
    still_failed = 0
    raw_failures = []

    for i, (output, meta) in enumerate(zip(outputs, meta_list)):
        raw = output.outputs[0].text
        try:
            signal = robust_json_parse(raw)
            existing_results[meta["idx"]] = {
                "article_idx": meta["idx"], "date": meta["date"],
                "title": meta["title"], "source": meta["source"],
                "status": "success", "signal": signal,
            }
            recovered += 1
        except Exception as e:
            existing_results[meta["idx"]] = {
                "article_idx": meta["idx"], "date": meta["date"],
                "title": meta["title"], "source": meta["source"],
                "status": "error", "error": str(e)[:200],
                "raw_output_snippet": raw[:500],
            }
            still_failed += 1
            raw_failures.append({"idx": meta["idx"], "title": meta["title"], "raw": raw[:1000]})

    print(f"\nRetry results: recovered={recovered}, still_failed={still_failed}")

    # Write updated predictions file
    with open(PREDICTIONS_FILE, "w") as f:
        for idx in sorted(existing_results.keys()):
            f.write(json.dumps(existing_results[idx]) + "\n")

    output_vol.commit()

    # Print sample raw outputs for remaining failures
    if raw_failures:
        print("\nRemaining failures (raw output samples):")
        for rf in raw_failures[:3]:
            print(f"\n  Article {rf['idx']}: {rf['title'][:50]}")
            print(f"  Raw: {rf['raw'][:300]}")

    return json.dumps({"recovered": recovered, "still_failed": still_failed})


@app.local_entrypoint()
def main():
    result = retry_failures.remote()
    print(f"\nResult: {result}")
