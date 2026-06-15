"""
Baseline Predictions: OpenReasoning-Nemotron-7B BASE (no fine-tuning)

Same prompts, same system prompt, same vLLM config as the fine-tuned version.
The only difference: we load nvidia/OpenReasoning-Nemotron-7B directly from HuggingFace
instead of the merged fine-tuned model from the volume.

This gives us a clean comparison to determine whether fine-tuning added signal.

Usage:
    modal run scripts/predict_base_nemotron7b.py
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-nemotron7b-base-predict")

# Same image as the fine-tuned prediction run
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
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

# KEY DIFFERENCE: Load the base model from HuggingFace, not the fine-tuned merge
BASE_MODEL = "nvidia/OpenReasoning-Nemotron-7B"
EVAL_FILE = "/outputs/articles_eval.jsonl"
OUTPUT_FILE = "/outputs/predictions_openreasoning7b_BASE.jsonl"

# Identical system prompt to the fine-tuned run
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


@app.function(
    image=image,
    gpu="A10G",
    timeout=7200,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def run_predictions():
    """Run predictions using the BASE (non-fine-tuned) model via vLLM."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    # =========================================================
    # STEP 1: Load the BASE model with vLLM
    # =========================================================
    # vLLM's LLM class handles:
    #   - Model weight loading (downloads from HF, caches locally)
    #   - KV cache allocation (PagedAttention)
    #   - Continuous batching scheduler
    #   - CUDA graph capture for decode steps
    #
    # Key parameters:
    #   model: HuggingFace model ID or local path
    #   max_model_len: Maximum sequence length (input + output)
    #   gpu_memory_utilization: Fraction of GPU memory for KV cache
    #   dtype: Weight precision (bfloat16 for A10G)
    print("Loading BASE model with vLLM...")
    llm = LLM(
        model=BASE_MODEL,              # <-- This is the only difference from fine-tuned script
        trust_remote_code=True,
        max_model_len=2048,            # Limits KV cache allocation
        gpu_memory_utilization=0.90,   # Use 90% of GPU for model + KV cache
        dtype="bfloat16",              # A10G supports bfloat16
    )
    print("vLLM engine ready!")

    # =========================================================
    # STEP 2: Load tokenizer for chat template formatting
    # =========================================================
    # We load the tokenizer separately to apply the chat template
    # BEFORE passing to vLLM. This ensures the prompt format
    # exactly matches what the model expects.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # =========================================================
    # STEP 3: Load and prepare all prompts
    # =========================================================
    articles = []
    with open(EVAL_FILE) as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(article)
    print(f"Loaded {len(articles)} articles")

    # Format all prompts using the tokenizer's chat template
    # This produces the exact token sequence the model was trained on
    prompts = []
    article_meta = []
    for article in articles:
        text = clean_text(article.get("text", ""))
        source = article.get("source", "news")

        if len(text) < 30:
            article_meta.append({"idx": article["idx"], "skip": True,
                                 "date": article.get("date", ""), "title": article.get("title", ""),
                                 "source": source})
            continue

        source_inst = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        user_msg = f"{source_inst}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{text[:2500]}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ]

        # apply_chat_template converts messages into the model's expected format
        # add_generation_prompt=True appends the assistant turn marker
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompts.append(prompt)
        article_meta.append({"idx": article["idx"], "skip": False,
                             "date": article.get("date", ""), "title": article.get("title", ""),
                             "source": source})

    print(f"Prepared {len(prompts)} prompts")

    # =========================================================
    # STEP 4: Batch generate with vLLM
    # =========================================================
    # SamplingParams controls generation behavior:
    #   temperature: Lower = more deterministic (0.3 for structured output)
    #   top_p: Nucleus sampling threshold
    #   max_tokens: Maximum output length
    #
    # vLLM's .generate() processes ALL prompts as a single batch:
    #   - Schedules prefill and decode across all sequences
    #   - Uses PagedAttention to share KV cache memory efficiently
    #   - Continuous batching: finished sequences free memory for others
    #   - Returns results in the same order as input prompts
    sampling_params = SamplingParams(
        temperature=0.3,
        top_p=0.9,
        max_tokens=1500,
    )

    print("Generating (batch of all prompts simultaneously)...")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    total_time = time.time() - start_time
    print(f"Done in {total_time/60:.1f} minutes")
    print(f"Throughput: {sum(len(o.outputs[0].token_ids) for o in outputs)/total_time:.0f} output tok/s")

    # =========================================================
    # STEP 5: Parse outputs and save results
    # =========================================================
    success = 0
    errors = 0
    prompt_idx = 0

    with open(OUTPUT_FILE, "w") as out_f:
        for meta in article_meta:
            if meta.get("skip"):
                result = {"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "skipped", "reason": "too short"}
                out_f.write(json.dumps(result) + "\n")
                continue

            # vLLM returns outputs in order, one per input prompt
            raw = outputs[prompt_idx].outputs[0].text
            prompt_idx += 1

            try:
                # Extract JSON from model output
                start_j = raw.find("{")
                end_j = raw.rfind("}") + 1
                if start_j >= 0 and end_j > start_j:
                    json_str = re.sub(r',\s*([}\]])', r'\1', raw[start_j:end_j])
                    signal = json.loads(json_str)
                else:
                    raise ValueError("No JSON found")

                result = {"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "success", "signal": signal}
                success += 1
            except Exception as e:
                result = {"article_idx": meta["idx"], "date": meta["date"],
                          "title": meta["title"], "source": meta["source"],
                          "status": "error", "error": str(e)[:200]}
                errors += 1

            out_f.write(json.dumps(result) + "\n")

    output_vol.commit()
    print(f"\nResults: success={success} errors={errors}")
    print(f"Output: {OUTPUT_FILE}")

    return json.dumps({"success": success, "errors": errors, "total_minutes": round(total_time/60, 1)})


@app.local_entrypoint()
def main():
    result = run_predictions.remote()
    print(f"\nResult: {result}")
