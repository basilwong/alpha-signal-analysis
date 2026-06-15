"""
Quick inference test to debug the V4 fine-tuned model output format.
Run: modal run scripts/test_inference_v4.py
"""
import modal
import json

app = modal.App("quantum-alpha-test-inference")

predict_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "transformers",
        "accelerate",
        "peft",
        "bitsandbytes",
        "huggingface_hub",
        "sentencepiece",
        "protobuf",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

MODEL_ID = "basilwong/quantum-alpha-qwen3-8b"

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


@app.function(
    image=predict_image,
    gpu="A100",
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_inference():
    """Test inference with the fine-tuned model to debug output format."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    model.eval()
    print(f"Model loaded on {torch.cuda.get_device_name(0)}")

    test_article = "IonQ announces breakthrough: 50 algorithmic qubits achieved on latest trapped-ion processor, setting new industry record for quantum volume."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"This is a financial news article. Assess novelty and likely decay speed.\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{test_article}"},
    ]

    # Test with enable_thinking=False
    print("\n=== Test 1: enable_thinking=False ===")
    inputs = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=True, enable_thinking=False
    ).to(model.device)
    input_length = inputs["input_ids"].shape[-1]
    print(f"Input tokens: {input_length}")

    with torch.no_grad():
        outputs = model.generate(
            **inputs, max_new_tokens=1024, temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_ids = outputs[0][input_length:]
    raw_output = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Raw output (first 2000 chars):\n{raw_output[:2000]}")
    print(f"\n--- Full raw output length: {len(raw_output)} chars ---")

    # Also test with enable_thinking=True to compare
    print("\n=== Test 2: enable_thinking=True ===")
    inputs2 = tokenizer.apply_chat_template(
        messages, return_tensors="pt", add_generation_prompt=True,
        return_dict=True, enable_thinking=True
    ).to(model.device)
    input_length2 = inputs2["input_ids"].shape[-1]
    print(f"Input tokens: {input_length2}")

    with torch.no_grad():
        outputs2 = model.generate(
            **inputs2, max_new_tokens=2048, temperature=0.3, do_sample=True,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    generated_ids2 = outputs2[0][input_length2:]
    raw_output2 = tokenizer.decode(generated_ids2, skip_special_tokens=True)
    print(f"Raw output (first 2000 chars):\n{raw_output2[:2000]}")

    # Try to decode without skip_special_tokens to see all tokens
    print("\n=== Test 3: decode with special tokens visible ===")
    raw_with_special = tokenizer.decode(generated_ids, skip_special_tokens=False)
    print(f"With special tokens (first 500 chars):\n{raw_with_special[:500]}")

    return {"raw_output_length": len(raw_output), "raw_output_start": raw_output[:200]}


@app.local_entrypoint()
def main():
    result = test_inference.remote()
    print(f"\nResult: {result}")
