"""
Test the fine-tuned model by running inference on Modal.

Since the fine-tuned model requires a GPU to run, we test it on Modal
using the same infrastructure we trained on.

Usage:
    modal run scripts/test_finetuned_model.py
"""

import modal
import json

app = modal.App("alpha-signal-test-inference")

# Reuse the same image from training
test_image = (
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

hf_cache_vol = modal.Volume.from_name("hf-cache-alpha-signal", create_if_missing=True)

# Test articles
TEST_ARTICLES = [
    {
        "text": "IonQ announced a new partnership with Hyundai Motor Company to develop quantum machine learning algorithms for battery chemistry optimization. The multi-year deal is valued at $12 million and represents IonQ's largest commercial contract to date. IonQ shares rose 5% on the news.",
        "source": "news",
        "expected_sentiment": "bullish",
    },
    {
        "text": "Rigetti Computing announced it will delay the release of its 84-qubit Ankaa-3 processor by six months due to manufacturing yield issues. The company said it needs additional time to improve two-qubit gate fidelities to meet its target of 99.5%. Rigetti stock fell 12% in after-hours trading.",
        "source": "news",
        "expected_sentiment": "bearish",
    },
    {
        "text": "Title: Achieving 1000 logical qubits with a modular trapped-ion architecture\n\nAbstract: We present a theoretical framework for scaling trapped-ion quantum computers to 1000 logical qubits using a modular architecture with photonic interconnects. Our analysis shows that with current ion trap technology and demonstrated gate fidelities, a system of 50 interconnected modules, each containing 200 physical qubits, can support 1000 logical qubits with a logical error rate below 10^-10 per operation. We estimate the total system cost at approximately $500 million.",
        "source": "arxiv",
        "expected_sentiment": "bullish",
    },
]


@app.function(
    image=test_image,
    gpu="A100",
    timeout=600,
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def test_inference():
    """Run inference on test articles using the fine-tuned model."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL_ID = "basilwong/quantum-alpha-qwen3-8b"

    print("=" * 60)
    print("QUANTUM ALPHA: Testing Fine-Tuned Model Inference")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Test articles: {len(TEST_ARTICLES)}")
    print("=" * 60)

    # Load model
    print("\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded!")

    SYSTEM_PROMPT = """You are an expert quantum computing financial analyst with deep knowledge of both quantum physics and capital markets. Your role is to analyze news articles, press releases, academic papers, and regulatory filings related to the quantum computing sector and produce structured intelligence reports.

You must output a valid JSON object with the following fields:

{
    "sentiment": "strongly_bearish" | "bearish" | "neutral" | "bullish" | "strongly_bullish",
    "confidence": 0.0 to 1.0,
    "event_type": one of ["physical_qubit_milestone", "logical_qubit_breakthrough", "error_correction_advance", "quantum_volume_increase", "government_funding", "commercial_partnership", "revenue_earnings", "executive_change", "patent_grant", "academic_publication", "product_launch", "competitive_development", "regulatory_filing", "analyst_rating_change"],
    "affected_tickers": ["IONQ", "RGTI", etc.],
    "urgency": "low" | "medium" | "high",
    "technical_translation": "A 2-3 sentence explanation of what this means commercially, written for an investor who does not have a physics background.",
    "key_facts": ["fact1", "fact2", "fact3"],
    "competitive_context": "How does this development position the company relative to competitors?"
}

Output ONLY the JSON object. No additional text, no markdown formatting, no code blocks."""

    results = []

    for i, article in enumerate(TEST_ARTICLES):
        print(f"\n{'─' * 60}")
        print(f"Test {i+1}/{len(TEST_ARTICLES)}: {article['text'][:80]}...")
        print(f"Expected sentiment: {article['expected_sentiment']}")

        user_message = f"Analyze the following {article['source']} content about the quantum computing sector and provide a structured intelligence report:\n\n{article['text']}"

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

        # Parse JSON
        try:
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            if start != -1 and end > start:
                signal = json.loads(raw_output[start:end])
            else:
                signal = json.loads(raw_output)

            sentiment_match = signal.get("sentiment", "").replace("strongly_", "") == article["expected_sentiment"].replace("strongly_", "")
            print(f"  Sentiment: {signal.get('sentiment')} {'✓' if sentiment_match else '✗'}")
            print(f"  Event type: {signal.get('event_type')}")
            print(f"  Tickers: {signal.get('affected_tickers')}")
            print(f"  Confidence: {signal.get('confidence')}")
            print(f"  Translation: {signal.get('technical_translation', '')[:100]}...")

            results.append({
                "article_idx": i,
                "expected": article["expected_sentiment"],
                "predicted": signal.get("sentiment"),
                "match": sentiment_match,
                "signal": signal,
            })

        except (json.JSONDecodeError, Exception) as e:
            print(f"  ERROR: Failed to parse output: {e}")
            print(f"  Raw output: {raw_output[:200]}...")
            results.append({
                "article_idx": i,
                "expected": article["expected_sentiment"],
                "predicted": "PARSE_ERROR",
                "match": False,
                "raw_output": raw_output[:500],
            })

    # Summary
    print(f"\n{'=' * 60}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 60}")
    correct = sum(1 for r in results if r["match"])
    print(f"Sentiment accuracy: {correct}/{len(results)} ({100*correct/len(results):.0f}%)")
    parse_errors = sum(1 for r in results if r["predicted"] == "PARSE_ERROR")
    print(f"Parse errors: {parse_errors}/{len(results)}")

    return results


@app.local_entrypoint()
def main():
    results = test_inference.remote()
    print("\nFull results:")
    for r in results:
        print(f"  Article {r['article_idx']}: expected={r['expected']}, predicted={r['predicted']}, match={r['match']}")
