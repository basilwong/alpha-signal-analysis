"""
Parallel Batch Predictions: Data-parallel inference across multiple A100s.

Uses Modal's .map() to distribute articles across multiple GPU containers,
each running the same frozen model independently. Avoids timeout issues
by keeping per-container workload under 1 hour.

Usage:
    # Upload eval articles to volume:
    modal volume put quantum-alpha-outputs data/raw/articles_eval.jsonl articles_eval.jsonl --force

    # Run parallel predictions:
    modal run scripts/generate_predictions_parallel.py

    # Download results:
    modal volume get quantum-alpha-outputs predictions_v3_final.jsonl data/eval/predictions_v3_final.jsonl
"""

import modal
import json
import re
import time
import os

app = modal.App("quantum-alpha-predictions-parallel")

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

Key domain knowledge:
- Trapped-ion breakthroughs: bullish IONQ/HON, bearish RGTI/IBM/GOOGL
- Superconducting breakthroughs: bullish RGTI/IBM/GOOGL, bearish IONQ/HON
- Error correction advances: benefit ALL gate-based approaches
- Government funding: broadly bullish for entire sector
- Scale by revenue exposure: GOOGL/MSFT max +/-0.05, HON max +/-0.3, IBM max +/-0.15
- If the content is NOT related to quantum computing, assign all scores to 0.0

Output a valid JSON object:
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
    "event_type": str,
    "time_horizon": "intraday" | "2-5 days" | "1-2 weeks" | "1+ month",
    "signal_decay": "fast" | "medium" | "slow",
    "information_novelty": "high" | "medium" | "low",
    "technical_translation": "2-3 sentences.",
    "signal_rationale": "Why these scores?"
}

Output ONLY the JSON object. No markdown, no code blocks, no extra text."""

SOURCE_INSTRUCTIONS = {
    "news": "This is a financial news article. Assess novelty and likely decay speed.",
    "arxiv": "This is an academic paper abstract. Most are incremental. Only significant scores for genuine breakthroughs. Slow decay.",
    "sec_filing": "This is a regulatory filing. High reliability. Fast decay.",
    "press_release": "Company press release. Be skeptical.",
    "social_media": "Social media post. High noise, low reliability.",
    "earnings_call": "Earnings call. Forward guidance matters most.",
}


def clean_article_text(text: str) -> str:
    """Strip HTML tags and URLs from article text."""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&#\d+;', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@app.cls(
    image=predict_image,
    gpu="A100",
    timeout=7200,  # 2 hours per container
    volumes={"/root/.cache/huggingface": hf_cache_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    max_containers=8,  # Max 8 containers running simultaneously
)
class Predictor:
    @modal.enter()
    def load_model(self):
        """Load model once when container starts."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"Loading model: {MODEL_ID}")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded on {torch.cuda.get_device_name(0)}")

    @modal.method()
    def predict_article(self, article_json: str) -> str:
        """Predict signal vector for a single article. Returns JSON string."""
        import torch

        article = json.loads(article_json)
        text = article.get("text", "")
        source = article.get("source", "news")
        idx = article.get("idx", -1)
        title = article.get("title", "")
        date = article.get("date", "")

        start_time = time.time()

        # Clean text
        cleaned_text = clean_article_text(text)

        if len(cleaned_text.strip()) < 30:
            return json.dumps({
                "article_idx": idx, "date": date, "title": title,
                "source": source, "status": "skipped",
                "reason": "text too short after cleaning",
                "time_seconds": 0,
            })

        source_instruction = SOURCE_INSTRUCTIONS.get(source, SOURCE_INSTRUCTIONS["news"])
        user_message = f"{source_instruction}\n\nAnalyze the following content and generate a cross-sectional signal vector:\n\n{cleaned_text}"

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ]

        try:
            inputs = self.tokenizer.apply_chat_template(
                messages, return_tensors="pt", add_generation_prompt=True,
                return_dict=True, enable_thinking=False
            ).to(self.model.device)

            input_length = inputs["input_ids"].shape[-1]

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=1024,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                )

            generated_ids = outputs[0][input_length:]
            raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Strip thinking tags
            if "<think>" in raw_output:
                parts = raw_output.split("</think>")
                if len(parts) > 1:
                    raw_output = parts[-1].strip()

            # Parse JSON
            start = raw_output.find("{")
            end = raw_output.rfind("}") + 1
            if start != -1 and end > start:
                signal = json.loads(raw_output[start:end])
            else:
                signal = json.loads(raw_output)

            elapsed = time.time() - start_time
            return json.dumps({
                "article_idx": idx, "date": date, "title": title,
                "source": source, "status": "success", "signal": signal,
                "input_tokens": input_length,
                "output_tokens": len(generated_ids),
                "time_seconds": round(elapsed, 2),
            })

        except Exception as e:
            elapsed = time.time() - start_time
            return json.dumps({
                "article_idx": idx, "date": date, "title": title,
                "source": source, "status": "error",
                "error": str(e)[:300],
                "time_seconds": round(elapsed, 2),
            })


@app.local_entrypoint()
def main():
    """Load eval articles and distribute across multiple GPU containers."""

    # Load evaluation articles
    eval_path = "data/raw/articles_eval.jsonl"
    articles = []
    with open(eval_path, "r") as f:
        for i, line in enumerate(f):
            if line.strip():
                article = json.loads(line)
                article["idx"] = i
                articles.append(json.dumps(article))

    print(f"Loaded {len(articles)} evaluation articles")
    print(f"Distributing across up to 4 parallel GPU containers...")
    print(f"Estimated time: ~15-20 minutes (vs 3+ hours sequential)")

    # Use .map() for parallel processing with error handling
    predictor = Predictor()
    results = []

    for result_json in predictor.predict_article.map(articles, return_exceptions=True):
        if isinstance(result_json, Exception):
            results.append({"status": "error", "error": str(result_json)[:200]})
        else:
            try:
                result = json.loads(result_json)
                results.append(result)
            except (json.JSONDecodeError, TypeError) as e:
                results.append({"status": "error", "error": f"JSON decode: {str(e)[:100]}"})
        if len(results) % 50 == 0:
            success = sum(1 for r in results if r.get("status") == "success")
            errors = sum(1 for r in results if r.get("status") == "error")
            print(f"  [{len(results)}/{len(articles)}] success={success} errors={errors}")

    # Summary
    success = sum(1 for r in results if r["status"] == "success")
    errors = sum(1 for r in results if r["status"] == "error")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    print(f"\n{'='*60}")
    print(f"COMPLETE: {len(results)} processed")
    print(f"  Success: {success}")
    print(f"  Errors: {errors}")
    print(f"  Skipped: {skipped}")

    # Save results
    output_path = "data/eval/predictions_v3_final.jsonl"
    os.makedirs("data/eval", exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"  Saved to: {output_path}")

    # Also compute timing stats
    times = [r.get("time_seconds", 0) for r in results if r["status"] == "success"]
    if times:
        print(f"\nTiming stats:")
        print(f"  Mean: {sum(times)/len(times):.1f}s")
        print(f"  Min: {min(times):.1f}s")
        print(f"  Max: {max(times):.1f}s")
        print(f"  Total GPU-time: {sum(times)/60:.1f} min")
