# Scripts: Pipeline Runbook

All commands should be run from the project root (`quantum-alpha-intelligence/`).

## Prerequisites

```bash
# Install Python dependencies
pip install arxiv feedparser openai

# Install Modal CLI (for fine-tuning)
pip install modal
modal token set --token-id ak-O0yCWJtr9WXovf2nFqg9mI --token-secret as-frqox6GMOfq88Mdq7jk8sJ
```

## Step 1: Collect Raw Articles

Collects quantum computing news and papers from the last 2 years.

```bash
# Full collection (arXiv + Google News RSS + samples) — takes ~5 min
python scripts/collect_historical_articles.py --output data/raw/articles.jsonl

# Skip arXiv (faster, news only)
python scripts/collect_historical_articles.py --output data/raw/articles.jsonl --skip-arxiv

# Samples only (instant, for testing)
python scripts/collect_articles.py --output data/raw/articles.jsonl --sources samples
```

Expected output: `data/raw/articles.jsonl` with 200-600+ articles.

## Step 2: Generate Training Data (Teacher Model)

Uses `qwen3-max` (Alibaba Cloud Model Studio, Singapore free tier) to label each article.

```bash
# Test with 5 articles first (verify quality)
python scripts/generate_training_data.py \
    --input data/raw/articles.jsonl \
    --output data/training/quantum_alpha_train.jsonl \
    --limit 5

# Generate 200 labeled examples (recommended for first training run)
python scripts/generate_training_data.py \
    --input data/raw/articles.jsonl \
    --output data/training/quantum_alpha_train.jsonl \
    --limit 200
```

API configuration is hardcoded in the script (Singapore endpoint, qwen3-max model).
To override, set the environment variable:
```bash
export DASHSCOPE_API_KEY="your-key-here"
```

Expected output: `data/training/quantum_alpha_train.jsonl` with instruction-tuning pairs.

## Step 3: Upload Training Data to Modal

```bash
modal volume put quantum-alpha-outputs data/training/quantum_alpha_train.jsonl quantum_alpha_train.jsonl
```

## Step 4: Test Modal Environment

Verifies GPU access, model loading, and basic inference on Modal.

```bash
modal run scripts/modal_finetune.py --test
```

Expected: Prints GPU info, loads Qwen3-8B, runs a test inference.
Cost: ~$0.21 (5 min on A100).

## Step 5: Run Fine-Tuning

```bash
modal run scripts/modal_finetune.py
```

Configuration (in `scripts/modal_finetune.py`):
- Base model: `unsloth/Qwen3-8B-Instruct`
- Method: QLoRA (4-bit quantization)
- LoRA rank: 64
- Epochs: 4
- Checkpoints: Saved per epoch
- Output: Auto-pushed to `basilwong/quantum-alpha-qwen3-8b` on HF Hub

Cost estimate: ~$2.50-5.00 (30-60 min on A100).

## Step 6: Verify Model on HF Hub

After training, check: https://huggingface.co/basilwong/quantum-alpha-qwen3-8b

## Scripts Reference

| Script | Purpose | Dependencies |
|--------|---------|-------------|
| `collect_articles.py` | Basic article collection (samples + arXiv + RSS) | arxiv, feedparser |
| `collect_historical_articles.py` | Full 2-year historical collection | arxiv, feedparser |
| `generate_training_data.py` | Teacher model labeling pipeline | openai |
| `modal_finetune.py` | QLoRA fine-tuning on Modal GPUs | modal, unsloth |

## Cost Summary

| Step | Resource | Estimated Cost |
|------|----------|---------------|
| Data generation (200 articles) | Qwen Cloud free tier | ~200K tokens (of 1M free) |
| Modal env test | Modal A100 | ~$0.21 |
| Fine-tuning (200 examples, 4 epochs) | Modal A100 | ~$2.50 |
| **Total** | | **~$2.71 of $280 Modal + free Qwen tokens** |
