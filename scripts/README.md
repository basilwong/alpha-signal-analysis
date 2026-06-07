# Scripts Directory

This directory contains the infrastructure scripts for training, data generation, and deployment.

## Setup

### 1. Install Modal CLI

```bash
pip install modal
modal setup  # This will open a browser for authentication
```

### 2. Set Up Modal Secrets

The fine-tuning script needs your Hugging Face token to push the trained model:

```bash
modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

### 3. Set Up Qwen Cloud API Key

The data generation script uses the Qwen Cloud API (DashScope):

```bash
export DASHSCOPE_API_KEY="your-dashscope-api-key"
```

## Pipeline Workflow

### Step 1: Collect Raw Articles

```bash
# Collect from all sources (samples + arXiv + RSS)
python scripts/collect_articles.py --output data/raw/articles.jsonl

# Or just use the built-in samples for testing
python scripts/collect_articles.py --output data/raw/articles.jsonl --sources samples
```

### Step 2: Generate Training Data (Teacher Model)

```bash
# Generate labels for all articles (uses Qwen3.7-Max as teacher)
python scripts/generate_training_data.py \
    --input data/raw/articles.jsonl \
    --output data/training/quantum_alpha_train.jsonl

# Test with just 5 articles first
python scripts/generate_training_data.py \
    --input data/raw/articles.jsonl \
    --output data/training/quantum_alpha_train.jsonl \
    --limit 5
```

### Step 3: Upload Training Data to Modal Volume

```bash
modal volume put quantum-alpha-outputs data/training/quantum_alpha_train.jsonl quantum_alpha_train.jsonl
```

### Step 4: Test the Training Environment

```bash
modal run scripts/modal_finetune.py --test
```

### Step 5: Run Full Fine-Tuning

```bash
modal run scripts/modal_finetune.py
```

### Step 6: Verify the Model on HF Hub

After training completes, the model will be available at:
https://huggingface.co/basilwong/quantum-alpha-qwen3-8b

## Cost Estimates

| Operation | GPU | Estimated Time | Estimated Cost |
|-----------|-----|---------------|----------------|
| Environment test | A100 | 5 min | ~$0.21 |
| Fine-tuning (500 examples, 4 epochs) | A100 | 30-60 min | ~$1.25-2.50 |
| Fine-tuning (2000 examples, 4 epochs) | A100 | 2-3 hours | ~$5.00-7.50 |

Total estimated cost for the full pipeline: **$5-10** out of your $280 Modal credits.
