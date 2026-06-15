"""
Push V7d GRPO merged model to HuggingFace.

Usage:
    modal run scripts/push_to_hf.py
"""

import modal
import os

app = modal.App("quantum-alpha-hf-push")

image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .pip_install("huggingface_hub", "transformers", "safetensors")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

MERGED_MODEL = "/outputs/v7d-grpo-merged"
HF_REPO = "basilwong/quantum-alpha-openreasoning-7b-grpo"


@app.function(
    image=image, timeout=3600,
    volumes={"/outputs": output_vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def push():
    from huggingface_hub import HfApi
    import json

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)

    # Create repo
    try:
        api.create_repo(HF_REPO, private=True, exist_ok=True)
    except Exception as e:
        print(f"Repo creation: {e}")

    # Upload all files
    print(f"Uploading {MERGED_MODEL} to {HF_REPO}...")
    api.upload_folder(
        folder_path=MERGED_MODEL,
        repo_id=HF_REPO,
        token=token,
    )
    print("Done!")

    # Create model card
    card = """---
tags:
- quantum-computing
- financial-signal
- grpo
- reinforcement-learning
license: apache-2.0
base_model: nvidia/OpenReasoning-Nemotron-7B
---

# Quantum Alpha: OpenReasoning-Nemotron-7B GRPO

Fine-tuned for quantum computing sector signal generation using GRPO (Group Relative Policy Optimization) with actual market returns as the reward signal.

## Performance (out-of-sample, 421 evaluation articles)

| Horizon | IC | p-value | Direction Accuracy |
|---------|----|---------|--------------------|
| 1 day | +0.151 | 0.003 | - |
| 5 days | +0.157 | 0.006 | 58.6% |
| 10 days | +0.160 | 0.008 | - |
| 20 days | +0.159 | 0.024 | - |

## Training

- Base: nvidia/OpenReasoning-Nemotron-7B
- Stage 1: SFT on 881 quantum computing articles (V4 training data)
- Stage 2: GRPO with reward function based on actual 5-day cumulative abnormal returns
- 184 training articles with return data, 4 generations per prompt
- Proper train/test split (no data leakage)
"""
    api.upload_file(
        path_or_fileobj=card.encode(),
        path_in_repo="README.md",
        repo_id=HF_REPO,
        token=token,
    )
    print("Model card uploaded!")
    return "success"


@app.local_entrypoint()
def main():
    result = push.remote()
    print(f"Result: {result}")
