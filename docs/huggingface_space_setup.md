# Hugging Face Space Setup Guide

This document covers how to create, configure, and deploy the Alpha Signal Analysis Platform as a Hugging Face Space under the `build-small-hackathon` organization.

## Overview

Hugging Face Spaces host Gradio apps as git repositories. When you push code to the Space repo, Hugging Face automatically builds and deploys the app. Our project uses `gradio.Server` (which is built on FastAPI), so we select the Gradio SDK.

## Creating the Space

### Option 1: Via the Hugging Face Web UI

1. Go to https://huggingface.co/spaces
2. Click **Create new Space**
3. Set the **Owner** to `build-small-hackathon` (the hackathon org you joined)
4. Set the **Space name** (e.g., `hackathon-test` for now)
5. Select **Gradio** as the SDK
6. Set visibility to **Public**
7. Click **Create Space**

### Option 2: Via the huggingface_hub Python Library

```python
from huggingface_hub import create_repo

create_repo(
    "build-small-hackathon/hackathon-test",
    repo_type="space",
    space_sdk="gradio",
    visibility="public"
)
```

### Option 3: Via the Hugging Face CLI

```bash
# Login first
huggingface-cli login

# Create the Space
huggingface-cli repo create hackathon-test --type space --organization build-small-hackathon
```

## Renaming the Space Later

Yes, you can rename a Space after creation. Use the `move_repo` function from the `huggingface_hub` library:

```python
from huggingface_hub import move_repo

move_repo(
    from_id="build-small-hackathon/hackathon-test",
    to_id="build-small-hackathon/alpha-signal-analysis"
)
```

Or via CLI:

```bash
huggingface-cli repo move build-small-hackathon/hackathon-test build-small-hackathon/alpha-signal-analysis --type space
```

**Note:** After renaming, the old URL will redirect to the new one for a period of time, but you should update any references.

## Space Configuration (README.md YAML Header)

The Space's behavior is configured via a YAML block at the top of the `README.md` in the Space repo:

```yaml
---
title: Alpha Signal Analysis
emoji: ⚛
colorFrom: indigo
colorTo: cyan
sdk: gradio
sdk_version: 5.0.0
app_file: app.py
pinned: false
models:
- basilwong/quantum-alpha-qwen3-8b
---
```

## Space File Structure

The Space repo should contain only the production deployment files:

```
hackathon-test/
├── README.md              # YAML config + description
├── app.py                 # Main entry point (Gradio Server)
├── requirements.txt       # Production dependencies only
├── src/
│   ├── config.py          # Ticker universe and settings
│   ├── signals.py         # Signal processing logic
│   └── inference.py       # Model inference wrapper
└── frontend/
    ├── index.html         # Custom trading terminal
    ├── css/dashboard.css  # Styles
    └── js/app.js          # Frontend logic
```

## The app.py Entry Point

The `app_file` in the YAML config points to the main script. For our project:

```python
# app.py (root of Space repo)
import sys
sys.path.insert(0, ".")

from src.api.app import app

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
```

## Environment Variables and Secrets

Set these in the Space Settings (not in code):

| Variable | Type | Purpose |
|----------|------|---------|
| `HF_TOKEN` | Secret | Access to gated models on HF Hub |
| `MODAL_TOKEN_ID` | Secret | Modal serverless GPU access |
| `MODAL_TOKEN_SECRET` | Secret | Modal authentication |

## Hardware Selection

For the demo to run smoothly with model inference:

| Option | Cost | Use Case |
|--------|------|----------|
| CPU Basic (free) | $0/hr | Frontend demo only (inference via Modal) |
| ZeroGPU | Free | On-demand GPU for inference within the Space |
| Nvidia T4 small | $0.40/hr | Dedicated GPU for real-time inference |

**Recommended approach:** Use the free CPU tier for the Space and call out to Modal for GPU inference. This keeps hosting costs at zero while still delivering fast model responses.

## Syncing GitHub Repo with HF Space

You can set up a GitHub Action to automatically sync pushes to your GitHub repo with the HF Space:

```yaml
# .github/workflows/sync-to-hf.yml
name: Sync to Hugging Face Space
on:
  push:
    branches: [main]

jobs:
  sync:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
        with:
          fetch-depth: 0
      - name: Push to HF Space
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          git remote add hf https://huggingface.co/spaces/build-small-hackathon/hackathon-test
          git push hf main --force
```

Add your `HF_TOKEN` as a GitHub repository secret for this to work.

## Testing the Deployment

After pushing to the Space, Hugging Face will build and deploy automatically. Monitor the build logs in the Space's "Logs" tab. The app will be accessible at:

```
https://huggingface.co/spaces/build-small-hackathon/hackathon-test
```

Or via the direct embed URL:

```
https://build-small-hackathon-hackathon-test.hf.space
```
