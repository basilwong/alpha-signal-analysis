"""
Merge LoRA adapter into base model and save as 16-bit for vLLM serving.

Usage:
    modal run scripts/merge_and_export.py
"""

import modal
import json

app = modal.App("quantum-alpha-minicpm-merge")

finetune_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git")
    .pip_install("torch", "torchvision", "torchaudio")
    .pip_install(
        "datasets",
        "huggingface_hub",
        "trl",
        "transformers==4.57.3",
        "accelerate",
        "peft",
        "bitsandbytes",
        "sentencepiece",
        "protobuf",
    )
    .run_commands(
        "pip install --no-deps unsloth unsloth-zoo",
        "pip install --no-deps cut-cross-entropy",
    )
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

hf_cache_vol = modal.Volume.from_name("hf-cache-quantum-alpha", create_if_missing=True)
output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

BASE_MODEL = "openbmb/MiniCPM-2B-sft-bf16-llama-format"
ADAPTER_PATH = "/outputs/quantum-alpha-minicpm-2b/checkpoint-82"
MERGED_OUTPUT = "/outputs/minicpm-2b-merged"


@app.function(
    image=finetune_image,
    gpu="A100",
    timeout=1800,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def merge_and_save():
    """Load base + adapter, merge, save 16-bit model to volume."""
    import torch
    import os
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    print("Step 1: Loading base model (standard HF, no Unsloth)...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    print(f"  Base model loaded: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    print(f"\nStep 2: Loading LoRA adapter from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    print("  Adapter loaded.")

    print("\nStep 3: Merging adapter into base model...")
    model = model.merge_and_unload()
    print(f"  Merged model: {sum(p.numel() for p in model.parameters())/1e9:.2f}B params")

    print(f"\nStep 4: Saving merged model to {MERGED_OUTPUT}...")
    os.makedirs(MERGED_OUTPUT, exist_ok=True)
    model.save_pretrained(MERGED_OUTPUT)
    tokenizer.save_pretrained(MERGED_OUTPUT)

    # Verify files exist
    files = os.listdir(MERGED_OUTPUT)
    total_size = sum(os.path.getsize(os.path.join(MERGED_OUTPUT, f)) for f in files)
    print(f"  Saved {len(files)} files, total size: {total_size/1e9:.2f} GB")
    for f in sorted(files):
        size = os.path.getsize(os.path.join(MERGED_OUTPUT, f))
        print(f"    {f}: {size/1e6:.1f} MB")

    print("\nStep 5: Committing to volume...")
    output_vol.commit()
    print("  Volume committed!")

    # Quick sanity check: load and generate
    print("\nStep 6: Sanity check inference...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    test_tok = AutoTokenizer.from_pretrained(MERGED_OUTPUT, trust_remote_code=True)
    test_model = AutoModelForCausalLM.from_pretrained(
        MERGED_OUTPUT, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
    )
    test_input = test_tok("Hello, I am a quantum computing", return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = test_model.generate(**test_input, max_new_tokens=30)
    print(f"  Test output: {test_tok.decode(out[0], skip_special_tokens=True)}")

    return json.dumps({"status": "success", "files": len(files), "size_gb": round(total_size/1e9, 2)})


@app.local_entrypoint()
def main():
    result = merge_and_save.remote()
    print(f"\nMerge result: {result}")
