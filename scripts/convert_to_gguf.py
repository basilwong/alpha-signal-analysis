"""
Convert merged MiniCPM-2B model to GGUF Q4_K_M format on Modal.

Usage:
    modal run scripts/convert_to_gguf.py
"""

import modal

app = modal.App("quantum-alpha-gguf-convert")

convert_image = (
    modal.Image.from_registry("nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11")
    .entrypoint([])
    .apt_install("git", "build-essential", "cmake", "curl", "libcurl4-openssl-dev", "pciutils")
    .pip_install("torch", "transformers==4.57.3", "sentencepiece", "protobuf", "numpy", "gguf")
    .run_commands(
        "git clone https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cmake /opt/llama.cpp -B /opt/llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DLLAMA_CUDA=ON",
        "cmake --build /opt/llama.cpp/build --config Release -j --target llama-quantize",
        "cp /opt/llama.cpp/build/bin/llama-quantize /usr/local/bin/",
    )
)

output_vol = modal.Volume.from_name("quantum-alpha-outputs", create_if_missing=True)

MERGED_MODEL = "/outputs/minicpm-2b-merged"
GGUF_OUTPUT = "/outputs/minicpm-2b-quantum-alpha-Q4_K_M.gguf"


@app.function(
    image=convert_image,
    gpu="T4",  # Only need GPU for quantize, minimal cost
    timeout=1800,
    volumes={
        "/outputs": output_vol,
    },
)
def convert():
    """Convert merged model to GGUF Q4_K_M."""
    import subprocess
    import os
    import sys

    print("Step 1: Converting HF model to GGUF BF16...")
    bf16_path = "/tmp/minicpm-2b-bf16.gguf"

    result = subprocess.run(
        [
            sys.executable, "/opt/llama.cpp/convert_hf_to_gguf.py",
            MERGED_MODEL,
            "--outfile", bf16_path,
            "--outtype", "bf16",
        ],
        capture_output=True, text=True
    )
    print(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"BF16 conversion failed with code {result.returncode}")

    bf16_size = os.path.getsize(bf16_path)
    print(f"  BF16 GGUF: {bf16_size/1e9:.2f} GB")

    print("\nStep 2: Quantizing to Q4_K_M...")
    result = subprocess.run(
        ["llama-quantize", bf16_path, GGUF_OUTPUT, "Q4_K_M"],
        capture_output=True, text=True
    )
    print(result.stdout[-2000:] if result.stdout else "")
    if result.returncode != 0:
        print(f"STDERR: {result.stderr[-2000:]}")
        raise RuntimeError(f"Quantization failed with code {result.returncode}")

    q4_size = os.path.getsize(GGUF_OUTPUT)
    print(f"  Q4_K_M GGUF: {q4_size/1e9:.2f} GB")

    print("\nStep 3: Committing to volume...")
    output_vol.commit()
    print("  Done!")

    # Cleanup
    os.remove(bf16_path)

    print(f"\nGGUF file ready at: {GGUF_OUTPUT}")
    print(f"Download with: modal volume get quantum-alpha-outputs minicpm-2b-quantum-alpha-Q4_K_M.gguf ./minicpm-2b-quantum-alpha-Q4_K_M.gguf")

    return f"Success: {q4_size/1e9:.2f} GB"


@app.local_entrypoint()
def main():
    result = convert.remote()
    print(f"\nResult: {result}")
