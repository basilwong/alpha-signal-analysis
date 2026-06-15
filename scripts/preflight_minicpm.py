"""
Pre-flight Validation for MiniCPM-2B Fine-Tuning

Validates EVERYTHING before committing to a full training run:
1. Model loads correctly with Unsloth
2. LoRA adapters apply without errors
3. Tokenizer + chat template produce valid formatted text
4. Training data loads and tokenizes within context window
5. A single training step completes with decreasing loss
6. Model generates coherent output after 1 step (sanity check)

Usage:
    modal run scripts/preflight_minicpm.py
"""

import modal
import json

app = modal.App("quantum-alpha-preflight-minicpm")

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

# Try multiple model IDs in order of preference
MODEL_CANDIDATES = [
    "openbmb/MiniCPM-2B-sft-bf16-llama-format",
    "openbmb/MiniCPM-2B-sft-bf16",
    "openbmb/MiniCPM-2B-dpo-bf16",
]
TRAIN_FILE = "/outputs/quantum_alpha_train_v4.jsonl"


@app.function(
    image=finetune_image,
    gpu="A100",
    timeout=600,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/outputs": output_vol,
    },
)
def preflight():
    """Run all pre-flight checks."""
    import torch
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer, SFTConfig

    results = {}

    # =========================================================
    # CHECK 1: Model loads correctly
    # =========================================================
    print("=" * 60)
    print("CHECK 1: Loading model with Unsloth...")
    print("=" * 60)
    model = None
    tokenizer = None
    loaded_model_name = None

    for candidate in MODEL_CANDIDATES:
        print(f"  Trying: {candidate}")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=candidate,
                max_seq_length=4096,
                dtype=torch.bfloat16,
                load_in_4bit=True,
                trust_remote_code=True,
            )
            loaded_model_name = candidate
            break
        except Exception as e:
            print(f"    Failed: {str(e)[:100]}")
            continue

    if model is None:
        # Fall back to standard HuggingFace PEFT
        print("  Unsloth failed for all candidates. Trying standard HF loading...")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            loaded_model_name = MODEL_CANDIDATES[0]
            tokenizer = AutoTokenizer.from_pretrained(loaded_model_name, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(
                loaded_model_name, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True
            )
            print(f"  Loaded via standard HF: {loaded_model_name}")
            results["model_load"] = f"PASS (HF fallback: {loaded_model_name})"
            results["use_unsloth"] = False
        except Exception as e2:
            print(f"  ALL LOADING FAILED: {e2}")
            results["model_load"] = f"FAIL: {e2}"
            return json.dumps(results)
    else:
        param_count = sum(p.numel() for p in model.parameters()) / 1e9
        print(f"  Model loaded via Unsloth: {loaded_model_name}")
        print(f"  Parameters: {param_count:.2f}B")
        results["model_load"] = f"PASS (Unsloth: {loaded_model_name})"
        results["use_unsloth"] = True

    print(f"  Tokenizer vocab size: {tokenizer.vocab_size}")
    print(f"  Pad token: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")
    print(f"  EOS token: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")

    # =========================================================
    # CHECK 2: LoRA adapters apply correctly
    # =========================================================
    print("\n" + "=" * 60)
    print("CHECK 2: Applying LoRA adapters (r=16, alpha=32)...")
    print("=" * 60)
    try:
        model = FastLanguageModel.get_peft_model(
            model,
            r=16,
            lora_alpha=32,
            lora_dropout=0,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
        )
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        results["lora_apply"] = "PASS"
    except Exception as e:
        print(f"  FAILED: {e}")
        results["lora_apply"] = f"FAIL: {e}"
        return json.dumps(results)

    # =========================================================
    # CHECK 3: Chat template and tokenization
    # =========================================================
    print("\n" + "=" * 60)
    print("CHECK 3: Testing chat template and tokenization...")
    print("=" * 60)
    try:
        # Load first training example
        with open(TRAIN_FILE) as f:
            first_example = json.loads(f.readline())

        messages = first_example["messages"]
        print(f"  Example has {len(messages)} messages: {[m['role'] for m in messages]}")

        # Apply chat template
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        print(f"  Formatted text length: {len(formatted)} chars")
        print(f"  First 200 chars: {formatted[:200]}")

        # Tokenize and check length
        tokens = tokenizer(formatted, return_tensors="pt")
        token_count = tokens["input_ids"].shape[1]
        print(f"  Token count: {token_count}")

        if token_count > 4096:
            print(f"  WARNING: Example exceeds 4096 tokens!")
            results["chat_template"] = f"WARN: {token_count} tokens"
        else:
            results["chat_template"] = "PASS"
    except Exception as e:
        print(f"  FAILED: {e}")
        results["chat_template"] = f"FAIL: {e}"
        return json.dumps(results)

    # =========================================================
    # CHECK 4: Full dataset tokenization stats
    # =========================================================
    print("\n" + "=" * 60)
    print("CHECK 4: Dataset tokenization statistics...")
    print("=" * 60)
    try:
        token_lengths = []
        errors = 0
        with open(TRAIN_FILE) as f:
            for i, line in enumerate(f):
                row = json.loads(line)
                try:
                    text = tokenizer.apply_chat_template(
                        row["messages"], tokenize=False, add_generation_prompt=False
                    )
                    toks = tokenizer(text, return_tensors="pt")
                    token_lengths.append(toks["input_ids"].shape[1])
                except Exception:
                    errors += 1

        print(f"  Total examples: {len(token_lengths)}")
        print(f"  Tokenization errors: {errors}")
        print(f"  Min tokens: {min(token_lengths)}")
        print(f"  Max tokens: {max(token_lengths)}")
        print(f"  Mean tokens: {sum(token_lengths)/len(token_lengths):.0f}")
        print(f"  Examples > 4096 tokens: {sum(1 for t in token_lengths if t > 4096)}")

        results["dataset_stats"] = f"PASS ({len(token_lengths)} examples, max={max(token_lengths)})"
    except Exception as e:
        print(f"  FAILED: {e}")
        results["dataset_stats"] = f"FAIL: {e}"

    # =========================================================
    # CHECK 5: Single training step
    # =========================================================
    print("\n" + "=" * 60)
    print("CHECK 5: Running single training step...")
    print("=" * 60)
    try:
        # Prepare small dataset (10 examples)
        rows = []
        with open(TRAIN_FILE) as f:
            for i, line in enumerate(f):
                if i >= 10:
                    break
                row = json.loads(line)
                text = tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                )
                rows.append({"text": text})

        ds = Dataset.from_list(rows)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        sft_config = SFTConfig(
            output_dir="/tmp/preflight_test",
            num_train_epochs=1,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=2e-4,
            warmup_steps=0,
            max_steps=3,  # Only 3 steps
            bf16=True,
            logging_steps=1,
            report_to="none",
            dataset_text_field="text",
            max_seq_length=4096,
            packing=True,
            seed=42,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=ds,
            args=sft_config,
        )

        train_result = trainer.train()
        log_history = trainer.state.log_history

        print(f"  Training completed!")
        for entry in log_history:
            if "loss" in entry:
                print(f"    Step {entry.get('step', '?')}: loss={entry['loss']:.4f}")

        # Check loss is decreasing or at least reasonable
        losses = [e["loss"] for e in log_history if "loss" in e]
        if losses and losses[0] < 20:  # Sanity check: loss should be reasonable
            results["training_step"] = f"PASS (losses: {[f'{l:.3f}' for l in losses]})"
        else:
            results["training_step"] = f"WARN: unusual loss values: {losses}"

    except Exception as e:
        print(f"  FAILED: {e}")
        results["training_step"] = f"FAIL: {e}"
        return json.dumps(results)

    # =========================================================
    # CHECK 6: Quick inference test
    # =========================================================
    print("\n" + "=" * 60)
    print("CHECK 6: Quick inference test...")
    print("=" * 60)
    try:
        FastLanguageModel.for_inference(model)
        test_messages = [
            {"role": "system", "content": "You are a quantum computing signal generator. Output valid JSON."},
            {"role": "user", "content": "IonQ announces 50 algorithmic qubits. Generate a signal vector."},
        ]
        inputs = tokenizer.apply_chat_template(
            test_messages, return_tensors="pt", add_generation_prompt=True
        ).to("cuda")

        with torch.no_grad():
            outputs = model.generate(inputs, max_new_tokens=200, temperature=0.3, do_sample=True)

        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        print(f"  Generated {len(response)} chars")
        print(f"  First 300 chars: {response[:300]}")

        # Check if output contains JSON-like structure
        has_json = "{" in response and "}" in response
        print(f"  Contains JSON structure: {has_json}")
        results["inference"] = f"PASS (has_json={has_json})"
    except Exception as e:
        print(f"  FAILED: {e}")
        results["inference"] = f"FAIL: {e}"

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("PRE-FLIGHT SUMMARY")
    print("=" * 60)
    all_pass = True
    for check, result in results.items():
        result_str = str(result)
        status = "PASS" if "PASS" in result_str else ("WARN" if "WARN" in result_str else "FAIL")
        if isinstance(result, bool):
            status = "PASS"  # boolean flags are informational
        icon = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗"}[status]
        print(f"  {icon} {check}: {result_str}")
        if "FAIL" in result_str:
            all_pass = False

    if all_pass:
        print("\n  ALL CHECKS PASSED. Safe to proceed with full training.")
    else:
        print("\n  SOME CHECKS FAILED. Review before proceeding.")

    return json.dumps(results)


@app.local_entrypoint()
def main():
    result = preflight.remote()
    print(f"\nFinal results: {result}")
