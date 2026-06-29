# 4-Way Model Comparison: Fine-Tuning and Memory Evaluation

This document outlines the strategic roadmap for conducting a rigorous 4-way model comparison for the **Alpha Signal Analysis** project. The goal is to evaluate the relative and combined contributions of **Supervised Fine-Tuning (SFT)** and **Persistent Memory** using the `qwen3-vl-8b` model class.

This comparison directly addresses the core thesis of the Qwen Cloud Memory Agent hackathon: **How does persistent memory compare to, and complement, domain-specific fine-tuning?**

---

## The 4-Way Comparison Matrix

To isolate the variables of fine-tuning and memory, we will evaluate the model across four distinct configurations on the same 421-article evaluation set (`articles_eval.jsonl`).

| Configuration | Model Used | Memory Context | Purpose |
|---|---|---|---|
| **1. Baseline** | `qwen3-vl-8b-instruct` (Base) | None (Article only) | Establishes the zero-shot performance ceiling of the un-tuned model without context. |
| **2. Baseline + Memory** | `qwen3-vl-8b-instruct` (Base) | Yes (Accumulating) | Measures the value of persistent memory alone (the zero-cost improvement path). |
| **3. Fine-Tuned** | `ft:qwen3-vl-8b-instruct` (Custom) | None (Article only) | Measures the value of SFT alone (the high-cost, static improvement path). |
| **4. Fine-Tuned + Memory** | `ft:qwen3-vl-8b-instruct` (Custom) | Yes (Accumulating) | Measures the synergistic effect of combining both approaches (the production-grade ceiling). |

---

## Technical Feasibility & Model Selection

### Qwen Cloud Fine-Tuning Capabilities
Our research [1] [2] indicates that while the pure text `qwen3-8b` model is not available for fine-tuning on the international edition of Alibaba Cloud Model Studio, the **`qwen3-vl-8b-instruct`** (vision-language) model is fully supported. 

Since our dataset is text-only, we can fine-tune `qwen3-vl-8b-instruct` using our text-only `quantum_alpha_train_v4.jsonl` file (the model accepts text-only inputs fine; we simply omit the `image` fields from the ChatML messages).

### Training Cost Estimate
- **Dataset size**: 881 examples [3]
- **Average length**: ~2,161 tokens per example [3]
- **Total tokens per epoch**: ~1.9 million tokens
- **Training duration**: 4 epochs (total ~7.6 million tokens)
- **Unit price**: $0.002 per 1,000 tokens for `qwen3-vl-8b-instruct` [2]
- **Total cost**: **~$15.20** (fully covered by your $40 voucher)

---

## Execution Roadmap (Phased Plan)

### Phase 1: Run Baseline Configurations (1-2 Days)
We will generate predictions for the two non-fine-tuned configurations using the existing Qwen Cloud API.

1. **Run Configuration 1 (Baseline)**:
   - Run predictions on 421 articles using `qwen3-vl-8b-instruct` with NO memory context.
   - Save to `data/eval/predictions_qwen3_8b_base_nomemory.jsonl`.
2. **Run Configuration 2 (Baseline + Memory)**:
   - Run predictions on 421 articles using `qwen3-vl-8b-instruct` with the `run_iterative_memory_loop.py` script.
   - Save to `data/eval/predictions_qwen3_8b_base_memory.jsonl`.

### Phase 2: Qwen Cloud Fine-Tuning (1 Day)
We will upload the V4 training data and trigger the fine-tuning job on Alibaba Cloud Model Studio.

1. **Upload Dataset**:
   - Upload `data/training/quantum_alpha_train_v4.jsonl` to Model Studio via the files API [1].
2. **Create SFT Job**:
   - Submit a fine-tuning job for `qwen3-vl-8b-instruct` using the uploaded file ID [1].
   - Configure hyperparameters: 4 epochs, learning rate 5e-5, LoRA rank 64 [3].
3. **Monitor and Deploy**:
   - Poll the job status until complete.
   - Deploy the fine-tuned model to a custom endpoint on Model Studio [1].

### Phase 3: Run Fine-Tuned Configurations (1-2 Days)
We will generate predictions for the two fine-tuned configurations using our newly deployed custom model.

1. **Run Configuration 3 (Fine-Tuned)**:
   - Run predictions on 421 articles using the fine-tuned model with NO memory context.
   - Save to `data/eval/predictions_qwen3_8b_ft_nomemory.jsonl`.
2. **Run Configuration 4 (Fine-Tuned + Memory)**:
   - Run predictions on 421 articles using the fine-tuned model with the `run_iterative_memory_loop.py` script.
   - Save to `data/eval/predictions_qwen3_8b_ft_memory.jsonl`.

### Phase 4: Evaluation & Visualization (1 Day)
We will run the evaluation pipeline on all four prediction files and update the frontend.

1. **Run Evaluation**:
   - Run `eval/run_multi_model_eval.py` on all four files to compute IC, direction accuracy, and decay curves.
2. **Update Dashboard**:
   - Update the Evaluation Dashboard (Tab 3) in `app_server.py` to display the 4-way comparison.
   - Show overlaid decay curves for all four configurations to visually demonstrate the cumulative value of memory and SFT.

---

## References

[1] [Alibaba Cloud Model Studio: Fine-tune with the API or CLI](https://help.aliyun.com/en/model-studio/fine-tuning-api-guide)  
[2] [Alibaba Cloud Model Studio: Fine-tune Qwen (International Edition)](https://www.alibabacloud.com/help/en/model-studio/text-generation-model-tuning)  
[3] [Alpha Signal Analysis: V4 Fine-Tuning Specifications](https://github.com/basilwong/alpha-signal-analysis/blob/main/README.md)
