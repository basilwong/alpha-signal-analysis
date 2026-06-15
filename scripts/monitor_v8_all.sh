#!/bin/bash
# Monitor all V8 jobs. Check every 2 minutes.
# Detect: completion, OOM (app stops quickly without output), and progress.

echo "=== V8 FULL MONITOR (started $(date -u)) ==="

BASE_APP="ap-qxzY4hhsoNJxO2dx7LDsSm"
PHASE2_APP="ap-sh3vx8FvtR7ZUyC26FPy4i"

base_14b_done=false
base_32b_done=false
v8_sft_pred_done=false
v8_grpo_done=false

while true; do
    echo ""
    echo "[$(date -u +%H:%M:%S)]"

    # Check 14B base predictions
    if [ "$base_14b_done" = false ]; then
        if modal volume ls quantum-alpha-outputs predictions_base_14b_fixed.jsonl 2>&1 | grep -q "predictions_base_14b_fixed"; then
            size=$(modal volume ls quantum-alpha-outputs predictions_base_14b_fixed.jsonl 2>&1 | grep "predictions_base_14b_fixed" | awk '{print $NF}')
            echo "  14B base: DONE ($size)"
            base_14b_done=true
        else
            echo "  14B base: running"
        fi
    else
        echo "  14B base: COMPLETE"
    fi

    # Check 32B base predictions
    if [ "$base_32b_done" = false ]; then
        if modal volume ls quantum-alpha-outputs predictions_base_32b_fixed.jsonl 2>&1 | grep -q "predictions_base_32b_fixed"; then
            size=$(modal volume ls quantum-alpha-outputs predictions_base_32b_fixed.jsonl 2>&1 | grep "predictions_base_32b_fixed" | awk '{print $NF}')
            echo "  32B base: DONE ($size)"
            base_32b_done=true
        else
            # Check if the app stopped (possible OOM)
            app_status=$(modal app list 2>&1 | grep "$BASE_APP")
            if echo "$app_status" | grep -q "stopped"; then
                echo "  32B base: APP STOPPED WITHOUT OUTPUT - POSSIBLE OOM!"
                base_32b_done=true
            else
                containers=$(echo "$app_status" | grep -oP '\d+(?=\s+\|)')
                echo "  32B base: running (containers: $containers)"
            fi
        fi
    else
        echo "  32B base: COMPLETE"
    fi

    # Check V8 SFT predictions
    if [ "$v8_sft_pred_done" = false ]; then
        if modal volume ls quantum-alpha-outputs predictions_v8_sft.jsonl 2>&1 | grep -q "predictions_v8_sft"; then
            size=$(modal volume ls quantum-alpha-outputs predictions_v8_sft.jsonl 2>&1 | grep "predictions_v8_sft" | awk '{print $NF}')
            echo "  V8 SFT pred: DONE ($size)"
            v8_sft_pred_done=true
        else
            echo "  V8 SFT pred: running"
        fi
    else
        echo "  V8 SFT pred: COMPLETE"
    fi

    # Check V8 GRPO
    if [ "$v8_grpo_done" = false ]; then
        grpo_ckpts=$(modal volume ls quantum-alpha-outputs quantum-alpha-v8-grpo/ 2>&1)
        if echo "$grpo_ckpts" | grep -q "checkpoint"; then
            latest=$(echo "$grpo_ckpts" | grep checkpoint | tail -1 | awk '{print $1}')
            # Check if app is still running
            phase2_status=$(modal app list 2>&1 | grep "$PHASE2_APP")
            if echo "$phase2_status" | grep -q "stopped"; then
                echo "  V8 GRPO: STOPPED (latest: $latest)"
                v8_grpo_done=true
            else
                echo "  V8 GRPO: running (latest: $latest)"
            fi
        else
            phase2_status=$(modal app list 2>&1 | grep "$PHASE2_APP")
            if echo "$phase2_status" | grep -q "stopped"; then
                echo "  V8 GRPO: APP STOPPED - NO CHECKPOINTS (POSSIBLE FAILURE)"
                v8_grpo_done=true
            else
                echo "  V8 GRPO: running (no checkpoints yet)"
            fi
        fi
    else
        echo "  V8 GRPO: COMPLETE"
    fi

    # Check for early app death (OOM detection)
    # If the base app stopped within 5 minutes of launch without producing output, likely OOM
    base_status=$(modal app list 2>&1 | grep "$BASE_APP")
    if echo "$base_status" | grep -q "stopped"; then
        if [ "$base_32b_done" = false ]; then
            echo "  *** WARNING: Base predictions app stopped. 32B may have OOM'd ***"
            base_32b_done=true
        fi
    fi

    # Exit if all done
    if [ "$base_14b_done" = true ] && [ "$base_32b_done" = true ] && [ "$v8_sft_pred_done" = true ] && [ "$v8_grpo_done" = true ]; then
        echo ""
        echo "=== ALL JOBS COMPLETE ==="
        break
    fi

    sleep 120
done
