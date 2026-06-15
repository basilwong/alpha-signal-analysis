#!/bin/bash
# Monitor V7 jobs: check every 2 minutes, report status, detect completion

echo "Starting job monitor at $(date)"
echo "Monitoring: V7b/c predict, V7d GRPO"
echo "================================================"

GRPO_APP="ap-BIime2JxpoQpLdc1G4GGEe"
V7BC_APP="ap-wSYLnRnfgnmNkzLzpiXtJJ"

check_count=0
v7bc_done=false
v7d_done=false

while true; do
    check_count=$((check_count + 1))
    echo ""
    echo "[Check #$check_count at $(date -u +%H:%M:%S) UTC]"
    
    # Check V7b/c predict
    if [ "$v7bc_done" = false ]; then
        v7bc_status=$(modal app list 2>&1 | grep "$V7BC_APP" | awk '{print $6}')
        if echo "$v7bc_status" | grep -q "stopped"; then
            echo "  V7b/c predict: STOPPED"
            # Check if predictions exist
            v7b_file=$(modal volume ls quantum-alpha-outputs predictions_v7b_rejection.jsonl 2>&1)
            v7c_file=$(modal volume ls quantum-alpha-outputs predictions_v7c_dpo.jsonl 2>&1)
            if echo "$v7b_file" | grep -q "predictions_v7b"; then
                echo "    V7b predictions: SAVED"
                v7bc_done=true
            else
                echo "    V7b predictions: NOT FOUND (may have failed)"
            fi
            if echo "$v7c_file" | grep -q "predictions_v7c"; then
                echo "    V7c predictions: SAVED"
            else
                echo "    V7c predictions: NOT FOUND"
            fi
        else
            echo "  V7b/c predict: RUNNING"
        fi
    else
        echo "  V7b/c predict: COMPLETE"
    fi
    
    # Check V7d GRPO
    if [ "$v7d_done" = false ]; then
        v7d_status=$(modal app list 2>&1 | grep "$GRPO_APP" | awk '{print $6}')
        if echo "$v7d_status" | grep -q "stopped"; then
            echo "  V7d GRPO: STOPPED"
            # Check for checkpoints
            v7d_ckpts=$(modal volume ls quantum-alpha-outputs quantum-alpha-grpo-v7d/ 2>&1)
            if echo "$v7d_ckpts" | grep -q "checkpoint"; then
                echo "    Checkpoints found!"
                echo "$v7d_ckpts" | grep checkpoint
                v7d_done=true
            else
                echo "    NO CHECKPOINTS (failed before first save)"
            fi
        else
            echo "  V7d GRPO: RUNNING"
            # Check for intermediate checkpoints
            v7d_ckpts=$(modal volume ls quantum-alpha-outputs quantum-alpha-grpo-v7d/ 2>&1)
            if echo "$v7d_ckpts" | grep -q "checkpoint"; then
                echo "    Intermediate checkpoints:"
                echo "$v7d_ckpts" | grep checkpoint
            fi
        fi
    else
        echo "  V7d GRPO: COMPLETE"
    fi
    
    # Exit if all done
    if [ "$v7bc_done" = true ] && [ "$v7d_done" = true ]; then
        echo ""
        echo "ALL JOBS COMPLETE!"
        break
    fi
    
    # Wait 2 minutes
    sleep 120
done
