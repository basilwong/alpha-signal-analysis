#!/bin/bash
# Monitor corrected V7 jobs (using training data only, no eval contamination)
# Checks every 2 minutes. When V7b/c candidates finish, kicks off training + predict.

echo "=== V7 CLEAN MONITOR (started $(date -u)) ==="
echo "V7b/c candidates: ap-A306zQL3Q9ZeYf1xYEUZJX"
echo "V7d GRPO: ap-k8s9zgMCSDxNz49VVHShYo"
echo ""

CANDIDATES_APP="ap-A306zQL3Q9ZeYf1xYEUZJX"
GRPO_APP="ap-k8s9zgMCSDxNz49VVHShYo"

candidates_done=false
v7b_trained=false
v7d_done=false

while true; do
    echo "[$(date -u +%H:%M:%S)]"

    # Check V7b/c candidates
    if [ "$candidates_done" = false ]; then
        status=$(modal app list 2>&1 | grep "$CANDIDATES_APP")
        if echo "$status" | grep -q "stopped"; then
            # Check if output files exist
            v7b_file=$(modal volume ls quantum-alpha-outputs v7b_best_of_4_clean.jsonl 2>&1)
            if echo "$v7b_file" | grep -q "v7b_best"; then
                echo "  V7b/c candidates: COMPLETE (files saved)"
                candidates_done=true
                
                # Immediately kick off V7b training + V7c training
                echo "  Kicking off V7b rejection SFT..."
                cd /home/ubuntu/quantum-alpha-intelligence
                # Update V7b script to use clean data
                sed -i 's|/outputs/v7b_best_of_4.jsonl|/outputs/v7b_best_of_4_clean.jsonl|' scripts/train_v7b_rejection.py
                sed -i 's|/outputs/quantum-alpha-v7b-rejection|/outputs/quantum-alpha-v7b-clean|' scripts/train_v7b_rejection.py
                modal run --detach scripts/train_v7b_rejection.py > /dev/null 2>&1 &
                
                echo "  Kicking off V7c DPO..."
                sed -i 's|/outputs/v7c_dpo_pairs.jsonl|/outputs/v7c_dpo_pairs_clean.jsonl|' scripts/train_v7c_dpo.py
                sed -i 's|/outputs/quantum-alpha-v7c-dpo|/outputs/quantum-alpha-v7c-dpo-clean|' scripts/train_v7c_dpo.py
                modal run --detach scripts/train_v7c_dpo.py > /dev/null 2>&1 &
                
                echo "  V7b + V7c training launched!"
            else
                echo "  V7b/c candidates: STOPPED but NO OUTPUT (failed)"
                echo "  Check logs for errors."
                candidates_done=true  # Don't keep checking
            fi
        else
            echo "  V7b/c candidates: RUNNING"
        fi
    fi

    # Check V7b/c training (after candidates done)
    if [ "$candidates_done" = true ] && [ "$v7b_trained" = false ]; then
        v7b_status=$(modal app list 2>&1 | grep "v7b-rejection" | grep -v stopped | tail -1)
        v7c_status=$(modal app list 2>&1 | grep "v7c-dpo" | grep -v stopped | tail -1)
        
        if [ -z "$v7b_status" ] && [ -z "$v7c_status" ]; then
            # Both stopped, check for checkpoints
            v7b_ckpt=$(modal volume ls quantum-alpha-outputs quantum-alpha-v7b-clean/ 2>&1)
            v7c_ckpt=$(modal volume ls quantum-alpha-outputs quantum-alpha-v7c-dpo-clean/ 2>&1)
            
            if echo "$v7b_ckpt" | grep -q "checkpoint"; then
                echo "  V7b training: COMPLETE"
            fi
            if echo "$v7c_ckpt" | grep -q "checkpoint"; then
                echo "  V7c training: COMPLETE"
            fi
            
            if echo "$v7b_ckpt" | grep -q "checkpoint" && echo "$v7c_ckpt" | grep -q "checkpoint"; then
                echo "  Both V7b + V7c training done! Ready for merge+predict."
                v7b_trained=true
            fi
        else
            echo "  V7b/c training: STILL RUNNING"
        fi
    fi

    # Check V7d GRPO
    if [ "$v7d_done" = false ]; then
        grpo_status=$(modal app list 2>&1 | grep "$GRPO_APP")
        if echo "$grpo_status" | grep -q "stopped"; then
            v7d_ckpts=$(modal volume ls quantum-alpha-outputs quantum-alpha-grpo-v7d-clean/ 2>&1)
            if echo "$v7d_ckpts" | grep -q "checkpoint"; then
                echo "  V7d GRPO: COMPLETE (checkpoints saved)"
                echo "$v7d_ckpts" | grep checkpoint
                v7d_done=true
            else
                echo "  V7d GRPO: STOPPED (no checkpoints - FAILED)"
                v7d_done=true
            fi
        else
            # Check for intermediate checkpoints
            v7d_ckpts=$(modal volume ls quantum-alpha-outputs quantum-alpha-grpo-v7d-clean/ 2>&1)
            if echo "$v7d_ckpts" | grep -q "checkpoint"; then
                echo "  V7d GRPO: RUNNING (has checkpoints)"
                echo "$v7d_ckpts" | grep checkpoint
            else
                echo "  V7d GRPO: RUNNING (no checkpoints yet)"
            fi
        fi
    else
        echo "  V7d GRPO: DONE"
    fi

    # Exit conditions
    if [ "$v7b_trained" = true ] && [ "$v7d_done" = true ]; then
        echo ""
        echo "=== ALL JOBS COMPLETE ==="
        break
    fi

    sleep 120
done
