#!/bin/bash
# Monitor Modal batch predictions and restart when needed.
# Checks every 5 minutes. Downloads completed batches. Kicks off next batch.

cd /home/ubuntu/alpha-signal-analysis
TOTAL_BATCHES=9

while true; do
    echo "$(date): Checking batch status..."
    
    # Check how many batches are done locally
    DONE=0
    for i in $(seq 0 $((TOTAL_BATCHES - 1))); do
        BATCH_FILE="data/eval/predictions_batch_$(printf '%03d' $i).jsonl"
        if [ -f "$BATCH_FILE" ]; then
            DONE=$((DONE + 1))
        fi
    done
    
    echo "  Local batches done: $DONE/$TOTAL_BATCHES"
    
    # If all done, exit
    if [ $DONE -ge $TOTAL_BATCHES ]; then
        echo "ALL BATCHES COMPLETE!"
        break
    fi
    
    # Try to download any new batches from the volume
    for i in $(seq 0 $((TOTAL_BATCHES - 1))); do
        BATCH_FILE="data/eval/predictions_batch_$(printf '%03d' $i).jsonl"
        if [ ! -f "$BATCH_FILE" ]; then
            # Try to download from volume
            modal volume get quantum-alpha-outputs "predictions_batch_$(printf '%03d' $i).jsonl" "$BATCH_FILE" --force 2>/dev/null
            if [ -f "$BATCH_FILE" ] && [ -s "$BATCH_FILE" ]; then
                COUNT=$(wc -l < "$BATCH_FILE")
                echo "  Downloaded batch $i: $COUNT results"
            else
                rm -f "$BATCH_FILE" 2>/dev/null
            fi
        fi
    done
    
    # Recount after downloads
    DONE=0
    for i in $(seq 0 $((TOTAL_BATCHES - 1))); do
        BATCH_FILE="data/eval/predictions_batch_$(printf '%03d' $i).jsonl"
        if [ -f "$BATCH_FILE" ] && [ -s "$BATCH_FILE" ]; then
            DONE=$((DONE + 1))
        fi
    done
    
    echo "  After download check: $DONE/$TOTAL_BATCHES batches done"
    
    # If all done now, exit
    if [ $DONE -ge $TOTAL_BATCHES ]; then
        echo "ALL BATCHES COMPLETE!"
        break
    fi
    
    # Check if Modal app is running
    RUNNING=$(modal app list 2>&1 | grep "quantum-alpha-predictions-v2" | grep -c "ephemeral (detached)")
    
    if [ $RUNNING -eq 0 ]; then
        echo "  Modal app not running. Restarting for next batch..."
        modal run --detach scripts/generate_predictions_v2.py 2>&1 | grep -E "Starting batch|already done|Initialized"
        echo "  Restarted."
    else
        echo "  Modal app still running. Waiting..."
    fi
    
    echo "  Sleeping 5 minutes..."
    sleep 300
done

echo "$(date): Monitor complete. All $TOTAL_BATCHES batches done."
echo "Combining all batches..."
cat data/eval/predictions_batch_*.jsonl > data/eval/predictions_finetuned_all.jsonl
TOTAL=$(wc -l < data/eval/predictions_finetuned_all.jsonl)
echo "Combined: $TOTAL total predictions"
