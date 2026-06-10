#!/bin/bash
cd /home/ubuntu/quantum-alpha-intelligence
ITERATION=$1

echo "=== ITERATION $ITERATION ($(date -u)) ==="

# Step 1: Try to download all missing batch files from volume
echo "Downloading from volume..."
for i in $(seq 0 8); do
  batch_file="predictions_batch_$(printf '%03d' $i).jsonl"
  local_file="data/eval/$batch_file"
  if [ ! -s "$local_file" ]; then
    modal volume get quantum-alpha-outputs "$batch_file" "$local_file" --force 2>/dev/null
  fi
done

# Step 2: Check status
echo "---STATUS---"
COMPLETE=0
for i in $(seq 0 8); do
  f="data/eval/predictions_batch_$(printf '%03d' $i).jsonl"
  if [ -s "$f" ]; then
    echo "  Batch $i: EXISTS ($(wc -l < "$f") lines)"
    COMPLETE=$((COMPLETE + 1))
  else
    echo "  Batch $i: MISSING"
  fi
done
echo "Complete: $COMPLETE/9"

# If all done, exit with special code
if [ $COMPLETE -eq 9 ]; then
  echo "ALL BATCHES COMPLETE!"
  exit 0
fi

# Step 3: Check if app is running
echo "---APP STATUS---"
APP_LINE=$(modal app list 2>&1 | grep "quantum-alpha-predictions-v2" | grep "ephemeral")
if [ -n "$APP_LINE" ]; then
  echo "  App is running: $APP_LINE"
  echo "ACTION: Waiting (app still running)"
else
  echo "  App is NOT running"
  echo "ACTION: Restarting batch run..."
  modal run --detach scripts/generate_predictions_v2.py 2>&1
  echo "  Restart initiated."
fi

exit 1
