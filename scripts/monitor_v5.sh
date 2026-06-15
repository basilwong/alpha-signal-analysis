#!/bin/bash
# Monitor V5 pipeline - check every 10 minutes, restart if stalled

cd /home/ubuntu/alpha-signal-analysis
LOG="logs/v5_monitor.log"

echo "=== V5 Monitor Started $(date) ===" >> "$LOG"

PREV_COUNT=0

while true; do
    # Get current count
    CURRENT_COUNT=$(wc -l < data/training/alpha_signal_train_v5_raw.jsonl 2>/dev/null || echo 0)
    SUCCESSES=$(grep -c '"success": true' data/training/alpha_signal_train_v5_raw.jsonl 2>/dev/null || echo 0)
    FAILURES=$(grep -c '"success": false' data/training/alpha_signal_train_v5_raw.jsonl 2>/dev/null || echo 0)
    RUNNING=$(ps aux | grep generate_v5_thinking | grep -v grep | wc -l)
    
    echo "[$(date)] Total=$CURRENT_COUNT Success=$SUCCESSES Fail=$FAILURES Running=$RUNNING" >> "$LOG"
    
    # Check if complete
    if [ "$CURRENT_COUNT" -ge 832 ]; then
        echo "[$(date)] COMPLETE! All 832 tasks processed." >> "$LOG"
        echo "COMPLETE! Total=$CURRENT_COUNT Success=$SUCCESSES Fail=$FAILURES"
        break
    fi
    
    # Check if process died
    if [ "$RUNNING" -eq 0 ]; then
        echo "[$(date)] Process not running! Restarting..." >> "$LOG"
        # Remove failures so they get retried
        python3 -c "
import json
with open('data/training/alpha_signal_train_v5_raw.jsonl') as f:
    records = [json.loads(l) for l in f if l.strip()]
successes = [r for r in records if r.get('success')]
with open('data/training/alpha_signal_train_v5_raw.jsonl', 'w') as f:
    for r in successes:
        f.write(json.dumps(r) + '\n')
print(f'Kept {len(successes)} successes, removed {len(records)-len(successes)} failures')
" >> "$LOG" 2>&1
        # Restart
        nohup python3 -u scripts/generate_v5_thinking.py >> logs/v5_generation.log 2>&1 &
        echo "[$(date)] Restarted with PID $!" >> "$LOG"
    fi
    
    # Check if stalled (no progress in 10 minutes)
    if [ "$CURRENT_COUNT" -eq "$PREV_COUNT" ] && [ "$CURRENT_COUNT" -gt 0 ] && [ "$RUNNING" -gt 0 ]; then
        echo "[$(date)] WARNING: No progress in last 10 min (stuck at $CURRENT_COUNT). Killing and restarting..." >> "$LOG"
        pkill -f generate_v5_thinking
        sleep 5
        # Remove failures
        python3 -c "
import json
with open('data/training/alpha_signal_train_v5_raw.jsonl') as f:
    records = [json.loads(l) for l in f if l.strip()]
successes = [r for r in records if r.get('success')]
with open('data/training/alpha_signal_train_v5_raw.jsonl', 'w') as f:
    for r in successes:
        f.write(json.dumps(r) + '\n')
print(f'Kept {len(successes)} successes, removed {len(records)-len(successes)} failures')
" >> "$LOG" 2>&1
        # Restart
        nohup python3 -u scripts/generate_v5_thinking.py >> logs/v5_generation.log 2>&1 &
        echo "[$(date)] Restarted with PID $!" >> "$LOG"
    fi
    
    PREV_COUNT=$CURRENT_COUNT
    
    # Wait 10 minutes
    sleep 600
done
