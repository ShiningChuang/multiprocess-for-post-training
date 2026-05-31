#!/usr/bin/env bash
# Watchdog: kill V0 when step 50 is reached
TARGET_STEP=50
PID=8552
LOG="/data/project/v0_results/run.log"

echo "Watchdog started. Waiting for step $TARGET_STEP in $LOG (PID=$PID)..."

while true; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "Process $PID already exited."
        break
    fi

    if grep -q "^Step[[:space:]]*${TARGET_STEP} " "$LOG" 2>/dev/null; then
        echo "Step $TARGET_STEP reached. Waiting 30s for training step to finish writing..."
        sleep 30
        echo "Sending SIGTERM to PID $PID..."
        kill $PID 2>/dev/null
        sleep 5
        if ps -p $PID > /dev/null 2>&1; then
            echo "SIGTERM didn't work, sending SIGKILL..."
            kill -9 $PID 2>/dev/null
        fi
        echo "V0 stopped at step $TARGET_STEP."
        break
    fi

    sleep 30
done

echo "---"
echo "Final log entries:"
grep "^Step" "$LOG" | tail -10
echo "---"
echo "Total steps logged: $(grep -c '^Step' "$LOG")"
