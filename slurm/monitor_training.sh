#!/bin/bash
# Monitor GeoCryoAI training job

JOB_ID=$1

if [ -z "$JOB_ID" ]; then
    echo "Usage: ./monitor_training.sh <job_id>"
    exit 1
fi

# Display job status
scontrol show job $JOB_ID

# Tail training log
LOG_FILE="logs/geocryoai_final_${JOB_ID}.log"
echo "Training log: $LOG_FILE"
tail -f $LOG_FILE