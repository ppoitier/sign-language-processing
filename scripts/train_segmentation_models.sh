#!/bin/bash

CONDA_ENV_NAME="slp"
PYTHON_SCRIPT="slp/tasks/segmentation/launch_training.py"
LOG_DIR="/home/sign-language/outputs/logs"
MAX_CONCURRENT_JOBS=1
TMUX_SESSION_PREFIX="slp"
CONFIG_FILES=(
    "configs/sls/dgs/dgs_actionness_mstcn.yaml"
)

echo "Starting experiment launcher..."
echo "Environment: $CONDA_ENV_NAME"
echo "Script: $PYTHON_SCRIPT"
echo "Max concurrency: $MAX_CONCURRENT_JOBS"
mkdir -p "$LOG_DIR"
echo "Saving logs to: $LOG_DIR"
echo "------------------------------------------------"


for config_file in "${CONFIG_FILES[@]}"; do
    config_basename=$(basename "$config_file")
    config_name_noext="${config_basename%.*}"
    session_name="${TMUX_SESSION_PREFIX}_${config_name_noext}"

    echo "Preparing to launch: $session_name (Config: $config_file)"
    CMD="set -o pipefail; \
             conda run -n \"$CONDA_ENV_NAME\" python \"$PYTHON_SCRIPT\" --config-path \"$config_file\" 2>&1 \
             | tee \"$LOG_FILE\"; \
             (exit \${PIPESTATUS[0]}); \
             echo ''; echo '---'; echo 'Script finished. Log saved to $LOG_FILE';"

    tmux new-session -d -s "$session_name" "$CMD" &
    job_count=$(jobs -r -p | wc -l)

    if [ "$job_count" -ge "$MAX_CONCURRENT_JOBS" ]; then
        echo "Max concurrent jobs ($MAX_CONCURRENT_JOBS) reached. Waiting..."
        wait -n
    fi
done

echo "All jobs have been launched. Waiting for the last batch..."
wait

echo "------------------------------------------------"
echo "All scripts have completed."