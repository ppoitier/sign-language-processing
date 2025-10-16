#!/bin/bash

if [ -z "$1" ]; then
    echo "Error: No config filepath provided.." >&2
    exit 1
fi

CONFIG_FILE="$1"

echo "Starting training with config: $CONFIG_FILE"
python ./slp/tasks/segmentation/launch_training.py -c "$CONFIG_FILE"
