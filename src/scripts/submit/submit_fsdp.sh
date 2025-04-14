#!/bin/bash
# Activate the Python virtual environment
source /home/guevel/.bashrc
conda activate archehr

# Define the arguments for the submit.py script
MODEL_NAME="Alibaba-NLP/gte-Qwen2-7B-instruct"
DATA_PATH="/home/guevel/projects/archehr/data/1.1/dev"
BATCH_SIZE=64
NUM_EPOCHS=300
LEARNING_RATE=1e-5
SAVE_NAME="/home/guevel/projects/archehr/logs/Qwen2_gte_mlp"
WORLD_SIZE=6  # Number of GPUs to use

# Make a log file for text outputs
mkdir -p "$SAVE_NAME"
LOGFILE="${SAVE_NAME}/output.log"

# Run the submit.py script with the specified arguments
python /home/guevel/projects/archehr/src/archehr/cross_encode/submit.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --save_folder "$SAVE_NAME" \
    --world_size "$WORLD_SIZE" \
    > "$LOGFILE" 2>&1
