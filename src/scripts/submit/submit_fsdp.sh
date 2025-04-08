#!/bin/bash

# Activate the Python virtual environment
source activate /home/guevel/.conda/envs/archehr

# Define the arguments for the submit.py script
MODEL_NAME="cross-encoder/nli-deberta-v3-base"
DATA_PATH="/home/guevel/projects/archehr/data/1.1/dev"
BATCH_SIZE=128
NUM_EPOCHS=300
LEARNING_RATE=1e-5
SAVE_NAME="/home/guevel/projects/archehr/logs/fsdp_deberta-v3-base-fc"
WORLD_SIZE=4  # Number of GPUs to use

# Run the submit.py script with the specified arguments
python /home/etienneguevel/Documents/SCAI/projects/archehr/src/archehr/cross_encode/submit.py \
    --model_name "$MODEL_NAME" \
    --data_path "$DATA_PATH" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --save_folder "$SAVE_NAME" \
    --world_size "$WORLD_SIZE"