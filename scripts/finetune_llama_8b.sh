#!/bin/bash

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH="data/musique_train_llama.jsonl"
OUTPUT_DIR="finetuned_models/llama_8b"
HF_TOKEN="hf_..."

python finetune.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --token $HF_TOKEN
