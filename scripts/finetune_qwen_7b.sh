#!/bin/bash

MODEL_NAME="Qwen/Qwen2.5-7B-Instruct"
DATA_PATH="data/musique_train_qwen.jsonl"
OUTPUT_DIR="finetuned_models/qwen_7b"
HF_TOKEN="hf_..."

python finetune.py \
    --model_name $MODEL_NAME \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --token $HF_TOKEN
