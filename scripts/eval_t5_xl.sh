#!/bin/bash

MODEL_NAME="google/flan-t5-xl"
DATA_PATH="data/musique_ans_v1.0_dev.jsonl"

# original order
python inference.py \
    --model_name $MODEL_NAME \
    --model_type "t5" \
    --data_path $DATA_PATH \
    --mode "t5" \

doc_orders=("forward" "backward" "forward_0" "forward_1" "forward_2" "forward_3" "forward_4" "forward_5" "rm_first")

# context permutation
for doc_order in "${doc_orders[@]}"
do
    python inference.py \
        --model_name $MODEL_NAME \
        --model_type "t5" \
        --data_path $DATA_PATH \
        --mode "t5" \
        --doc_order $doc_order
done
