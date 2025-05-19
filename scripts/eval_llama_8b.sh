#!/bin/bash

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
DATA_PATH="data/musique_ans_v1.0_dev.jsonl"
HF_TOKEN="hf_..."

modes=("answer_only_llama" "cot")

for mode in "${modes[@]}"
do
    # original order
    python inference.py \
        --model_name $MODEL_NAME \
        --model_type "llama" \
        --data_path $DATA_PATH \
        --mode $mode \
        --token $HF_TOKEN

    doc_orders=("forward" "backward" "forward_0" "forward_1" "forward_2" "forward_3" "forward_4" "forward_5" "rm_first")

    # context permutation
    for doc_order in "${doc_orders[@]}"
    do
        python inference.py \
            --model_name $MODEL_NAME \
            --model_type "llama" \
            --data_path $DATA_PATH \
            --mode $mode \
            --doc_order $doc_order \
            --token $HF_TOKEN
    done
done