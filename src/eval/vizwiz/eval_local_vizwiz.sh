#!/bin/bash

# Sourced from: https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/vizwiz.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m eval.vizwiz.eval_vizwiz \
        --model-path /content/drive/MyDrive/VizWiz/model_checkpoints/llava-v1.5-7b-task-lora/checkpoint-1
        --model-base liuhaotian/llava-v1.5-7b \
        --question-file /content/test.json \
        --image-folder /content/test \
        --answers-file /content/answers/vizwiz-llava-v1.5-7b.json \
        --temperature 0 \
        --conv-mode vicuna_v1 \
        --num-workers 8
done

wait