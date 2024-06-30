#!/bin/bash

# Sourced from: https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/vqav2.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

# Python Script
VQA_LOADER="eval.eval_vqa_v2"

# Model
MODEL_PATH="liuhaotian/llava-v1.5-13b"
CKPT="llava-v1.5-13b"
CONV_MODE="vicuna_v1"

# Data Paths
IMAGE_FOLDER="/content/drive/MyDrive/VQA/data/Images/test2015"
QUESTION_FILE="/content/v2_OpenEnded_mscoco_test-dev2015_questions.json"
ANSWERS_DIR="/content/answers"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m $VQA_LOADER \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file $ANSWERS_DIR/${CHUNKS}_${IDX}.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

# Clear out the output file if it exists.
OUTPUT_FILE="$ANSWERS_DIR/merge.json"
> $OUTPUT_FILE

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ANSWERS_DIR/${CHUNKS}_${IDX}.json >> $OUTPUT_FILE
done

# # python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

