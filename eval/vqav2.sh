#!/bin/bash

# Sourced from: https://github.com/haotian-liu/LLaVA/blob/c121f0432da27facab705978f83c4ada465e46fd/scripts/v1_5/eval/vqav2.sh

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}

MODEL_PATH="liuhaotian/llava-v1.5-13b"
CKPT="llava-v1.5-13b"
CONV_MODE="vicuna_v1"

DATA_DIR="/content/drive/MyDrive/VQA/data"
# #SPLIT="v2_OpenEnded_mscoco_test2015_questions"
SPLIT="v2_OpenEnded_mscoco_test-dev2015_questions"
QUESTION_FILE="$DATA_DIR/Questions/$SPLIT.json"
IMAGE_FOLDER="$DATA_DIR/Images/test2015"

ANSWERS_DIR="/content/drive/MyDrive/VQA/testing_results/answers"
OUTPUT_FILE="$ANSWERS_DIR/$SPLIT/$CKPT/merge.jsonl"

for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m llava.eval.model_vqa_loader \
        --model-path $MODEL_PATH \
        --question-file $QUESTION_FILE \
        --image-folder $IMAGE_FOLDER \
        --answers-file $ANSWERS_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.json \
        --num-chunks $CHUNKS \
        --chunk-idx $IDX \
        --temperature 0 \
        --conv-mode $CONV_MODE &
done

wait

# Clear out the output file if it exists.
> $OUTPUT_FILE

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat $ANSWERS_DIR/$SPLIT/$CKPT/${CHUNKS}_${IDX}.json >> $OUTPUT_FILE
done

# python scripts/convert_vqav2_for_submission.py --split $SPLIT --ckpt $CKPT

