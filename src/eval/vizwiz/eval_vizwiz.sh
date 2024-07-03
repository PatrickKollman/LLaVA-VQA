#!/bin/bash

# Sourced from: https://github.com/haotian-liu/LLaVA/blob/main/scripts/v1_5/eval/vizwiz.sh

python -m eval.vizwiz.eval_vizwiz \
    --model-path liuhaotian/llava-v1.5-13b \
    --question-file /content/train.json \
    --image-folder /content/test \
    --answers-file /content/answers/llava-v1.5-13b.json \
    --temperature 0 \
    --conv-mode vicuna_v1

