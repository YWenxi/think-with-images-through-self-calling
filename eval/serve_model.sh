#!/bin/bash

# Qwen2.5-VL-7B-Instruct via vllm
# MODEL_NAME=Qwen2.5-VL-7B-Instruct
# MODEL_NAME=SubagentVL-7B-Fine+Chart+Reason-80
MODEL_NAME=YOUR-MODEL-NAME
CKPT_DIR=YOUR-LOCAL-DIR-FOR-MODEL-CKPT

echo "Serving ${MODEL_NAME} via vllm"

vllm serve "${CKPT_DIR}" \
    --port 18901 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 32768 \
    --tensor-parallel-size 1 \
    --served-model-name "${MODEL_NAME}" \
    --trust-remote-code \
    --disable-log-requests \
    --dtype bfloat16