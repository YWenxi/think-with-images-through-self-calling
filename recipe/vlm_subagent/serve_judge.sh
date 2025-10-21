export CUDA_VISIBLE_DEVICES=7

python -m sglang.launch_server --model-path $HOME/Workspace/subagent-rl/pretrained_models/Qwen/Qwen2.5-VL-7B-Instruct \
    --port 18901 \
    --tp-size 1 \
    --context-length 32768 \
    --trust-remote-code \