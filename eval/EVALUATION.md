# Evaluation for `SubagentVL`
aka `Thinking with Images via Self-Calling Agent`

## First Look: Implement SubagentVL-like model using Qwen Agent.

Any VLM that could be served as an OpenAI Compatible api could be easily implemented as a self-calling agent to tackle complex visual problems. If you want to take a look at costum VLMs using self-calling CoT at inference time, we provide an easy implementation based on `qwen-agent` and `vllm/sglang`.

1. Make sure you have `qwen-agent` installed
    ```bash
    uv pip install qwen-vl-utils==0.0.14 qwen-agent==0.0.31
    ```
2. Note that the qwen-agent also provide the [typical *thinking-with-images* agent](https://github.com/QwenLM/Qwen-Agent/blob/main/examples/cookbook_think_with_images.ipynb) using zoom-in tools, which is implemented for Qwen3-VL.
3. Based on this example, we make small modification (e.g. not saving intermediate temp images in the reasoning trajectories; transfering the tools to be compatible with Qwen2.5-VL).
4. The tools are implemented in [tool.py](./tools.py) with examples here.

## Evaluation on V* Benchmark

1. Preparation.
    - Download dataset for V* benchmark from [huggingface](https://huggingface.co/datasets/craigwu/vstar_bench).
    - Serve the model to be tested using vLLM or SGLang. For example,
        ```bash 
        vllm serve "${CKPT_DIR}" \
            --port 18901 \
            --gpu-memory-utilization 0.8 \
            --max-model-len 32768 \
            --tensor-parallel-size 1 \
            --served-model-name "${MODEL_NAME}" \
            --trust-remote-code \
            --disable-log-requests \
            --dtype bfloat16
        ```
        An example is provided [here](./serve_model.sh).
2. Evaluate on V*-Bench
3. Calculate the score.
    - Serve an LLM-as-a-Judge using the above serving scripts.
        > [!NOTE]
        > DeepEyes uses Qwen2.5-VL-75B-Instruct as the judge and we tested it with a judge using Qwen2.5-VL-7B-Instruct. There is only a minor difference for the final difference. See results in our paper.
    - Calculate the score.