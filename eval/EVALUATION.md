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