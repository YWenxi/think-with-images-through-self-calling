<h1 style="text-align: center;">Thinking with Images via Self-Calling Agent</h1>

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2512.08511-b31b1b.svg)](http://arxiv.org/abs/2512.08511)

</div>

## Introduction



## Installation

> [!NOTE]
> This project is developed on top of **`verl@0.5.0.dev`** and the **DeepEyes** training recipe.

We use **`uv`** to set up the environment, though using `conda` or `pip` follows a similar procedure.

```bash
# create and activate environment
uv venv --python=3.12
source .venv/bin/activate

# install PyTorch (CUDA 12.8)
uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# install verl with the sglang specification
uv pip install -e .[sglang]

# additional dependencies
uv pip install debugpy==1.8.0 flash-attn==2.8.3 \
    torch-memory-saver==0.0.9 qwen-vl-utils==0.0.14 qwen-agent==0.0.31
```

A full dependency list is provided in  
[`requirements_subagent.txt`](./requirements_subagent.txt) for reference.

## Reproduce the experiments

### End-to-End RL Training 

- We provide the training scripts to reproduce the training process in [recipe](./recipe/vlm_subagent/).
    - The implementation of subagent calling is following the implementation of tools such as `image_zoom_in_tool`. 
    - Please see the implementation in [`vlm_subagent_tool`](./verl/tools/vlm_subagent_tool.py).


### Evaluation

We evaluate DeepEyes using its official official 

## License

This project is released under [Apache license](LICENSE).

## Citation

```
@misc{yang2025thinkingimagesselfcallingagent,
      title={Thinking with Images via Self-Calling Agent}, 
      author={Wenxi Yang and Yuzhong Zhao and Fang Wan and Qixiang Ye},
      year={2025},
      eprint={2512.08511},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.08511}, 
}
```