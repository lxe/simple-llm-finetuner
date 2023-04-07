---
title: Simple LLM Finetuner
emoji: 🦙
colorFrom: yellow
colorTo: orange
sdk: gradio
app_file: app.py
pinned: false
---

# 🦙 Simple LLM Finetuner

[![Open In Colab](https://img.shields.io/static/v1?label=Open%20in%20Colab&message=Select%20HIGH%20RAM&color=yellow&logo=googlecolab)](https://colab.research.google.com/github/lxe/simple-llama-finetuner/blob/master/Simple_LLaMA_FineTuner.ipynb)
[![Open In Spaces](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)](https://huggingface.co/spaces/lxe/simple-llama-finetuner)
[![](https://img.shields.io/badge/no-bugs-brightgreen.svg)](https://github.com/lxe/no-bugs) 
[![](https://img.shields.io/badge/coverage-%F0%9F%92%AF-green.svg)](https://github.com/lxe/onehundred/tree/master)

Simple LLM Finetuner is a beginner-friendly interface designed to facilitate fine-tuning various language models using [LoRA](https://arxiv.org/abs/2106.09685) method via the [PEFT library](https://github.com/huggingface/peft) on commodity NVIDIA GPUs. With small dataset and sample lengths of 256, you can even run this on a regular Colab Tesla T4 instance.

With this intuitive UI, you can easily manage your dataset, customize parameters, train, and evaluate the model's inference capabilities.

## Acknowledgements

 - https://github.com/zphang/minimal-llama/
 - https://github.com/tloen/alpaca-lora
 - https://github.com/huggingface/peft

## Features

- Simply paste datasets in the UI, separated by double blank lines
- Adjustable parameters for fine-tuning and inference
- Beginner-friendly UI with explanations for each parameter

## Getting Started

### Prerequisites

- Linux or WSL
- Modern NVIDIA GPU with >= 16 GB of VRAM (but it might be possible to run with less for smaller sample lengths)

### Usage

I recommend using a virtual environment to install the required packages. Conda preferred.

```
conda create -n simple-llm-finetuner python=3.10
conda activate simple-llm-finetuner
conda install -y cuda -c nvidia/label/cuda-11.7.0
conda install -y pytorch=2 pytorch-cuda=11.7 -c pytorch
```

On WSL, you might need to install CUDA manually by following [these steps](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local), then running the following before you launch:

```
export LD_LIBRARY_PATH=/usr/lib/wsl/lib
```

Clone the repository and install the required packages.

```
git clone https://github.com/lxe/simple-llm-finetuner.git
cd simple-llm-finetuner
pip install -r requirements.txt
```

Launch it

```
python main.py
```

Open http://127.0.0.1:7860/ in your browser. Prepare your training data by separating each sample with 2 blank lines. Paste the whole training dataset into the textbox. Specify the new LoRA adapter name in the "New PEFT Adapter Name" textbox, then click train. You might need to adjust the max sequence length and batch size to fit your GPU memory. The model will be saved in the `lora/` directory.

After training is done, navigate to "Inference" tab, select your LoRA, and play with it.

Have fun!

## YouTube Walkthough

https://www.youtube.com/watch?v=yM1wanDkNz8

## License

MIT License
