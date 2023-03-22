# 🦙 Simple LLaMA Finetuner

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/lxe/simple-llama-finetuner/blob/master/Simple_LLaMA_FineTuner.ipynb)
[![](https://img.shields.io/badge/no-bugs-brightgreen.svg)](https://github.com/lxe/no-bugs) 
[![](https://img.shields.io/badge/coverage-%F0%9F%92%AF-green.svg)](https://github.com/lxe/onehundred/tree/master)

Simple LLaMA Finetuner is a beginner-friendly interface designed to facilitate fine-tuning the [LLaMA-7B](https://github.com/facebookresearch/llama) language model using [LoRA](https://arxiv.org/abs/2106.09685) method via the [PEFT library](https://github.com/huggingface/peft) on commodity NVIDIA GPUs. With small dataset and sample lengths of 256, you can even run this on a regular Colab Tesla T4 instance.

With this intuitive UI, you can easily manage your dataset, customize parameters, train, and evaluate the model's inference capabilities.

## Acknowledgements

 - https://github.com/zphang/minimal-llama/
 - https://github.com/tloen/alpaca-lora
 - https://github.com/huggingface/peft
 - https://huggingface.co/datasets/Anthropic/hh-rlhf

## Features

- Simply paste datasets in the UI, separated by double blank lines
- Adjustable parameters for fine-tuning and inference
- Begner-friendly UI with explanations for each parameter

## TODO

- [ ] Accelerate / DeepSpeed 
- [ ] Load other models
- [ ] More dataset preparation tools

## Getting Started

### Prerequisites

- Linux or WSL
- Modern NVIDIA GPU with >16 GB of VRAM (but it might be possible to run with less for smaller sample lengths)

### Usage

I recommend using a virtual environment to install the required packages. Conda preferred

```
conda install -y cuda -c nvidia/label/cuda-11.7.0
conda install -y pytorch=1.13.1 pytorch-cuda=11.7 -c pytorch
```

Clone the repository and install the required packages.

```
git clone https://github.com/lxe/simple-llama-finetuner.git
cd simple-llama-finetuner
pip install -r requirements.txt
```

Launch it

```
python main.py
```

Open http://127.0.0.1:7860/ in your browser. Prepare your training data by separating each sample with 2 blank lines. Paste the whole training dataset into the textbox. Specify the model name in the "LoRA Model Name" textbox, then click train. You might need to adjust the max sequence length and batch size to fit your GPU memory. The model will be saved in the `lora-{your model name}` directory.

After training is done, navigate to "Inference" tab, click "Reload Models", select your model, and play with it.

Have fun!

## Screenshots

|![Image1](https://user-images.githubusercontent.com/1486609/226793136-84531388-4081-49bb-b982-3f47e6ec25cd.png) | ![Image2](https://user-images.githubusercontent.com/1486609/226809466-b1eb6f3f-4049-4a41-a2e3-52b06a6e1230.png) |
|:---:|:---:|

## License

MIT License

Copyright (c) 2023 Aleksey Smolenchuk

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
