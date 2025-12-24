# Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography

## Introduction

This is the official implementation for our paper "[Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography](https://arxiv.org/abs/2512.20168v1)". This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026. If you find this repository useful for your research, please consider citing our paper. Any questions or issues are welcome to be raised via GitHub Issues or email (jmengcheng@163.com).

## Overview

![framework](./figures/pipline.png)

## Getting Started

### 1. Install Dependencies

```
conda create -n Odysseus python=3.10
conda activate Odysseus
pip install -r requirements.txt
```

### 2. Prepare Datasets and Evaluation Model

We use [COCO 2017](https://cocodataset.org/#download) as the image dataset for all experiments. Please download the dataset in advance and place it in the `./data` directory. After downloading, run the following command to preprocess and index the dataset:

```
python ./data/index.py
```

For safety evaluation, we adopt [HarmBench](https://huggingface.co/cais/HarmBench-Llama-2-13b-cls), a widely used benchmark for harmful content classification. You can download it from Hugging Face using the following command:

```
huggingface-cli download --resume-download cais/HarmBench-Llama-2-13b-cls --local-dir /home/model/HarmBench-Llama-2-13b-cls
```

### 3. Running

#### Training the Steganography Model

First, train the steganography model used to embed malicious queries into images:

```
python train.py new --name stego_model
```

**Available Commands:**

**`new`** - Start a new training run:

- `--name` (required): Name of the experiment
- `--data-dir`, `-d`: Data directory (default: `./data`)
- `--batch-size`, `-b`: Batch size (default: 10)
- `--epochs`, `-e`: Number of epochs (default: 300)
- `--continue-from-folder`, `-c`: Resume from previous checkpoint folder
- `--tensorboard`: Enable Tensorboard logging
- `--enable-fp16`: Enable mixed-precision training
- `--noise`: Configure noise layers, e.g., `'cropout((0.55, 0.6), (0.55, 0.6))'`

**`continue`** - Resume a previous training run:

```
python train.py continue --folder [checkpoint_folder]
```

- `--folder`, `-f` (required): Path to checkpoint folder
- `--data-dir`, `-d`: Override previous data directory
- `--epochs`, `-e`: Override previous epoch count


#### Launching the Jailbreak Attack

Before launching the attack, please modify the configuration file located at `./utils/config.json`. In particular:

- Set `harmbench-path` to the local path of the downloaded HarmBench model (e.g., `/home/model/HarmBench-Llama-2-13b-cls`).
- Set `stego-folder` to the directory containing the trained steganography model.
- Configure the API key of the target MLLM-integrated system for evaluation and testing.

After completing the above steps, execute the following command to perform the jailbreak attack:

```
python main.py
```

This will automatically generate steganographic images, submit them to the target MLLM-integrated system, and evaluate the responses using the HarmBench classifier.

**Available Commands:**

- `--dataset`: Specifies the benchmark dataset used to construct malicious queries.

- `--model`: Specifies the target MLLM-integrated system to be attacked. Supported models include:

  - `gpt-4o`
  - `grok-3`
  - **Gemini 2.0 series**, including `gemini-2.0-pro-exp-02-05` and `gemini-2.0-flash-exp`
  - **Doubao-seed-1.6 series**, including`doubao-seed-1-6-251015`, `doubao-seed-1-6-lite-251015`, `doubao-seed-1-6-flash-250828`, `doubao-seed-1-6-vision-250815`, `doubao-seed-1-6-thinking-250715`, `doubao-seed-code-preview-251028`
  - **Qwen3-VL series**, including `qwen3-vl-plus`, `qwen3-vl-flash`
  - **GLM-4.6V series**, including `glm-4.6v`, `glm-4.6v-flashx`, `glm-4.6v-flash`

  More generally, **other commercial MLLM-integrated systems equipped with function-calling capabilities can be evaluated by appropriately configuring this argument**.

- `--try_times`: Specifies the number of repeated attack attempts for each malicious query.

- `--cover_image`: Specifies the cover image used for steganographic embedding.

## Main Results

![main_results](./figures/main_res.png)

## Citation Info

If you find this repository useful for your research, it will be greatly appreciated to cite our paper😄.

```
@inproceedings{li2025odysseus,
    title={Odysseus: Jailbreaking Commercial Multimodal LLM-integrated Systems via Dual Steganography},
    author={Li, Songze and Cheng, Jiameng and Li, Yiming and Jia, Xiaojun and Tao, Dacheng},
    booktitle={Network and Distributed System Security Symposium},
    year={2026}
}
```