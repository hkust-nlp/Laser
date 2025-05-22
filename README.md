# Laser

<p align="center">
  :hugs: <a href="https://huggingface.co/collections/hkust-nlp/laser-682c7d44f347ac572ec054d3">HF Repo</a>&nbsp;&nbsp;&nbsp;
  :page_with_curl: <a href="">Paper</a>
</p>

This repo contains the resources (**Code**, **Data**, **Models**) for the paper "Laser: Learn to Reason Efficiently with Adaptive Length-based Reward Shaping"

Laser (**L**ength-b**A**sed **S**t**E**p **R**eward shaping) and its adaptive versions Laser-D, Laser-DE ( **Dynamic** and **D**ifficulty-aware **L**ength-b**A**sed **S**t**E**p **R**eward shaping) are three novel methods to successfully improve both the effectiveness and efficiency of reasoning. Laser-D and Laser-DE achieve a **6.1** improvement on AIME2024 while reducing token usage by **63\%**.

<p align="center">
  <img src="assets/main_figure.png" alt="Laser main figure">
</p> 

## Table of Contents

- [Laser](#laser)
  - [Table of Contents](#table-of-contents)
  - [News](#news)
  - [Introduction](#introduction)
    - [Unified Framework for Length-based Reward Shaping](#unified-framework-for-length-based-reward-shaping)
  - [Performance](#performance)
    - [Efficacy-Efficiency Trade-off](#efficacy-efficiency-trade-off)
  - [:rocket: Resources](#rocket-resources)
    - [Datasets](#datasets)
    - [Models](#models)
      - [1.5B Models (Based on DeepSeek-R1-Distill-Qwen-1.5B)](#15b-models-based-on-deepseek-r1-distill-qwen-15b)
        - [Laser Models](#laser-models)
        - [Laser-D Models](#laser-d-models)
        - [Laser-DE Models](#laser-de-models)
      - [7B Models (Based on DeepSeek-R1-Distill-Qwen-7B)](#7b-models-based-on-deepseek-r1-distill-qwen-7b)
  - [How to Start :running:?](#how-to-start-running)
    - [Installation](#installation)
    - [Training](#training)
    - [Evaluation](#evaluation)
  - [Citation](#citation)


## News

- :fire: [05/2025] We are excited to release the resources for the paper "Laser: Learn to Reason Efficiently with Adaptive Length-based Reward Shaping"

## Introduction

In Laser, we propose a **unified view** for length-based reward shaping, unifying various reward-shaping and truncation methods. Building on this view, we propose a novel **L**ength-b**A**sed **S**t**E**p **R**eward shaping method (**Laser**), which employs a step reward function based on target length. We further propose the adaptive version of Laser, **Laser-D** and **Laser-DE**, based on two key intuitions: 

1. The reasoning behavior of the model evolves dynamically during training, necessitating reward specifications that are also adaptive and dynamic; 

2. Rather than uniformly encouraging shorter or longer chains of thought (CoT), we posit that length-based reward shaping should be difficulty-aware i.e., it should penalize lengthy CoTs more for easy queries. 

This approach facilitates a combination of fast and slow thinking, leading to a better overall tradeoff. Unlike methods that improve token efficiency at the expense of accuracy, our proposed approaches deliver substantial gains in both dimensions—even on the challenging AIME2024 benchmark.

### Unified Framework for Length-based Reward Shaping

We propose a unified framework for length-based reward shaping, unifying various reward-shaping and truncation methods. More details can be found in our [paper](), section 4.

<p align="center">
  <img src="assets/unified_framework.png" alt="Unified Framework for Length-based Reward Shaping">
</p>


## Performance

### Efficacy-Efficiency Trade-off
Efficacy (accuracy) and efficiency (token efficiency) are actually two conflicting goals. The goal of RL-based CoT compression should be to find a better balance between the two and improve both.

Each point in the following figures represents an independent experiment, obtained through different training runs with different parameter configurations. Benchmarks consist of MATH500, AIME2024, AMC2023, and Olympiad Bench.

<p align="center">
  <img src="assets/average_performance.jpg" alt="Average Performance" width="48%">
  <img src="assets/average_aime.jpg" alt="Average AIME Performance" width="48%">
</p>

## :rocket: Resources

### Datasets
| Dataset Name | Description | Link |
|:------------:|:------------|:----:|
| **Laser-Deepscaler-Dataset** | Training dataset | [🤗 HuggingFace](https://huggingface.co/datasets/hkust-nlp/Laser-Deepscaler-Dataset) |

### Models

#### 1.5B Models (Based on [DeepSeek-R1-Distill-Qwen-1.5B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B))

##### Laser Models
| Model Name | Adaptive Target Length (L) | Size | Link |
|:----------:|:--------------------------:|:----:|:----:|
| **Laser-L2048** | 2048 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-L2048-1.5B) |
| **Laser-L4096** | 4096 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-L4096-1.5B) |
| **Laser-L8192** | 8192 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-L8192-1.5B) |

##### Laser-D Models
| Model Name | Adaptive Target Length (L) | Size | Link |
|:----------:|:--------------------------:|:----:|:----:|
| **Laser-D-L1024** | 1024 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-D-L1024-1.5B) |
| **Laser-D-L2048** | 2048 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-D-L2048-1.5B) |
| **Laser-D-L4096** | 4096 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-D-L4096-1.5B) |

##### Laser-DE Models
| Model Name | Adaptive Target Length (L) | Size | Link |
|:----------:|:--------------------------:|:----:|:----:|
| **Laser-DE-L1024** | 1024 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-DE-L1024-1.5B) |
| **Laser-DE-L2048** | 2048 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-DE-L2048-1.5B) |
| **Laser-DE-L4096** | 4096 | 1.5B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-DE-L4096-1.5B) |

#### 7B Models (Based on [DeepSeek-R1-Distill-Qwen-7B](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-7B))

| Model Name | Adaptive Target Length (L) | Size | Link |
|:----------:|:--------------------------:|:----:|:----:|
| **Laser-D-L4096** | 4096  | 7B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-D-L4096-7B) |
| **Laser-DE-L4096** | 4096 | 7B | [🤗 HuggingFace](https://huggingface.co/hkust-nlp/Laser-DE-L4096-7B) |


> **Note**: Smaller value of $L$ indicates more rapid compression during training, resulting in more concise Chains of Thought (CoTs) during inference.

## How to Start :running:?
### Installation

```bash
conda create -n laser python=3.10
git clone https://github.com/hkust-nlp/Laser.git

pip install -r requirement.txt
pip install flash-attn==2.6.3 --no-build-isolation
pip install -e . --no-dependencies
```

### Data Preparation

```bash
python scripts/pull_from_hub.py --repo_id hkust-nlp/Laser-Deepscaler-Dataset --local_path ./data/deepscaler --repo_type dataset --ignore_patterns "global_step*"
```

or you can download the dataset from [🤗 HuggingFace](https://huggingface.co/datasets/hkust-nlp/Laser-Deepscaler-Dataset) and put it in the `data/deepscaler` folder.

### Training

If you use slurm to run the training with ray, you can use the following command:

```bash
bash scripts/example/ray_start_slurm.sh $SCRIPT

# e.g. bash scripts/example/ray_start_slurm.sh scripts/training/laser-de-1.5b/laser-de-1.5B-l4096.sh
```

Otherwise, you can use the following command to run the training with ray:
```bash
bash scripts/example/ray_start_sh.sh $SCRIPT
```

SCRIPT is the script you want to run, for example, `scripts/training/laser-de-1.5b/laser-de-1.5B-l4096.sh`.
```bash
# Laser
scripts/training/laser-1.5b/laser-1.5b-l2048.sh
scripts/training/laser-1.5b/laser-1.5b-l4096.sh
scripts/training/laser-1.5b/laser-1.5b-l8192.sh

# Laser-D
scripts/training/laser-d-1.5b/laser-d-1.5b-l1024.sh
scripts/training/laser-d-1.5b/laser-d-1.5b-l2048.sh
scripts/training/laser-d-1.5b/laser-d-1.5b-l4096.sh

# Laser-DE
scripts/training/laser-de-1.5b/laser-de-1.5b-l1024.sh
scripts/training/laser-de-1.5b/laser-de-1.5b-l2048.sh
scripts/training/laser-de-1.5b/laser-de-1.5b-l4096.sh
```

### Evaluation

```bash

RUNNAME=""
INIT_MODEL_PATH=""  # path to the init model, or any hf model path
TPSIZE=1
STEPS="" # if empty, init model will be evaluated

bash Qwen2.5-Math/evaluation/sh/nodes/run_eval.sh $RUNNAME $INIT_MODEL_PATH $TPSIZE $STEPS
```

## Citation
If you find the content of this project helpful, please cite our paper as follows:

```
@misc{liu2025learnreasonefficientlyadaptive,
      title={Learn to Reason Efficiently with Adaptive Length-based Reward Shaping}, 
      author={Wei Liu and Ruochen Zhou and Yiyun Deng and Yuzhen Huang and Junteng Liu and Yuntian Deng and Yizhe Zhang and Junxian He},
      year={2025},
      eprint={2505.15612},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.15612}, 
}
```

## Acknowledgements

- As a sister project of [SimpleRL](https://github.com/hkust-nlp/simpleRL-reason), we would like to thank the authors of [SimpleRL](https://github.com/hkust-nlp/simpleRL-reason) for their great work.
- Our code is built on the great work of [verl](https://github.com/volcengine/verl).