# Distributed RLVR Training

This repository contains scripts for distributed Reinforcement Learning with Verifiable Rewards (RLVR) training using the TRL library.

## Overview

The training pipeline consists of two stages:
1. **SFT Warmup**: Supervised fine-tuning on the MBPP dataset
2. **GRPO Training**: Group Relative Policy Optimization on the APPS dataset with BLEU score rewards

## Dataset Processing

### MBPP Dataset
- Original columns: `task_id`, `text`, `code`, `test_list`, `test_setup_code`, `challenge_test_list`
- Processed columns: `prompt` (from `text`), `sample_code` (from `code`)
- Used for SFT warmup

### APPS Dataset
- Original columns: `problem_id`, `question`, `solutions`, `input_output`, `difficulty`, `url`, `starter_code`
- Processed columns: `prompt` (from `question`), `sample_code` (from first solution)
- Used for GRPO training

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt')"

# Login to HuggingFace (for pushing model)
huggingface-cli login
```

## Usage

### Local Multi-GPU Training

```bash
# Simple 4-GPU training
accelerate launch --config_file configs/multi_gpu.yaml train_grpo.py \
    --model_name Qwen/Qwen3-1.7B-Base \
    --per_device_batch_size 1 \
    --sample_ratio 0.01 \
    --push_to_hub \
    --hub_username HuggingFaceAlbert
```

### SLURM Cluster Training

For simple multi-GPU training on a single node:
```bash
sbatch run_simple.slurm
```

For distributed training with vLLM inference server (5 nodes, 20 GPUs):
```bash
sbatch run_training.slurm
```

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | `Qwen/Qwen3-1.7B-Base` | Base model name or path |
| `--hub_username` | `HuggingFaceAlbert` | HuggingFace username for model upload |
| `--sample_ratio` | `0.01` | Fraction of dataset to use (1% for testing) |
| `--per_device_batch_size` | `1` | Training batch size per GPU |
| `--sft_epochs` | `1` | Number of SFT warmup epochs |
| `--skip_sft` | `False` | Skip SFT and go directly to GRPO |
| `--sft_checkpoint` | `None` | Path to existing SFT checkpoint |
| `--use_vllm` | `False` | Use vLLM for generation |
| `--vllm_server_host` | `""` | vLLM server host address |
| `--push_to_hub` | `True` | Push final model to HuggingFace Hub |

## Reward Function

The GRPO training uses BLEU score as the reward function:

```python
from nltk.translate.bleu_score import sentence_bleu

# Reference: ground truth code from sample_code
# Prediction: model-generated code
score = sentence_bleu([reference.split()], prediction.split())
```

## Monitoring

Training is logged to Weights & Biases:
- Entity: `twu376@wisc.edu`
- Project: `mini_rlvr_test`

## File Structure

```
rlvr_training/
├── configs/
│   └── multi_gpu.yaml      # Accelerate config for multi-GPU
├── data_processing.py       # Dataset loading and preprocessing
├── train_grpo.py           # Main training script
├── run_simple.slurm        # SLURM script for single-node training
├── run_training.slurm      # SLURM script for distributed training with vLLM
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Output

After training, the model is saved to:
- SFT checkpoint: `outputs/sft_checkpoint/`
- GRPO model: `outputs/Qwen3-1.7B-GRPO/`
- HuggingFace Hub: `HuggingFaceAlbert/Qwen3-1.7B-GRPO`
