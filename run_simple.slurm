#!/bin/bash
#SBATCH --job-name=rlvr_simple
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rlvr_simple_%j.out
#SBATCH --error=logs/rlvr_simple_%j.err

# ============================================================================
# Simple Multi-GPU RLVR Training (Single Node, 4 GPUs)
# 
# For small models like Qwen3-1.7B, no need for FSDP.
# Uses data parallelism for fast training.
# ============================================================================

# Create log directory
mkdir -p logs

# Load required modules
module load cuda/12.1
module load python/3.10

# Activate virtual environment
source ~/venv/bin/activate

# Set environment variables
export WANDB_PROJECT="mini_rlvr_test"
export WANDB_ENTITY="twu376@wisc.edu"
export HF_TOKEN="${HF_TOKEN:-}"
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Run training with accelerate
accelerate launch \
    --config_file configs/multi_gpu.yaml \
    train_grpo.py \
    --model_name Qwen/Qwen3-1.7B-Base \
    --per_device_batch_size 1 \
    --sample_ratio 0.01 \
    --push_to_hub \
    --hub_username HuggingFaceAlbert \
    --output_dir ./outputs

echo "Training completed!"
