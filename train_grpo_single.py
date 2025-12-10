"""
Single-Machine RLVR Training Script
1. SFT warmup with MBPP dataset
2. GRPO training with APPS dataset using BLEU score reward

Run with: python train_single_machine.py [args]
For multi-GPU: torchrun --nproc_per_node=<num_gpus> train_single_machine.py [args]
"""

import argparse
import os
import logging
from typing import List

import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data_processing import process_mbpp_dataset, process_apps_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Reward Function: BLEU Score
# ============================================================================

def compute_bleu_reward(completions: List[str], references: List[str], **kwargs) -> List[float]:
    """
    Compute BLEU score reward for generated completions.
    
    Args:
        completions: List of generated code completions (model outputs)
        references: List of reference solutions (ground truth completion)
    
    Returns:
        List of BLEU scores as rewards
    """
    rewards = []
    smoother = SmoothingFunction()
    
    for completion, reference in zip(completions, references):
        # Tokenize by splitting on whitespace and special characters
        pred_tokens = completion.split()
        ref_tokens = [reference.split()]  # BLEU expects list of reference lists
        
        if len(pred_tokens) == 0:
            rewards.append(0.0)
            continue
        
        try:
            # Calculate unweighted BLEU score (default weights)
            score = sentence_bleu(
                ref_tokens,
                pred_tokens,
                smoothing_function=smoother.method1  # Avoid zero scores for short sequences
            )
            rewards.append(float(score))
        except Exception as e:
            logger.warning(f"BLEU computation failed: {e}")
            rewards.append(0.0)
    
    return rewards


def bleu_reward_function(completions, prompts=None, **kwargs):
    """
    TRL-compatible reward function wrapper.
    
    This function is called by GRPOTrainer with generated completions.
    We need to extract the completion from the dataset to use as reference.
    """
    # Extract text content from completions
    if isinstance(completions[0], list):
        # Handle batch format: [[{"content": "..."}], ...]
        completion_texts = []
        for comp in completions:
            if isinstance(comp, list) and len(comp) > 0:
                if isinstance(comp[0], dict):
                    completion_texts.append(comp[0].get("content", ""))
                else:
                    completion_texts.append(str(comp[0]))
            else:
                completion_texts.append(str(comp))
    else:
        completion_texts = [str(c) for c in completions]
    
    # Get references from kwargs if available
    references = kwargs.get("completion", [""] * len(completion_texts))
    
    rewards = compute_bleu_reward(completion_texts, references)
    return [torch.tensor(r) for r in rewards]


# ============================================================================
# Training Functions
# ============================================================================

def run_sft_warmup(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
    per_device_batch_size: int = 1,
    num_epochs: int = 1,
    local_rank: int = -1,
):
    """
    Run SFT warmup training on MBPP dataset.
    
    Args:
        model_name: Base model name/path
        dataset: Processed MBPP dataset with 'prompt' and 'completion' columns
        output_dir: Directory to save the model
        per_device_batch_size: Training batch size per device
        num_epochs: Number of training epochs
        local_rank: Local rank for distributed training
    """
    logger.info("=" * 50)
    logger.info("Starting SFT Warmup Training")
    logger.info("=" * 50)
    
    # Format dataset for SFT: combine prompt and completion
    def format_for_sft(example):
        return {
            "text": f"### Instruction:\n{example['prompt']}\n\n### Response:\n{example['completion']}"
        }
    
    formatted_dataset = dataset.map(format_for_sft)
    
    # SFT Training Config
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        num_train_epochs=num_epochs,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="wandb" if local_rank <= 0 else "none",
    )
    
    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model_name,
        args=sft_config,
        train_dataset=formatted_dataset,
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    logger.info(f"SFT model saved to {output_dir}")
    
    return output_dir


def run_grpo_training(
    model_name: str,
    dataset: Dataset,
    output_dir: str,
    per_device_batch_size: int = 1,
    local_rank: int = -1,
    use_vllm: bool = False,
    push_to_hub: bool = True,
    hub_username: str = "HuggingFaceAlbert",
):
    """
    Run GRPO training on APPS dataset with BLEU reward.
    
    Args:
        model_name: Model name/path (either base or SFT checkpoint)
        dataset: Processed APPS dataset with 'prompt' and 'completion' columns
        output_dir: Directory to save the model
        per_device_batch_size: Training batch size per device
        local_rank: Local rank for distributed training
        use_vllm: Whether to use vLLM for generation (local only)
        push_to_hub: Whether to push model to HuggingFace Hub
        hub_username: HuggingFace username for pushing
    """
    logger.info("=" * 50)
    logger.info("Starting GRPO Training")
    logger.info("=" * 50)
    
    # GRPO Training Config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_batch_size,
        save_strategy="epoch",
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="wandb" if local_rank <= 0 else "none",
        use_vllm=use_vllm,
        push_to_hub=push_to_hub,
        hub_model_id=f"{hub_username}/Qwen3-1.7B-GRPO" if push_to_hub else None,
    )
    
    # Initialize GRPO Trainer
    trainer = GRPOTrainer(
        model=model_name,
        reward_funcs=bleu_reward_function,
        args=grpo_config,
        train_dataset=dataset,
    )
    
    # Train
    trainer.train()
    
    # Save the model
    trainer.save_model(output_dir)
    logger.info(f"GRPO model saved to {output_dir}")
    
    # Push to hub
    if push_to_hub and local_rank <= 0:
        logger.info(f"Pushing model to HuggingFace Hub: {hub_username}/Qwen3-1.7B-GRPO")
        trainer.push_to_hub()
    
    return output_dir


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Single-Machine RLVR Training")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-1.7B-Base",
                        help="Base model name or path")
    parser.add_argument("--hub_username", type=str, default="HuggingFaceAlbert",
                        help="HuggingFace username for pushing model")
    
    # Data arguments
    parser.add_argument("--sample_ratio", type=float, default=0.01,
                        help="Fraction of dataset to use (default: 1%)")
    
    # Training arguments
    parser.add_argument("--per_device_batch_size", type=int, default=1,
                        help="Training batch size per device")
    parser.add_argument("--sft_epochs", type=int, default=1,
                        help="Number of SFT warmup epochs")
    parser.add_argument("--skip_sft", action="store_true",
                        help="Skip SFT warmup and go directly to GRPO")
    parser.add_argument("--sft_checkpoint", type=str, default=None,
                        help="Path to existing SFT checkpoint (skips SFT training)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Base output directory")
    parser.add_argument("--push_to_hub", action="store_true", default=True,
                        help="Push final model to HuggingFace Hub")
    
    # vLLM arguments
    parser.add_argument("--use_vllm", action="store_true",
                        help="Use vLLM for generation (local only)")
    
    # Distributed training arguments
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank for distributed training")
    
    args = parser.parse_args()
    
    # Get local rank from environment if not provided
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", -1))
    
    # Initialize wandb only on main process
    if args.local_rank <= 0:
        wandb.init(
            entity="twu376-uw-madison",
            project="mini_rlvr_test",
            config=vars(args),
        )
    
    # Create output directories
    sft_output_dir = os.path.join(args.output_dir, "sft_checkpoint")
    grpo_output_dir = os.path.join(args.output_dir, "Qwen3-1.7B-GRPO")
    
    os.makedirs(sft_output_dir, exist_ok=True)
    os.makedirs(grpo_output_dir, exist_ok=True)
    
    # ========================================================================
    # Step 1: Load and process datasets
    # ========================================================================
    logger.info("Loading and processing datasets...")
    
    mbpp_dataset = process_mbpp_dataset(sample_ratio=args.sample_ratio)
    apps_dataset = process_apps_dataset(sample_ratio=args.sample_ratio)
    
    logger.info(f"MBPP dataset size: {len(mbpp_dataset)}")
    logger.info(f"APPS dataset size: {len(apps_dataset)}")
    
    # ========================================================================
    # Step 2: SFT Warmup with MBPP
    # ========================================================================
    if args.sft_checkpoint:
        # Use existing checkpoint
        sft_model_path = args.sft_checkpoint
        logger.info(f"Using existing SFT checkpoint: {sft_model_path}")
    elif args.skip_sft:
        # Skip SFT, use base model directly
        sft_model_path = args.model_name
        logger.info("Skipping SFT warmup, using base model for GRPO")
    else:
        # Run SFT warmup
        sft_model_path = run_sft_warmup(
            model_name=args.model_name,
            dataset=mbpp_dataset,
            output_dir=sft_output_dir,
            per_device_batch_size=args.per_device_batch_size,
            num_epochs=args.sft_epochs,
            local_rank=args.local_rank,
        )
    
    # ========================================================================
    # Step 3: GRPO Training with APPS
    # ========================================================================
    run_grpo_training(
        model_name=sft_model_path,
        dataset=apps_dataset,
        output_dir=grpo_output_dir,
        per_device_batch_size=args.per_device_batch_size,
        local_rank=args.local_rank,
        use_vllm=args.use_vllm,
        push_to_hub=args.push_to_hub,
        hub_username=args.hub_username,
    )
    
    # Finish wandb
    if args.local_rank <= 0:
        wandb.finish()
    
    logger.info("Training completed successfully!")


if __name__ == "__main__":
    main()