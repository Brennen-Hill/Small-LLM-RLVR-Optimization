"""
Single-Machine RLVR Training Script
1. SFT warmup with MBPP dataset
2. GRPO training with APPS dataset using BLEU score reward

Run with: python train_single_machine.py [args]
For multi-GPU: torchrun --nproc_per_node=<num_gpus> train_single_machine.py [args]
"""

import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Enable expandable segments to help mitigate CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import logging
from typing import List

import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig, GRPOConfig, GRPOTrainer
import time

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from code_structure_reward import unit_test_reward_function, code_similarity_reward_function, compile_check_reward_function

from data_processing_optimized import process_mbpp_dataset, process_apps_dataset, process_apps_cccs_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

NUM_GENERATIONS = 2

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


def bleu_reward_function(completions, ground_truth, prompts=None, **kwargs):
    """
    BLEU reward function hardcoded for:
      - APPS dataset processed into {'prompt', 'ground_truth'} => here passed as ground_truth
      - completions may come in different TRL formats:
          [["text"]], ["text"], [{"content": "text"}], [[{"content": "text"}]]
    """

    # ----------------------------------------------------
    # 1. Extract generated completions (robust)
    # ----------------------------------------------------
    def extract_text(c):
        # Case A: [["text"]] → take c[0] → "text"
        if isinstance(c, list):
            if len(c) == 0:
                return ""
            first = c[0]

            # Case A1: [{"content": "text"}]
            if isinstance(first, dict):
                return first.get("content", "")

            # Case A2: ["text"]
            if isinstance(first, str):
                return first

            # Case A3: nested lists like [[{"content": ...}]]
            return extract_text(first)

        # Case B: {"content": "text"}
        if isinstance(c, dict):
            return c.get("content", "")

        # Case C: already a raw string
        return str(c)

    completion_texts = [extract_text(c) for c in completions]

    # ----------------------------------------------------
    # 2. Extract references from dataset
    # ----------------------------------------------------
    # ground_truth is already a list of reference strings
    references_raw = ground_truth  # ex: ["solution1", "solution2", ...]

    # ----------------------------------------------------
    # 3. Expand references to match flattened completions
    # ----------------------------------------------------
    expanded_refs = []
    for ref in references_raw:
        expanded_refs.extend([ref] * NUM_GENERATIONS)

    # Truncate to match completions exactly
    expanded_refs = expanded_refs[:len(completion_texts)]

    # ----------------------------------------------------
    # 4. Compute BLEU rewards
    # ----------------------------------------------------
    rewards = compute_bleu_reward(completion_texts, expanded_refs)

    # ----------------------------------------------------
    # 5. Return torch tensors
    # ----------------------------------------------------
    return [torch.tensor(r, dtype=torch.float32) for r in rewards]


def reward_disagreement_penalty_function(completions, ground_truth, prompts=None, **kwargs):
    """
    Penalty reward function that penalizes when similarity-based rewards
    (BLEU + structural similarity) diverge too much from functional-based rewards
    (compile check + unit test heuristic).
    
    This prevents the model from:
    - Generating code that looks similar but doesn't work (high similarity, low functional)
    - Generating working code that's completely different from expected (high functional, low similarity)
    
    Similarity-based (how much it looks like the reference):
        - bleu_similarity
        - code_similarity (structural)
    
    Functional-based (how likely it is to work):
        - compile_check
        - unit_test_heuristic
    
    Reward Scale:
        +0.0: No divergence (similarity and functional scores agree)
        -0.5: Moderate divergence (difference > 0.5)
        -1.0: High divergence (difference > 1.0)
    
    Args:
        completions: Model generations in various TRL formats
        ground_truth: Reference solutions
        prompts: Problem descriptions
        **kwargs: Additional arguments including:
            - inputs: Test inputs for heuristic hints
            - outputs: Test outputs for heuristic hints
            - bleu_reward_fn: BLEU reward function (required for full comparison)
            - divergence_threshold: When penalty starts (default 0.5)
            - max_penalty: Maximum penalty value (default 1.0)
            - similarity_weight_bleu: Weight for BLEU in similarity score (default 0.4)
            - similarity_weight_structural: Weight for structural in similarity score (default 0.6)
            - functional_weight_compile: Weight for compile in functional score (default 0.4)
            - functional_weight_heuristic: Weight for heuristic in functional score (default 0.6)
    
    Returns:
        List of torch tensors containing penalty values for each completion
    """
    # ----------------------------------------------------
    # 1. Compute all reward components
    # ----------------------------------------------------
    # Functional-based rewards
    compile_rewards = compile_check_reward_function(completions, ground_truth, prompts, **kwargs)
    heuristic_rewards = unit_test_reward_function(completions, ground_truth, prompts, **kwargs)
    
    # Similarity-based rewards
    structural_rewards = code_similarity_reward_function(completions, ground_truth, prompts, **kwargs)
    bleu_rewards = bleu_reward_function(completions, ground_truth, prompts, **kwargs)
    
    # ----------------------------------------------------
    # 2. Get configurable parameters
    # ----------------------------------------------------
    divergence_threshold = kwargs.get('divergence_threshold', 0.5)
    max_penalty = kwargs.get('max_penalty', 1.0)
    
    # Weights for combining into similarity vs functional scores
    sim_weight_bleu = kwargs.get('similarity_weight_bleu', 0.4)
    sim_weight_structural = kwargs.get('similarity_weight_structural', 0.6)
    func_weight_compile = kwargs.get('functional_weight_compile', 0.4)
    func_weight_heuristic = kwargs.get('functional_weight_heuristic', 0.6)
    
    # ----------------------------------------------------
    # 3. Calculate divergence penalty for each completion
    # ----------------------------------------------------
    penalties = []
    
    for i in range(len(compile_rewards)):
        # Functional score (how likely to work)
        c = compile_rewards[i].item()
        h = heuristic_rewards[i].item()
        functional_score = func_weight_compile * c + func_weight_heuristic * h
        
        # Similarity score (how much it looks like reference)
        b = bleu_rewards[i].item()
        s = structural_rewards[i].item()
        similarity_score = sim_weight_bleu * b + sim_weight_structural * s
        
        # Calculate absolute divergence between the two composite scores
        divergence = abs(functional_score - similarity_score)
        
        # Penalty kicks in when divergence exceeds threshold
        # Linear scaling: 0 at threshold, -max_penalty at threshold + 1.0
        if divergence <= divergence_threshold:
            penalty = 0.0
        else:
            excess = divergence - divergence_threshold
            penalty = -min(max_penalty, excess * max_penalty)
        
        penalties.append(penalty)
    
    # ----------------------------------------------------
    # 4. Return torch tensors
    # ----------------------------------------------------
    return [torch.tensor(p, dtype=torch.float32) for p in penalties]


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

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    
    # Initialize SFT Trainer
    trainer = SFTTrainer(
        model=model_name,
        processing_class=tokenizer,
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
    num_generations: int = NUM_GENERATIONS,
    per_device_batch_size: int = 2,
    local_rank: int = -1,
    use_vllm: bool = False,
    push_to_hub: bool = True,
    hub_username: str = "HuggingFaceAlbert",
    baseline: str = "grpo"
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

    hub_model_id = f"{hub_username}/Qwen3-1.7B-{baseline}-{int(time.time())}"
    # GRPO Training Config
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        num_generations=num_generations,
        per_device_train_batch_size=per_device_batch_size,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        report_to="wandb" if local_rank <= 0 else "none",
        use_vllm=use_vllm,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id if push_to_hub else None,

        save_strategy="steps",
        save_steps=200,
        save_total_limit=1,

        reward_weights=[0.25, 0.30, 0.20, 0.15, 0.10],

        # conservative optimization
        learning_rate=0.98e-6,
        max_grad_norm=0.8,

        # GRPO core
        temperature=0.8,

        # trust region
        epsilon=0.18,

        # mild KL regularization
        beta=0.01,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )

    # Initialize GRPO Trainer
    trainer = GRPOTrainer(
        model=model_name,
        processing_class=tokenizer,
        reward_funcs=[compile_check_reward_function, unit_test_reward_function, code_similarity_reward_function, bleu_reward_function, reward_disagreement_penalty_function],
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
        logger.info(f"Pushing model to HuggingFace Hub: {hub_model_id}")
        trainer.push_to_hub()
    
    return output_dir


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    torch.cuda.empty_cache()
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
    parser.add_argument("--baseline", type=str, default="grpo",
                        help="Name of the baseline")
    parser.add_argument("--per_device_batch_size", type=int, default=2,
                        help="Training batch size per device")
    NUM_GENERATIONS = 4
    parser.add_argument("--num_generations", type=int, default=NUM_GENERATIONS,
                    help="Number of generation per input")
    parser.add_argument("--sft_epochs", type=int, default=5,
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
            project="mini_rlvr",
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
    apps_cccs_dataset = process_apps_cccs_dataset(sample_ratio=args.sample_ratio)
    
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
        dataset=apps_cccs_dataset,
        output_dir=grpo_output_dir,
        num_generations=args.num_generations,
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