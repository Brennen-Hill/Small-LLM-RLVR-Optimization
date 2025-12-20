# train_grpo.py
from datasets import load_dataset
from trl import SFTTrainer, GRPOConfig, GRPOTrainer

dataset_sft = load_dataset("Muennighoff/mbpp", "full")
dataset_grpo = load_dataset("codeparrot/apps")

training_args = GRPOConfig(output_dir="Qwen3-1.7B-GRPO")
trainer = GRPOTrainer(
    model="Qwen/Qwen3-1.7B-Base",
    reward_funcs=,
    args=training_args,
    train_dataset=dataset,
)
trainer.train()