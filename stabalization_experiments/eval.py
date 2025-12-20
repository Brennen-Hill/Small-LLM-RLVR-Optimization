from trl import SFTTrainer, SFTConfig
from data_processing_eval import process_apps_dataset

import torch
torch.cuda.empty_cache()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

dataset = process_apps_dataset(1, "test")
dataset = dataset.shuffle(seed=42).select(range(int(0.1* len(dataset))))

sft_config = SFTConfig(
    per_device_train_batch_size=8,
)

trainer = SFTTrainer(
    model="Qwen/Qwen3-1.7B-Base",
    train_dataset=dataset,
    args=sft_config,
)
trainer.train()