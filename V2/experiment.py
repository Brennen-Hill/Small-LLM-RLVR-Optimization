import os
import re
import torch
from dataclasses import dataclass
from typing import List, Optional
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig


# ==========================================
# FIX: Patch to stop Model Card Crash
# ==========================================
def mock_create_model_card(*args, **kwargs):
    print("Skipping model card generation to avoid importlib crash.")

# Overwrite the method in the class definition
GRPOTrainer.create_model_card = mock_create_model_card
# ==========================================

# ==========================================
# 1. Configuration Strategy
# ==========================================

@dataclass
class ExperimentConfig:
    """Controls the switch between Test (Debugging) and Production modes."""
    # Toggle this to False for the full run
    TEST_MODE: bool = True 
 
    @property
    def max_steps(self): return 10 if self.TEST_MODE else 300
    
    # Needs to be >1 for GRPO to calculate variance/advantage
    @property
    def num_generations(self): return 4 if self.TEST_MODE else 8
    
    @property
    def batch_size(self): return 1 if self.TEST_MODE else 4
    
    @property
    def grad_accum(self): return 4 if self.TEST_MODE else 4
    
    # CRITICAL: Increased to allow CoT reasoning
    @property
    def max_len(self): return 512 if self.TEST_MODE else 1024

    # Hyperparameters
    learning_rate: float = 2e-5 if not TEST_MODE else 1e-5
    beta: float = 0.04
    
    # Model & Hardware
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if torch.backends.mps.is_available() else torch.float32

cfg = ExperimentConfig()

# ==========================================
# 2. Robust Reward Functions
# ==========================================

SYSTEM_PROMPT = (
    "Respond to the user's question. "
    "Think step by step in <reasoning> tags. "
    "Put your final answer in <answer> tags."
)

def format_reward_func(completions, **kwargs) -> List[float]:
    """Reward for adhering to XML structure."""
    rewards = []
    for c in completions:
        if isinstance(c, list):
            c = c[-1]["content"]
        
        has_reasoning = "<reasoning>" in c and "</reasoning>" in c
        has_answer = "<answer>" in c and "</answer>" in c
        
        if has_reasoning and has_answer:
            rewards.append(1.0)
        elif has_reasoning or has_answer:
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards

def correctness_reward_func(prompts, completions, ground_truth, **kwargs) -> List[float]:
    """
    Robust reward function:
    1. Checks for Strict XML match (Reward 2.0)
    2. Fallback: Checks for Loose match at end of text (Reward 1.0)
    """
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        if isinstance(completion, list):
            completion = completion[-1]["content"]
            
        truth_str = str(truth).strip().lower()
        
        # --- Step 1: Extract Prediction ---
        match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match:
            pred_str = match.group(1).strip().lower()
            extraction_method = "strict"
        else:
            # Fallback: take the last non-empty line
            lines = [line.strip() for line in completion.strip().split('\n') if line.strip()]
            pred_str = lines[-1].lower() if lines else ""
            extraction_method = "loose"

        # --- Step 2: Compare ---
        # 2a. Numerical extraction helper
        def extract_val(text):
            nums = re.findall(r"[-+]?\d*\.\d+|\d+", text)
            return float(nums[-1]) if nums else None

        reward = 0.0
        
        # Exact string match (for multiple choice or simple words)
        if pred_str == truth_str:
            reward = 2.0
        else:
            # Numerical match
            val_pred = extract_val(pred_str)
            val_truth = extract_val(truth_str)
            if val_pred is not None and val_truth is not None:
                if abs(val_pred - val_truth) < 1e-6:
                    reward = 2.0

        # --- Step 3: Penalty for Bad Formatting ---
        # If we got the right answer but missed the tags, cap reward at 1.0
        if reward == 2.0 and extraction_method == "loose":
            reward = 1.0
            
        rewards.append(reward)
            
    return rewards

# ==========================================
# 3. Data Preparation
# ==========================================

def prepare_datasets():
    print("Loading Datasets...")
    
    def format_gsm8k(example):
        ans = example['answer'].split("####")[-1].strip()
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example['question']}
            ],
            "ground_truth": ans
        }

    def format_arc(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example['question']}
            ],
            "ground_truth": example['answerKey']
        }

    ds_math = load_dataset("gsm8k", "main", split="train")
    ds_math = ds_math.map(format_gsm8k, remove_columns=ds_math.column_names)
    
    ds_science = load_dataset("ai2_arc", "ARC-Challenge", split="train")
    ds_science = ds_science.map(format_arc, remove_columns=ds_science.column_names)

    ds_joint = concatenate_datasets([ds_math, ds_science]).shuffle(seed=42)

    return ds_math, ds_science, ds_joint

# ==========================================
# 4. The Experiment Controller
# ==========================================

class ExperimentController:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Define PEFT Config (LoRA)
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
            task_type=TaskType.CAUSAL_LM,
            lora_dropout=0.05,
            bias="none",
        )

    def get_model(self):
        print(f"Loading Base Model on {cfg.device}...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            torch_dtype=cfg.dtype,
            device_map=cfg.device,
            attn_implementation="sdpa"
        )
        return model

    def evaluate_model(self, model, dataset, name="Test", samples=5):
        """Quickly checks accuracy on a few samples."""
        print(f"\n--- Evaluating Model on {name} ({samples} samples) ---")
        model.eval()
        correct = 0
        
        # Ensure we don't go out of bounds
        indices = range(min(samples, len(dataset)))
        
        for i in indices:
            prompt = dataset[i]['prompt']
            truth = str(dataset[i]['ground_truth']).strip()
            
            # Extract just the user text for the chat template
            inputs = self.tokenizer.apply_chat_template(
                prompt, return_tensors="pt", add_generation_prompt=True
            ).to(cfg.device)

            with torch.no_grad():
                outputs = model.generate(
                    inputs, 
                    max_new_tokens=200,
                    do_sample=False # Greedy for eval
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Simple check
            if truth in response:
                correct += 1
            else:
                pass
        
        acc = correct / len(indices)
        print(f"Accuracy on {name}: {acc:.2%}")
        return acc

    def run_training(self, run_name, train_dataset, model=None):
        print(f"\n>>> Starting Run: {run_name}")
        
        output_dir = f"outputs/{run_name}"
        
        # Logic: If model is passed, we continue training it. 
        # If None, we load fresh base + apply NEW LoRA.
        if model is None:
            model = self.get_model()
            peft_config = self.peft_config
        else:
            # We are continuing training on an existing PEFT model
            print(">>> Continuing training on existing model weights...")
            peft_config = None # Don't overwrite existing config
            
        training_args = GRPOConfig(
            output_dir=output_dir,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.max_steps,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_len,
            beta=cfg.beta,
            logging_steps=1,
            report_to="none",
            use_vllm=False,
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=self.tokenizer,
            reward_funcs=[format_reward_func, correctness_reward_func],
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config,
        )

        trainer.train()
        
        # Save adapter for record keeping
        trainer.model.save_pretrained(f"{output_dir}/final_adapter")
        return trainer.model # Return the actual model object

    def free_memory(self, model_obj):
        """Helper to clear memory for 'Reset Model' phases"""
        del model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def run_full_experiment(self):
        # Dataset A = Math (gsm8k)
        # Dataset B = Science (ai2_arc)
        ds_a_math, ds_b_science, ds_joint = prepare_datasets()

        # ==========================================
        # 1) PHASE 1: BASELINE EVALUATION
        #    Test performance on A and B before finetuning
        # ==========================================
        print("\n" + "="*40)
        print("PHASE 1: BASELINE (NO FINETUNING)")
        print("="*40)
        
        base_model = self.get_model()
        self.evaluate_model(base_model, ds_a_math, "Dataset A (Math) - Baseline")
        self.evaluate_model(base_model, ds_b_science, "Dataset B (Science) - Baseline")
        
        # Reset memory for next phase
        self.free_memory(base_model)

        # ==========================================
        # 2) PHASE 2: SEQUENCE A -> B
        #    Finetune A -> Test A,B -> Finetune B -> Test A,B
        # ==========================================
        print("\n" + "="*40)
        print("PHASE 2: SEQUENCE A -> B")
        print("="*40)

        # 2a. Finetune on Dataset A
        print(">>> Step 2a: Finetuning on Dataset A (Math)")
        model_a = self.run_training("phase2_step1_math", ds_a_math, model=None) # Start fresh
        self.evaluate_model(model_a, ds_a_math, "Dataset A (Math) - After A")
        self.evaluate_model(model_a, ds_b_science, "Dataset B (Science) - After A")

        # 2b. Finetune THAT model on Dataset B
        print(">>> Step 2b: Finetuning SAME model on Dataset B (Science)")
        model_a_b = self.run_training("phase2_step2_science", ds_b_science, model=model_a) # Continue training
        self.evaluate_model(model_a_b, ds_a_math, "Dataset A (Math) - After A->B")
        self.evaluate_model(model_a_b, ds_b_science, "Dataset B (Science) - After A->B")

        # Reset memory for next phase
        self.free_memory(model_a_b)

        # ==========================================
        # 3) PHASE 3: SEQUENCE B -> A
        #    Reset -> Finetune B -> Test A,B -> Finetune A -> Test A,B
        # ==========================================
        print("\n" + "="*40)
        print("PHASE 3: SEQUENCE B -> A (RESET MODEL)")
        print("="*40)

        # 3a. Finetune on Dataset B (Fresh Start)
        print(">>> Step 3a: Finetuning on Dataset B (Science)")
        model_b = self.run_training("phase3_step1_science", ds_b_science, model=None) # Start fresh
        self.evaluate_model(model_b, ds_a_math, "Dataset A (Math) - After B")
        self.evaluate_model(model_b, ds_b_science, "Dataset B (Science) - After B")

        # 3b. Finetune THAT model on Dataset A
        print(">>> Step 3b: Finetuning SAME model on Dataset A (Math)")
        model_b_a = self.run_training("phase3_step2_math", ds_a_math, model=model_b) # Continue training
        self.evaluate_model(model_b_a, ds_a_math, "Dataset A (Math) - After B->A")
        self.evaluate_model(model_b_a, ds_b_science, "Dataset B (Science) - After B->A")

        # Reset memory for next phase
        self.free_memory(model_b_a)

        # ==========================================
        # 4) PHASE 4: JOINT TRAINING
        #    Reset -> Finetune A & B -> Test A,B
        # ==========================================
        print("\n" + "="*40)
        print("PHASE 4: JOINT TRAINING (RESET MODEL)")
        print("="*40)

        print(">>> Step 4: Finetuning on Joint Dataset")
        model_joint = self.run_training("phase4_joint", ds_joint, model=None) # Start fresh
        self.evaluate_model(model_joint, ds_a_math, "Dataset A (Math) - Joint")
        self.evaluate_model(model_joint, ds_b_science, "Dataset B (Science) - Joint")

if __name__ == "__main__":
    controller = ExperimentController()
    controller.run_full_experiment()