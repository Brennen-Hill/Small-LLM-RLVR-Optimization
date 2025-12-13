import os
import re
import torch
import sys
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import DataLoader

# ==========================================
# 0. OUTPUT REDIRECTION
# ==========================================
class FileLogger:
    def __init__(self, filename="experiment_output.txt", output_to_console=False):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        self.output_to_console = output_to_console

    def write(self, message):
        self.log.write(message)
        self.log.flush()
        if self.output_to_console:
            self.terminal.write(message)
            self.terminal.flush()

    def flush(self):
        self.log.flush()
        if self.output_to_console:
            self.terminal.flush()

# FIX: Patch to stop Model Card Crash
def mock_create_model_card(*args, **kwargs):
    pass
GRPOTrainer.create_model_card = mock_create_model_card

# ==========================================
# 1. Configuration Strategy
# ==========================================

@dataclass
class ExperimentConfig:
    # Set to True for quick debugging (10 steps), False for real run (30-50 steps)
    TEST_MODE: bool = False 
 
    @property
    def max_steps(self): return 10 if self.TEST_MODE else 50
    
    @property
    def num_generations(self): return 4
    
    @property
    def batch_size(self): return 4
    
    @property
    def grad_accum(self): return 4
    
    @property
    def max_len(self): return 512 if self.TEST_MODE else 1024
    
    # Evaluation settings
    @property
    def eval_samples(self): return 10 if self.TEST_MODE else 50
    
    @property
    def eval_batch_size(self): return 10

    learning_rate: float = 2e-5
    beta: float = 0.04
    
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"
    # Fallback to float32 if mps/cuda not available
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
        if isinstance(c, list): c = c[-1]["content"]
        
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
    Robust reward function handling both Math (Numbers) and Logic (Words).
    """
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        if isinstance(completion, list): completion = completion[-1]["content"]
            
        truth_str = str(truth).strip().lower()
        
        # 1. Extract Prediction
        match = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if match:
            pred_str = match[-1].strip().lower()
        else:
            # Fallback: take the last non-empty line
            lines = [line.strip() for line in completion.strip().split('\n') if line.strip()]
            pred_str = lines[-1].lower() if lines else ""

        # 2. Compare
        # Check for exact string match (Logic/bAbI)
        if pred_str == truth_str:
            rewards.append(2.0)
            continue
            
        # Check for numerical match (Math/GSM8K)
        try:
            # Remove non-numeric chars for loose matching
            clean_pred = re.sub(r"[^\d\.-]", "", pred_str)
            clean_truth = re.sub(r"[^\d\.-]", "", truth_str)
            if clean_pred and clean_truth and abs(float(clean_pred) - float(clean_truth)) < 1e-6:
                rewards.append(2.0)
                continue
        except:
            pass
            
        # Check for containment (Partial credit)
        if truth_str in pred_str:
            rewards.append(1.0)
        else:
            rewards.append(0.0)
            
    return rewards

# ==========================================
# 3. Data Preparation
# ==========================================

def prepare_datasets(tokenizer):
    print("Loading Datasets...")
    
    # --- Dataset A: Math (GSM8K) ---
    ds_math = load_dataset("gsm8k", "main", split="train")
    
    def format_math(example):
        question = example['question']
        answer = example['answer'].split("####")[-1].strip()
        
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question}
        ]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        text += "<reasoning>" 
        return {"prompt": text, "ground_truth": answer}
    
    ds_math = ds_math.map(format_math, remove_columns=ds_math.column_names)

    # --- Dataset B: Logic (bAbI Task 1) ---
    # FIX: Load 'default' config and filter for task 1
    ds_babi = load_dataset("muennighoff/babi", split="train")
    ds_babi = ds_babi.filter(lambda x: x["task"] == 1) # Task 1 is "Single Supporting Fact"

    def format_babi(example):
        # FIX: Access new column names directly
        story = example['passage'] # 'passage' is already a single string
        question = example['question']
        answer = example['answer']
        
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context: {story}\nQuestion: {question}"}
        ]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        text += "<reasoning>"
        return {"prompt": text, "ground_truth": answer}

    ds_babi = ds_babi.map(format_babi, remove_columns=ds_babi.column_names)

    # --- Joint Dataset ---
    ds_joint = concatenate_datasets([ds_math, ds_babi]).shuffle(seed=42)

    return ds_math, ds_babi, ds_joint

# ==========================================
# 4. The Experiment Controller
# ==========================================

class ExperimentController:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
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
            attn_implementation="sdpa" # Use "eager" if sdpa causes issues
        )
        return model

    def evaluate_model(self, model, dataset, name="Test"):
        """
        Batched evaluation for speed.
        """
        samples = cfg.eval_samples
        batch_size = cfg.eval_batch_size
        
        print(f"\n--- Evaluating Model on {name} ({samples} samples) ---")
        model.eval()
        correct = 0
        total = 0
        
        # Select reproducible subset
        indices = range(min(samples, len(dataset)))
        subset = dataset.select(indices)
        
        # Custom collate because 'prompt' is string
        def collate_fn(batch):
            return {
                "prompt": [b['prompt'] for b in batch],
                "ground_truth": [str(b['ground_truth']).strip().lower() for b in batch]
            }

        loader = DataLoader(subset, batch_size=batch_size, collate_fn=collate_fn)

        for batch in loader:
            prompts = batch["prompt"]
            truths = batch["ground_truth"]
            
            inputs = self.tokenizer(
                prompts, 
                return_tensors="pt", 
                padding=True, 
                padding_side="left"
            ).to(cfg.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=200,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Slice off input tokens
            input_len = inputs['input_ids'].shape[1]
            generated_ids = outputs[:, input_len:]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for i, response in enumerate(responses):
                if total == 0:
                    print(f"Sample Output: {response[:150]}...") # Sanity check

                # Extract Answer
                match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
                if match:
                    pred = match.group(1).strip().lower()
                else:
                    pred = response.strip().lower()

                truth = truths[i]
                
                # Check correctness
                is_correct = False
                if pred == truth:
                    is_correct = True
                elif truth in pred: # Loose match fallback
                    is_correct = True
                else:
                    # Number check
                    try:
                        clean_pred = re.sub(r"[^\d\.-]", "", pred)
                        clean_truth = re.sub(r"[^\d\.-]", "", truth)
                        if clean_pred and clean_truth and abs(float(clean_pred) - float(clean_truth)) < 1e-6:
                            is_correct = True
                    except:
                        pass
                
                if is_correct:
                    correct += 1
                total += 1
        
        acc = correct / total if total > 0 else 0
        print(f"Accuracy on {name}: {acc:.2%}")
        return acc

    def run_training(self, run_name, train_dataset, model=None):
        print(f"\n>>> Starting Run: {run_name}")
        output_dir = f"outputs/{run_name}"
        
        if model is None:
            model = self.get_model()
            peft_config = self.peft_config
        else:
            print(">>> Continuing training on existing model weights...")
            peft_config = None 
            
        training_args = GRPOConfig(
            output_dir=output_dir,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.max_steps,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_len,
            beta=cfg.beta,
            logging_steps=5, # Reduce log spam
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
        
        # Save adapter but return model in memory for next step
        trainer.model.save_pretrained(f"{output_dir}/final_adapter")
        return trainer.model 

    def free_memory(self, model_obj):
        del model_obj
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def run_full_experiment(self):
        ds_a_math, ds_b_logic, ds_joint = prepare_datasets(self.tokenizer)

        print("\n" + "="*40)
        print("PHASE 1: BASELINE")
        print("="*40)
        
        base_model = self.get_model()
        self.evaluate_model(base_model, ds_a_math, "Dataset A (Math)")
        self.evaluate_model(base_model, ds_b_logic, "Dataset B (Logic)")
        self.free_memory(base_model)

        print("\n" + "="*40)
        print("PHASE 2: SEQUENCE A -> B")
        print("="*40)

        print(">>> Step 2a: Finetuning on Dataset A (Math)")
        model_a = self.run_training("phase2_step1_math", ds_a_math, model=None)
        self.evaluate_model(model_a, ds_a_math, "Dataset A (Math) - After A")
        self.evaluate_model(model_a, ds_b_logic, "Dataset B (Logic) - After A")

        print(">>> Step 2b: Finetuning SAME model on Dataset B (Logic)")
        model_a_b = self.run_training("phase2_step2_logic", ds_b_logic, model=model_a)
        self.evaluate_model(model_a_b, ds_a_math, "Dataset A (Math) - After A->B")
        self.evaluate_model(model_a_b, ds_b_logic, "Dataset B (Logic) - After A->B")
        self.free_memory(model_a_b)

        print("\n" + "="*40)
        print("PHASE 3: SEQUENCE B -> A (RESET MODEL)")
        print("="*40)

        print(">>> Step 3a: Finetuning on Dataset B (Logic)")
        model_b = self.run_training("phase3_step1_logic", ds_b_logic, model=None)
        self.evaluate_model(model_b, ds_a_math, "Dataset A (Math) - After B")
        self.evaluate_model(model_b, ds_b_logic, "Dataset B (Logic) - After B")

        print(">>> Step 3b: Finetuning SAME model on Dataset A (Math)")
        model_b_a = self.run_training("phase3_step2_math", ds_a_math, model=model_b)
        self.evaluate_model(model_b_a, ds_a_math, "Dataset A (Math) - After B->A")
        self.evaluate_model(model_b_a, ds_b_logic, "Dataset B (Logic) - After B->A")
        self.free_memory(model_b_a)

        print("\n" + "="*40)
        print("PHASE 4: JOINT TRAINING (RESET MODEL)")
        print("="*40)

        print(">>> Step 4: Finetuning on Joint Dataset")
        model_joint = self.run_training("phase4_joint", ds_joint, model=None)
        self.evaluate_model(model_joint, ds_a_math, "Dataset A (Math) - Joint")
        self.evaluate_model(model_joint, ds_b_logic, "Dataset B (Logic) - Joint")

if __name__ == "__main__":
    # --- Redirect Output Start ---
    # Set output_to_console=True if you want to see progress bars in real time
    logger = FileLogger("experiment_output.txt", output_to_console=True)
    sys.stdout = logger
    sys.stderr = logger 
    # --- Redirect Output End ---

    print("Experiment Starting...")
    controller = ExperimentController()
    controller.run_full_experiment()
