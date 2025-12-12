import os
import re
import torch
import sys
import gc
import traceback
from dataclasses import dataclass
from typing import List, Optional, Dict
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType, PeftModel
from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import DataLoader

# ==========================================
# 0. CONFIGURATION & TARGETS
# ==========================================

# Define the specific models to test for the "Capacity Cliff"
MODELS_TO_TEST = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    # "Qwen/Qwen2.5-3B-Instruct"  # UNCOMMENT if you have >16GB VRAM/RAM
]

@dataclass
class ExperimentConfig:
    # --- Experiment Settings ---
    TEST_MODE: bool = False  # Set True for quick debugging (10 steps)
    
    # Training Hyperparameters
    learning_rate: float = 2e-5
    beta: float = 0.04 
    max_len: int = 512
    
    @property
    def max_steps(self): return 10 if self.TEST_MODE else 100 # Increased to 100 to ensure convergence
    @property
    def num_generations(self): return 2 if self.TEST_MODE else 4
    @property
    def eval_samples(self): return 5 if self.TEST_MODE else 50
    @property
    def device(self): 
        return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    @property
    def dtype(self): 
        return torch.float16 if (torch.backends.mps.is_available() or torch.cuda.is_available()) else torch.float32

cfg = ExperimentConfig()

# Patch to prevent GRPOTrainer Model Card crash
def mock_create_model_card(*args, **kwargs): pass
GRPOTrainer.create_model_card = mock_create_model_card

# ==========================================
# 1. REWARD FUNCTIONS
# ==========================================

SYSTEM_PROMPT = (
    "Respond to the user's question. "
    "Think step by step in <reasoning> tags. "
    "Put your final answer in <answer> tags."
)

def format_reward_func(completions, **kwargs) -> List[float]:
    rewards = []
    for c in completions:
        if isinstance(c, list): c = c[-1]["content"]
        has_reasoning = "<reasoning>" in c and "</reasoning>" in c
        has_answer = "<answer>" in c and "</answer>" in c
        if has_reasoning and has_answer: rewards.append(1.0)
        elif has_reasoning or has_answer: rewards.append(0.5)
        else: rewards.append(0.0)
    return rewards

def correctness_reward_func(prompts, completions, ground_truth, **kwargs) -> List[float]:
    rewards = []
    for completion, truth in zip(completions, ground_truth):
        if isinstance(completion, list): completion = completion[-1]["content"]
        truth_str = str(truth).strip().lower()
        match = re.findall(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        pred_str = match[-1].strip().lower() if match else ""
        
        if pred_str == truth_str:
            rewards.append(2.0)
            continue
        try:
            # Flexible number matching
            clean_pred = re.sub(r"[^\d\.-]", "", pred_str)
            clean_truth = re.sub(r"[^\d\.-]", "", truth_str)
            if clean_pred and clean_truth and abs(float(clean_pred) - float(clean_truth)) < 1e-6:
                rewards.append(2.0)
            elif truth_str in pred_str:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        except:
            rewards.append(0.0)
    return rewards

# ==========================================
# 2. EXPERIMENT CONTROLLER
# ==========================================

class ExperimentController:
    def __init__(self):
        self.results = {} 

    def free_memory(self):
        """Aggressive memory cleaning to switch models safely."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    def get_batch_config(self, model_id):
        """Adjust batch size/accum steps based on model scale."""
        if "0.5B" in model_id:
            return 4, 4  # Batch 4, Accum 4 (Total 16)
        elif "1.5B" in model_id:
            return 2, 8  # Batch 2, Accum 8 (Total 16) - Save Memory
        elif "3B" in model_id:
            return 1, 16 # Batch 1, Accum 16 (Total 16) - Squeeze into VRAM
        return 2, 8

    def get_tokenizer(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def prepare_data(self, tokenizer):
        # --- Math (GSM8K) ---
        ds_math = load_dataset("gsm8k", "main", split="train").shuffle(seed=42).select(range(500))
        
        def format_math(x):
            msg = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": x['question']}]
            return {
                "prompt": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<reasoning>",
                "ground_truth": x['answer'].split("####")[-1].strip()
            }
        ds_math = ds_math.map(format_math, remove_columns=ds_math.column_names)

        # --- Logic (bAbI) ---
        try:
            ds_babi = load_dataset("muennighoff/babi", split="train").filter(lambda x: x["task"] == 1)
        except:
            ds_babi = load_dataset("facebook/babi_qa", "en-10k", split="train") # Fallback
            
        ds_babi = ds_babi.shuffle(seed=42).select(range(500))

        def format_babi(x):
            story = x.get('passage', x.get('story', ''))
            if isinstance(story, list): story = " ".join([str(s) for s in story])
            elif isinstance(story, dict): story = story.get('text', '')
            
            msg = [{"role": "system", "content": SYSTEM_PROMPT}, 
                   {"role": "user", "content": f"Context: {story}\nQuestion: {x['question']}"}]
            return {
                "prompt": tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<reasoning>",
                "ground_truth": x['answer']
            }
        ds_babi = ds_babi.map(format_babi, remove_columns=ds_babi.column_names)
        
        return ds_math, ds_babi

    def train_grpo(self, run_name, model_id, train_ds, model=None):
        print(f"\n>>> Training {run_name} ({len(train_ds)} samples)...")
        output_dir = f"outputs_exp1/{run_name}"
        
        # Load model if not provided
        if model is None:
            model = AutoModelForCausalLM.from_pretrained(
                model_id, torch_dtype=cfg.dtype, device_map=cfg.device, attn_implementation="sdpa"
            )
            # Add LoRA
            peft_config = LoraConfig(
                r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                task_type=TaskType.CAUSAL_LM, lora_dropout=0.05, bias="none"
            )
        else:
            peft_config = None # Existing model already has config if it's a PeftModel, or we add logic below
            if not isinstance(model, PeftModel):
                 peft_config = LoraConfig(
                    r=16, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    task_type=TaskType.CAUSAL_LM, lora_dropout=0.05, bias="none"
                )

        tokenizer = self.get_tokenizer(model_id)
        batch_size, grad_accum = self.get_batch_config(model_id)

        training_args = GRPOConfig(
            output_dir=output_dir,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            max_steps=cfg.max_steps,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_len,
            beta=cfg.beta,
            logging_steps=10,
            report_to="none",
            use_vllm=False
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=tokenizer,
            reward_funcs=[format_reward_func, correctness_reward_func],
            args=training_args,
            train_dataset=train_ds,
            peft_config=peft_config
        )
        
        trainer.train()
        return trainer.model

    def evaluate(self, model, dataset, tokenizer, name):
        model.eval()
        correct, total = 0, 0
        dataset = dataset.select(range(min(cfg.eval_samples, len(dataset))))
        
        loader = DataLoader(dataset, batch_size=4, collate_fn=lambda b: {
            "prompt": [x['prompt'] for x in b], 
            "ground_truth": [str(x['ground_truth']).strip().lower() for x in b]
        })

        for batch in loader:
            inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, padding_side="left").to(cfg.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            
            responses = tokenizer.batch_decode(outputs[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            
            for pred_full, truth in zip(responses, batch["ground_truth"]):
                match = re.search(r"<answer>(.*?)</answer>", pred_full, re.DOTALL)
                pred = match.group(1).strip().lower() if match else pred_full.strip().lower()
                
                # Loose check
                if pred == truth or truth in pred: 
                    correct += 1
                else:
                    # Number check
                    try:
                        c_p = re.sub(r"[^\d\.-]", "", pred)
                        c_t = re.sub(r"[^\d\.-]", "", truth)
                        if c_p and c_t and abs(float(c_p)-float(c_t)) < 1e-6: correct += 1
                    except: pass
                total += 1
                
        return correct / total if total > 0 else 0

    # ==========================================
    # METHODS
    # ==========================================

    def run_baseline(self, model_id, ds_math, ds_logic):
        """Method 1: Sequential Training (A -> B)"""
        print(f"--- Running Baseline on {model_id} ---")
        
        # 1. Train Math
        model = self.train_grpo(f"baseline_{model_id.split('/')[-1]}_math", model_id, ds_math)
        
        # 2. Merge
        model = model.merge_and_unload()
        
        # 3. Train Logic
        model = self.train_grpo(f"baseline_{model_id.split('/')[-1]}_logic", model_id, ds_logic, model=model)
        
        # 4. Evaluate
        tokenizer = self.get_tokenizer(model_id)
        acc_math = self.evaluate(model, ds_math, tokenizer, "Math")
        acc_logic = self.evaluate(model, ds_logic, tokenizer, "Logic")
        
        del model
        self.free_memory()
        return acc_math, acc_logic

    def run_replay(self, model_id, ds_math, ds_logic):
        """Method 2: Experience Replay (Mix 25% Math into Logic)"""
        print(f"--- Running Replay on {model_id} ---")
        
        # 1. Train Math
        model = self.train_grpo(f"replay_{model_id.split('/')[-1]}_math", model_id, ds_math)
        
        # 2. Merge
        model = model.merge_and_unload()
        
        # 3. Create Mixed Dataset
        n_replay = int(len(ds_logic) * 0.25)
        replay_subset = ds_math.shuffle(seed=42).select(range(n_replay))
        ds_mixed = concatenate_datasets([ds_logic, replay_subset]).shuffle(seed=42)
        
        # 4. Train Mixed
        model = self.train_grpo(f"replay_{model_id.split('/')[-1]}_mixed", model_id, ds_mixed, model=model)
        
        # 5. Evaluate
        tokenizer = self.get_tokenizer(model_id)
        acc_math = self.evaluate(model, ds_math, tokenizer, "Math")
        acc_logic = self.evaluate(model, ds_logic, tokenizer, "Logic")
        
        del model
        self.free_memory()
        return acc_math, acc_logic

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("=== EXPERIMENT 1: CAPACITY CLIFF ===")
    print(f"Testing Models: {MODELS_TO_TEST}")
    
    controller = ExperimentController()
    
    # Store results: results[model_id][method] = (math_score, logic_score)
    final_data = {m: {} for m in MODELS_TO_TEST}
    
    for model_id in MODELS_TO_TEST:
        print(f"\n\n{'='*40}\nPROCESSING MODEL: {model_id}\n{'='*40}")
        try:
            tokenizer = controller.get_tokenizer(model_id)
            ds_math, ds_logic = controller.prepare_data(tokenizer)
            
            # --- RUN BASELINE ---
            math_bl, logic_bl = controller.run_baseline(model_id, ds_math, ds_logic)
            final_data[model_id]['Baseline'] = (math_bl, logic_bl)
            print(f">>> {model_id} Baseline Result: Math={math_bl:.2%}, Logic={logic_bl:.2%}")
            
            # --- RUN REPLAY ---
            math_rep, logic_rep = controller.run_replay(model_id, ds_math, ds_logic)
            final_data[model_id]['Replay'] = (math_rep, logic_rep)
            print(f">>> {model_id} Replay Result: Math={math_rep:.2%}, Logic={logic_rep:.2%}")
            
        except Exception as e:
            print(f"!!! Error processing {model_id}: {e}")
            traceback.print_exc()
            controller.free_memory()

    # ==========================================
    # 4. REPORTING
    # ==========================================
    print("\n" + "="*60)
    print("FINAL CAPACITY CLIFF RESULTS")
    print("="*60)
    print(f"{'Model':<30} | {'Method':<10} | {'Math (Old)':<10} | {'Logic (New)':<10} | {'Retention Gap'}")
    print("-" * 75)
    
    for model_id, methods in final_data.items():
        if not methods: continue
        
        # Baseline Row
        m_b, l_b = methods.get('Baseline', (0,0))
        print(f"{model_id.split('/')[-1]:<30} | {'Base':<10} | {m_b:.2%}     | {l_b:.2%}     | -")
        
        # Replay Row
        m_r, l_r = methods.get('Replay', (0,0))
        gap = m_r - m_b
        gap_str = f"+{gap:.2%}" if gap > 0 else f"{gap:.2%}"
        print(f"{'':<30} | {'Replay':<10} | {m_r:.2%}     | {l_r:.2%}     | {gap_str}")
        print("-" * 75)