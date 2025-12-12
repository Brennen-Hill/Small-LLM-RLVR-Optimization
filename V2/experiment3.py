import os
import re
import torch
import sys
import copy
import gc
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
from trl import GRPOTrainer, GRPOConfig
from torch.utils.data import DataLoader

# ==========================================
# 0. UTILITIES & CONFIG
# ==========================================

class FileLogger:
    def __init__(self, filename="experiment_results.txt", output_to_console=False):
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

@dataclass
class ExperimentConfig:
    # --- TOGGLE THIS FOR DEBUGGING ---
    TEST_MODE: bool = False  
    
    @property
    def max_steps(self): return 10 if self.TEST_MODE else 60
    @property
    def num_generations(self): return 2 if self.TEST_MODE else 4
    @property
    def batch_size(self): return 2 if self.TEST_MODE else 4
    @property
    def grad_accum(self): return 2 if self.TEST_MODE else 4
    @property
    def max_len(self): return 512
    
    @property
    def eval_samples(self): return 5 if self.TEST_MODE else 50
    @property
    def eval_batch_size(self): return 4

    learning_rate: float = 2e-5
    beta: float = 0.04 
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    
    device: str = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype: torch.dtype = torch.float16 if (torch.backends.mps.is_available() or torch.cuda.is_available()) else torch.float32

cfg = ExperimentConfig()

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
        if match: pred_str = match[-1].strip().lower()
        else: 
            lines = [line.strip() for line in completion.strip().split('\n') if line.strip()]
            pred_str = lines[-1].lower() if lines else ""

        if pred_str == truth_str:
            rewards.append(2.0)
            continue
            
        try:
            clean_pred = re.sub(r"[^\d\.-]", "", pred_str)
            clean_truth = re.sub(r"[^\d\.-]", "", truth_str)
            if clean_pred and clean_truth and abs(float(clean_pred) - float(clean_truth)) < 1e-6:
                rewards.append(2.0)
                continue
        except: pass
            
        if truth_str in pred_str: rewards.append(1.0)
        else: rewards.append(0.0)
    return rewards

# ==========================================
# 2. DATA PREPARATION
# ==========================================

def prepare_datasets(tokenizer):
    print("Loading Datasets...")
    
    # --- Dataset A: Math (GSM8K) ---
    ds_math = load_dataset("gsm8k", "main", split="train")
    ds_math = ds_math.shuffle(seed=42).select(range(min(500, len(ds_math)))) 
    
    def format_math(example):
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": example['question']}
        ]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        text += "<reasoning>" # Prompt injection for reasoning start
        ans = example['answer'].split("####")[-1].strip()
        return {"prompt": text, "ground_truth": ans}
    
    ds_math = ds_math.map(format_math, remove_columns=ds_math.column_names)

    # --- Dataset B: Logic (bAbI) ---
    try:
        ds_babi = load_dataset("muennighoff/babi", split="train")
        ds_babi = ds_babi.filter(lambda x: x["task"] == 1)
    except:
        print("Using fallback bAbI dataset...")
        ds_babi = load_dataset("facebook/babi_qa", "en-10k", split="train")
        try:
            ds_babi = ds_babi.filter(lambda x: x.get("story", {}).get("id") == "1" or x.get("story_id") == "1")
        except: pass

    ds_babi = ds_babi.shuffle(seed=42).select(range(min(500, len(ds_babi))))

    def format_babi(example):
        story = example.get('passage', example.get('story', ''))
        if isinstance(story, list):
            if len(story) > 0 and isinstance(story[0], dict):
                story = " ".join([s.get('text', '') for s in story])
            else:
                story = " ".join([str(s) for s in story])
        elif isinstance(story, dict):
             story = story.get('text', '')
        
        msg = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context: {story}\nQuestion: {example['question']}"}
        ]
        text = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        text += "<reasoning>"
        return {"prompt": text, "ground_truth": example['answer']}

    ds_babi = ds_babi.map(format_babi, remove_columns=ds_babi.column_names)

    return ds_math, ds_babi

# ==========================================
# 3. EXPERIMENT CONTROLLER
# ==========================================

class ExperimentController:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.results = {}

    def get_peft_config(self, target_modules=None):
        if target_modules is None:
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"]
        
        return LoraConfig(
            r=16, lora_alpha=32, target_modules=target_modules,
            task_type=TaskType.CAUSAL_LM, lora_dropout=0.05, bias="none"
        )

    def get_model(self):
        print(f"Loading Base Model on {cfg.device}...")
        model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id, torch_dtype=cfg.dtype, device_map=cfg.device, attn_implementation="sdpa"
        )
        model.config.pad_token_id = self.tokenizer.pad_token_id
        return model

    def free_memory(self):
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        elif torch.backends.mps.is_available(): torch.mps.empty_cache()

    # --- TRAINING WRAPPER (FIXED) ---
    def run_training(self, run_name, train_dataset, model=None, peft_config=None, specific_beta=None):
        print(f"\n>>> Training: {run_name}")
        output_dir = f"outputs/{run_name}"
        
        # 1. Load Model if not provided
        if model is None:
            model = self.get_model()
            
        # 2. FIX: Ensure PeftConfig exists for Dense Models
        # If model is NOT a PeftModel (e.g., it's a merged model), we MUST provide a config
        # otherwise GRPOTrainer will attempt full finetuning.
        if peft_config is None and not isinstance(model, PeftModel):
            print(">>> Attaching new LoRA adapter to dense model...")
            peft_config = self.get_peft_config()
        elif isinstance(model, PeftModel) and peft_config is None:
            print(">>> Continuing training on existing Adapter...")
        
        training_args = GRPOConfig(
            output_dir=output_dir,
            learning_rate=cfg.learning_rate,
            per_device_train_batch_size=cfg.batch_size,
            gradient_accumulation_steps=cfg.grad_accum,
            max_steps=cfg.max_steps,
            num_generations=cfg.num_generations,
            max_completion_length=cfg.max_len, 
            beta=specific_beta if specific_beta is not None else cfg.beta, 
            logging_steps=5,
            report_to="none",
            use_vllm=False
        )

        trainer = GRPOTrainer(
            model=model,
            processing_class=self.tokenizer,
            reward_funcs=[format_reward_func, correctness_reward_func],
            args=training_args,
            train_dataset=train_dataset,
            peft_config=peft_config
        )
        
        trainer.train()
        
        os.makedirs(f"{output_dir}/final_adapter", exist_ok=True)
        trainer.model.save_pretrained(f"{output_dir}/final_adapter")
        
        return trainer.model, f"{output_dir}/final_adapter"

    # --- EVALUATION ---
    def evaluate_model(self, model, dataset, name, adapter_a=None, adapter_b=None):
        model.eval()
        correct, total = 0, 0
        
        subset = dataset.select(range(min(cfg.eval_samples, len(dataset))))
        
        def collate_fn(batch):
            return {
                "prompt": [b['prompt'] for b in batch],
                "ground_truth": [str(b['ground_truth']).strip().lower() for b in batch]
            }

        loader = DataLoader(subset, batch_size=cfg.eval_batch_size, collate_fn=collate_fn)
        
        for batch in loader:
            prompts = batch["prompt"]
            truths = batch["ground_truth"]
            
            # --- ORACLE SWITCHING LOGIC ---
            if adapter_a and adapter_b and isinstance(model, PeftModel):
                # Simple heuristic: if prompt contains "Context:", it's likely the Logic task
                is_logic_batch = "Context:" in prompts[0] 
                active_adapter = adapter_b if is_logic_batch else adapter_a
                try:
                    model.set_adapter(active_adapter)
                except: pass
            # ------------------------------

            inputs = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(cfg.device)
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, max_new_tokens=100, do_sample=False, pad_token_id=self.tokenizer.pad_token_id
                )
            
            input_len = inputs['input_ids'].shape[1]
            generated_ids = outputs[:, input_len:]
            responses = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            for pred_full, truth in zip(responses, truths):
                match = re.search(r"<answer>(.*?)</answer>", pred_full, re.DOTALL)
                pred = match.group(1).strip().lower() if match else pred_full.strip().lower()
                
                is_correct = False
                if pred == truth or truth in pred: is_correct = True
                else:
                    try:
                        c_p = re.sub(r"[^\d\.-]", "", pred)
                        c_t = re.sub(r"[^\d\.-]", "", truth)
                        if c_p and c_t and abs(float(c_p)-float(c_t)) < 1e-6: is_correct = True
                    except: pass
                
                if is_correct: correct += 1
                total += 1
                
        acc = correct / total if total > 0 else 0
        print(f"Accuracy on {name}: {acc:.2%}")
        return acc

    # ==========================================
    # 4. IMPLEMENTATION OF METHODS
    # ==========================================

    def run_method_1_baseline(self, ds_a, ds_b):
        """Sequential Training: A -> B (Expect Forgetting)"""
        print("\n=== METHOD 1: BASELINE (SEQUENTIAL) ===")
        
        self.free_memory() 
        model_a, path_a = self.run_training("m1_step1_math", ds_a, model=None)
        
        print(">>> Merging Step 1 for clean sequential training...")
        # Merge LoRA into base weights to simulate "forgetting" in the weights themselves
        model_dense = model_a.merge_and_unload()
        del model_a
        
        # Train B on top of A (Dense model)
        # run_training will detect model_dense is not PeftModel and add NEW LoRA adapter
        model_b, _ = self.run_training("m1_step2_logic", ds_b, model=model_dense)
        
        acc_a = self.evaluate_model(model_b, ds_a, "Task A (Math)")
        acc_b = self.evaluate_model(model_b, ds_b, "Task B (Logic)")
        self.results["Baseline"] = {"Math": acc_a, "Logic": acc_b}
        
        del model_b
        self.free_memory()

    def run_method_2_replay(self, ds_a, ds_b):
        """Experience Replay: Mix 20% of A into B"""
        print("\n=== METHOD 2: EXPERIENCE REPLAY ===")
        self.free_memory()
        
        model_a, _ = self.run_training("m2_step1_math", ds_a, model=None)
        
        print(">>> Merging Step 1...")
        model_dense = model_a.merge_and_unload()
        del model_a

        n_logic = len(ds_b)
        n_replay = int(n_logic * 0.25)
        
        replay_subset = ds_a.shuffle(seed=42).select(range(min(n_replay, len(ds_a))))
        ds_mixed = concatenate_datasets([ds_b, replay_subset]).shuffle(seed=42)
        
        model_b, _ = self.run_training("m2_step2_mixed", ds_mixed, model=model_dense)
        
        acc_a = self.evaluate_model(model_b, ds_a, "Task A (Math)")
        acc_b = self.evaluate_model(model_b, ds_b, "Task B (Logic)")
        self.results["Replay"] = {"Math": acc_a, "Logic": acc_b}
        
        del model_b
        self.free_memory()

    def run_method_3_task_arithmetic(self, ds_a, ds_b):
        """Model Merging: Train A, Train B Indep, Add Task Vectors"""
        print("\n=== METHOD 3: TASK ARITHMETIC (MERGING) ===")
        self.free_memory()
        
        # Train Independent Experts
        _, path_a = self.run_training("m3_indep_math", ds_a, model=None)
        self.free_memory()
        
        _, path_b = self.run_training("m3_indep_logic", ds_b, model=None)
        self.free_memory()
        
        print(">>> Merging Adapters...")
        model = self.get_model()
        model = PeftModel.from_pretrained(model, path_a, adapter_name="math")
        model.load_adapter(path_b, adapter_name="logic")
        
        if isinstance(model, PeftModel):
            model.add_weighted_adapter(
                adapters=["math", "logic"],
                weights=[1.0, 1.0], 
                adapter_name="merged",
                combination_type="linear"
            )
            model.set_adapter("merged")
        
        acc_a = self.evaluate_model(model, ds_a, "Task A (Math)")
        acc_b = self.evaluate_model(model, ds_b, "Task B (Logic)")
        self.results["TaskArithmetic"] = {"Math": acc_a, "Logic": acc_b}
        
        del model
        self.free_memory()

    def run_method_4_ewc_lite(self, ds_a, ds_b):
        """Regularization: High Beta (KL Penalty) as Anchor"""
        print("\n=== METHOD 4: ANCHORING (EWC-LITE) ===")
        self.free_memory()
        
        model_a, _ = self.run_training("m4_step1_math", ds_a, model=None)
        
        # We KEEP the adapter active (no merge).
        # GRPOTrainer will use the current model as the 'ref_model' for KL calculation.
        # High Beta forces the updated policy to stay close to the math-trained policy.
        model_b, _ = self.run_training(
            "m4_step2_logic", ds_b, model=model_a, 
            specific_beta=0.20 # Strong regularization
        )
        
        acc_a = self.evaluate_model(model_b, ds_a, "Task A (Math)")
        acc_b = self.evaluate_model(model_b, ds_b, "Task B (Logic)")
        self.results["EWC_Lite"] = {"Math": acc_a, "Logic": acc_b}
        
        del model_b
        del model_a
        self.free_memory()

    def run_method_5_orthogonal(self, ds_a, ds_b):
        """Parameter Isolation: Split Layers"""
        print("\n=== METHOD 5: ORTHOGONAL LORA ===")
        self.free_memory()
        
        # Task A gets Q, V, Up, Down
        peft_config_a = self.get_peft_config(["q_proj", "v_proj", "up_proj", "down_proj"])
        model_a, _ = self.run_training("m5_step1_math", ds_a, model=None, peft_config=peft_config_a)
        
        print(">>> Merging A for Orthogonal Step 2...")
        # Merge so we have a dense model with A's skills "burned in"
        model_dense = model_a.merge_and_unload()
        del model_a
        
        # Task B gets K, O, Gate (Disjoint from A)
        peft_config_b = self.get_peft_config(["k_proj", "o_proj", "gate_proj"])
        
        # Apply NEW, Orthogonal config to the Dense model
        model_b, _ = self.run_training("m5_step2_logic", ds_b, model=model_dense, peft_config=peft_config_b)
        
        acc_a = self.evaluate_model(model_b, ds_a, "Task A (Math)")
        acc_b = self.evaluate_model(model_b, ds_b, "Task B (Logic)")
        self.results["Orthogonal"] = {"Math": acc_a, "Logic": acc_b}
        
        del model_b
        self.free_memory()

    def run_method_6_ensemble(self, ds_a, ds_b):
        """Oracle: Inference Time Switching"""
        print("\n=== METHOD 6: ENSEMBLE (ORACLE) ===")
        self.free_memory()
        
        _, path_a = self.run_training("m6_indep_math", ds_a, model=None)
        self.free_memory()
        
        _, path_b = self.run_training("m6_indep_logic", ds_b, model=None)
        self.free_memory()
        
        model = self.get_model()
        model = PeftModel.from_pretrained(model, path_a, adapter_name="adapter_a")
        model.load_adapter(path_b, adapter_name="adapter_b")
        
        acc_a = self.evaluate_model(model, ds_a, "Task A (Math)", "adapter_a", "adapter_b")
        acc_b = self.evaluate_model(model, ds_b, "Task B (Logic)", "adapter_a", "adapter_b")
        self.results["Ensemble"] = {"Math": acc_a, "Logic": acc_b}
        
        del model
        self.free_memory()

    def print_summary(self):
        print("\n" + "="*50)
        print("FINAL RESEARCH RESULTS SUMMARY")
        print("="*50)
        print(f"{'Method':<20} | {'Math (Old)':<12} | {'Logic (New)':<12}")
        print("-" * 50)
        for method, scores in self.results.items():
            print(f"{method:<20} | {scores['Math']:.2%}            | {scores['Logic']:.2%}")
        print("-" * 50)

# ==========================================
# 5. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    logger = FileLogger("experiment_output.txt", output_to_console=True)
    sys.stdout = logger
    sys.stderr = logger

    print("Experiment Starting...")
    controller = ExperimentController()
    
    ds_math, ds_logic = prepare_datasets(controller.tokenizer)
    
    try:
        controller.run_method_1_baseline(ds_math, ds_logic)
        controller.run_method_2_replay(ds_math, ds_logic)
        controller.run_method_3_task_arithmetic(ds_math, ds_logic)
        controller.run_method_4_ewc_lite(ds_math, ds_logic)
        controller.run_method_5_orthogonal(ds_math, ds_logic)
        controller.run_method_6_ensemble(ds_math, ds_logic)
    except Exception as e:
        print(f"\nCRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()
    
    controller.print_summary()