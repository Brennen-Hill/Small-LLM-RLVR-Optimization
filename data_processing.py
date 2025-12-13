"""
Dataset Processing Module for RLVR Training
Handles loading, cleaning, and preprocessing of MBPP and APPS datasets.
"""

from datasets import load_dataset, Dataset
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def process_mbpp_dataset(sample_ratio: float = 0.01) -> Dataset:
    """
    Process MBPP dataset for SFT training.
    
    MBPP structure:
    - task_id, text, code, test_list, test_setup_code, challenge_test_list
    
    Renames:
    - 'text' -> 'prompt'
    - 'code' -> 'completion'
    
    Args:
        sample_ratio: Fraction of dataset to keep (default 1% for testing)
    
    Returns:
        Processed Dataset with only 'prompt' and 'completion' columns
    """
    logger.info("Loading MBPP dataset...")
    dataset = load_dataset("Muennighoff/mbpp", "full")
    
    # MBPP only has 'test' split in some versions, check available splits
    if "test" in dataset:
        data = dataset["test"]
    elif "train" in dataset:
        data = dataset["train"]
    else:
        data = dataset[list(dataset.keys())[0]]
    
    logger.info(f"Original MBPP columns: {data.column_names}")
    logger.info(f"Original MBPP size: {len(data)}")
    
    # Rename columns: text -> prompt, code -> completion
    data = data.rename_columns({
        "text": "prompt",
        "code": "completion"
    })
    
    # Keep only required columns
    columns_to_remove = [col for col in data.column_names if col not in ["prompt", "completion"]]
    data = data.remove_columns(columns_to_remove)
    
    # Sample dataset (keep only sample_ratio)
    num_samples = max(1, int(len(data) * sample_ratio))
    data = data.shuffle(seed=42).select(range(num_samples))
    
    logger.info(f"Processed MBPP columns: {data.column_names}")
    logger.info(f"Processed MBPP size: {len(data)}")
    
    return data


def process_apps_dataset(sample_ratio: float = 0.01, split: str = "train") -> Dataset:
    """
    Process APPS dataset for GRPO training.
    
    APPS structure:
    - problem_id, question, solutions, input_output, difficulty, url, starter_code
    
    Renames:
    - 'question' -> 'prompt'
    - 'solutions' -> 'ground_truth' (takes first solution if multiple)
    
    Args:
        sample_ratio: Fraction of dataset to keep (default 1% for testing)
        split: Which split to use ('train' or 'test')
    
    Returns:
        Processed Dataset with only 'prompt' and 'ground_truth' columns
    """
    logger.info("Loading APPS dataset...")
    dataset = load_dataset("codeparrot/apps", trust_remote_code=True)
    
    data = dataset[split]
    
    logger.info(f"Original APPS columns: {data.column_names}")
    logger.info(f"Original APPS size: {len(data)}")
    
    def extract_first_solution(example):
        """Extract first solution from solutions list/string."""
        solutions = example["solutions"]
        if isinstance(solutions, list) and len(solutions) > 0:
            return {"ground_truth": solutions[0]}
        elif isinstance(solutions, str):
            try:
                sols = json.loads(solutions)
                if isinstance(sols, list) and len(sols) > 0:
                    return {"ground_truth": sols[0]}
            except (json.JSONDecodeError, TypeError):
                pass
            return {"ground_truth": solutions}
        return {"ground_truth": ""}
    
    # Extract first solution and create ground_truth column
    data = data.map(extract_first_solution)
    
    # Rename question -> prompt
    data = data.rename_columns({"question": "prompt"})
    
    # Keep only required columns
    columns_to_remove = [col for col in data.column_names if col not in ["prompt", "ground_truth"]]
    data = data.remove_columns(columns_to_remove)
    
    # Filter out empty samples
    data = data.filter(lambda x: len(x["prompt"]) > 0 and len(x["ground_truth"]) > 0)
    
    # Sample dataset (keep only sample_ratio)
    num_samples = max(1, int(len(data) * sample_ratio))
    data = data.select(range(num_samples))
    
    logger.info(f"Processed APPS columns: {data.column_names}")
    logger.info(f"Processed APPS size: {len(data)}")
    
    return data

def process_apps_cccs_dataset(sample_ratio: float = 0.01, json_path: str = "apps_cccs.json") -> Dataset:
    """
    Process APPS dataset from local JSON file for GRPO training.
    
    Expected JSON format: List of dictionaries with:
    - 'prompt': The coding problem description
    - 'canonical_solution': The reference solution
    - Other fields (inputs, outputs, etc.) that will be filtered out
    
    Renames:
    - 'canonical_solution' -> 'ground_truth'
    
    Args:
        sample_ratio: Fraction of dataset to keep (default 1% for testing)
        json_path: Path to local JSON file
    
    Returns:
        Processed Dataset with only 'prompt' and 'ground_truth' columns
    """
    logger.info(f"Loading APPS dataset from {json_path}...")
    
    # Load JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    logger.info(f"Original APPS size: {len(json_data)}")
    
    # Extract only the fields we need to avoid type conflicts
    simplified_data = []
    for item in json_data:
        simplified_data.append({
            "prompt": item.get("prompt", ""),
            "ground_truth": item.get("canonical_solution", "")
        })
    
    # Convert to HuggingFace Dataset
    data = Dataset.from_list(simplified_data)
    
    logger.info(f"Original APPS columns: {data.column_names}")
    
    # Filter out empty samples
    data = data.filter(lambda x: len(x["prompt"]) > 0 and len(x["ground_truth"]) > 0)
    
    # Sample dataset (keep only sample_ratio) - WITHOUT shuffling
    num_samples = max(1, int(len(data) * sample_ratio))
    data = data.select(range(num_samples))
    
    logger.info(f"Processed APPS columns: {data.column_names}")
    logger.info(f"Processed APPS size: {len(data)}")
    
    return data


def get_datasets(sample_ratio: float = 0.01):
    """
    Load and process both datasets.
    
    Args:
        sample_ratio: Fraction of dataset to keep (default 1% for testing)
    
    Returns:
        Tuple of (mbpp_dataset, apps_dataset)
    """
    mbpp_data = process_mbpp_dataset(sample_ratio=sample_ratio)
    apps_data = process_apps_dataset(sample_ratio=sample_ratio)
    
    return mbpp_data, apps_data


if __name__ == "__main__":
    # Test the processing
    mbpp, apps = get_datasets(sample_ratio=0.01)
    
    print("\n=== MBPP Sample ===")
    print(mbpp[0])
    
    print("\n=== APPS Sample ===")
    print(apps[0])
