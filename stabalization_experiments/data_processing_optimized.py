"""
Dataset Processing Module for RLVR Training
Handles loading, cleaning, and preprocessing of MBPP and APPS datasets.
"""

from datasets import load_dataset, Dataset,Features, Value, Sequence
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

def process_apps_cccs_dataset(sample_ratio: float = 1.0, max_samples: int = 5000, json_path: str = "apps_cccs.json") -> Dataset:
    """
    Process APPS dataset from local JSON file for GRPO training.
    
    Expected JSON format: List of dictionaries with:
    - 'prompt': The coding problem description
    - 'canonical_solution': The reference solution
    - 'inputs': List of test inputs (each input is a string representing one test case)
    - 'outputs': List of expected outputs (each output is a string representing one test case)
    - 'difficulty': Optional difficulty label (introductory/interview/competition)
    - Other fields (starter_code, scope, etc.) that will be filtered out
    
    Renames:
    - 'canonical_solution' -> 'ground_truth'
    
    Problems are sorted from easiest to hardest using a multi-factor heuristic.
    
    Args:
        sample_ratio: Fraction of dataset to keep (default 1.0 = 100%)
        max_samples: Maximum number of samples to keep (default None = no limit)
                     If both sample_ratio and max_samples are set, takes the minimum.
        json_path: Path to local JSON file
    
    Returns:
        Processed Dataset with 'prompt', 'ground_truth', 'inputs', 'outputs' columns,
        sorted by difficulty (easiest first)
    """
    logger.info(f"Loading APPS dataset from {json_path}...")
    
    # Load JSON file
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    logger.info(f"Original APPS size: {len(json_data)}")
    
    def compute_difficulty_score(item):
        """
        Compute a difficulty score for sorting (lower = easier).
        Combines multiple fast-to-compute heuristics.
        """
        prompt = item.get("prompt", "")
        solution = item.get("canonical_solution", "")
        
        # 1. Explicit difficulty mapping (if available)
        difficulty_map = {
            "introductory": 0,
            "interview": 1000,
            "competition": 2000
        }
        base_score = difficulty_map.get(item.get("difficulty", "").lower(), 1000)
        
        # 2. Solution length (lines of code) - weight: 2
        solution_lines = len([line for line in solution.split('\n') if line.strip() and not line.strip().startswith('#')])
        
        # 3. Solution complexity indicators - weight: 50 each
        complexity_keywords = ['for', 'while', 'class', 'def ', 'import', 'lambda', 'yield']
        complexity_count = sum(solution.lower().count(kw) for kw in complexity_keywords)
        
        # 4. Prompt length (longer descriptions often = harder problems) - weight: 0.1
        prompt_length = len(prompt)
        
        # 5. Number of test cases - weight: 20
        inputs_list = item.get("inputs") or []
        num_tests = len(inputs_list) if isinstance(inputs_list, list) else 0
        
        # 6. Advanced algorithm indicators - weight: 100 each
        advanced_keywords = ['dynamic programming', 'dp', 'graph', 'tree', 'dfs', 'bfs', 
                            'binary search', 'greedy', 'backtrack', 'recursion', 'memoization',
                            'optimization', 'shortest path', 'minimum spanning']
        advanced_count = sum(1 for kw in advanced_keywords if kw in prompt.lower())
        
        # 7. Math complexity indicators - weight: 80 each
        math_keywords = ['modulo', 'prime', 'factorial', 'combinatorial', 'probability',
                        'matrix', 'gcd', 'lcm']
        math_count = sum(1 for kw in math_keywords if kw in prompt.lower())
        
        # 8. Data structure complexity - weight: 60 each
        ds_in_solution = ['dict', 'set', 'collections', 'heapq', 'deque', 'Counter', 'defaultdict']
        ds_count = sum(1 for ds in ds_in_solution if ds in solution)
        
        # Combine all factors
        score = (
            base_score +
            solution_lines * 2 +
            complexity_count * 50 +
            prompt_length * 0.1 +
            num_tests * 20 +
            advanced_count * 100 +
            math_count * 80 +
            ds_count * 60
        )
        
        return score
    
    def normalize_test_cases(value):
        """
        Normalize inputs/outputs to always be a list of strings.
        
        Handles multiple formats:
        1. List of strings: ["1\n2\n", "3\n4\n"] - stdin-style test cases
        2. List of lists: [[1, 1], [2, 3]] - function argument style
        3. None or empty
        4. Single values
        
        For nested lists (function args), converts to JSON string representation
        to maintain structure while satisfying PyArrow's type requirements.
        """
        if value is None:
            return []
        
        if isinstance(value, str):
            # Single string -> wrap in list
            return [value] if value else []
        
        if isinstance(value, list):
            if len(value) == 0:
                return []
            
            result = []
            for item in value:
                if item is None:
                    continue
                elif isinstance(item, str):
                    # Already a string (stdin-style)
                    result.append(item)
                elif isinstance(item, list):
                    # Nested list (function args) -> convert to JSON string
                    result.append(json.dumps(item))
                else:
                    # Single value (int, float, etc.) -> convert to JSON
                    result.append(json.dumps(item))
            return result
        
        # Fallback: convert to JSON string and wrap
        return [json.dumps(value)]
    
    # Extract fields and compute difficulty scores
    simplified_data = []
    skipped_count = 0
    
    for item in json_data:
        prompt = item.get("prompt", "")
        ground_truth = item.get("canonical_solution", "")
        
        # Skip empty entries
        if not prompt or not ground_truth:
            skipped_count += 1
            continue
        
        # Normalize inputs and outputs to be consistent lists of strings
        inputs = normalize_test_cases(item.get("inputs"))
        outputs = normalize_test_cases(item.get("outputs"))
        
        difficulty_score = compute_difficulty_score(item)
        
        simplified_data.append({
            "prompt": prompt,
            "ground_truth": ground_truth,
            "inputs": inputs,
            "outputs": outputs,
            "_difficulty_score": difficulty_score  # Temporary field for sorting
        })
    
    if skipped_count > 0:
        logger.info(f"Skipped {skipped_count} entries with empty prompt or solution")
    
    # Sort by difficulty score (ascending = easiest first)
    simplified_data.sort(key=lambda x: x["_difficulty_score"])
    
    logger.info(f"Difficulty score range: {simplified_data[0]['_difficulty_score']:.1f} (easiest) to {simplified_data[-1]['_difficulty_score']:.1f} (hardest)")
    
    # Remove temporary sorting field
    for item in simplified_data:
        del item["_difficulty_score"]
    
    # Define explicit schema to avoid PyArrow type inference issues
    features = Features({
        "prompt": Value("string"),
        "ground_truth": Value("string"),
        "inputs": Sequence(Value("string")),
        "outputs": Sequence(Value("string")),
    })
    
    # Convert to HuggingFace Dataset with explicit schema
    data = Dataset.from_list(simplified_data, features=features)
    
    logger.info(f"Processed APPS columns: {data.column_names}")
    
    # Sample dataset - apply both sample_ratio and max_samples, take minimum
    num_samples_by_ratio = max(1, int(len(data) * sample_ratio))
    
    if max_samples is not None:
        num_samples = min(num_samples_by_ratio, max_samples)
    else:
        num_samples = num_samples_by_ratio
    
    # Ensure we don't exceed dataset size
    num_samples = min(num_samples, len(data))
    
    data = data.select(range(num_samples))
    
    logger.info(f"Processed APPS size: {len(data)} (from {len(simplified_data)} after filtering)")
    
    return data


if __name__ == "__main__":
    # Test with sample data
    logging.basicConfig(level=logging.INFO)
    
    # Create test JSON with both formats
    test_data = [
        # Format 1: stdin-style (list of strings)
        {
            "prompt": "def test():\n    \"\"\"Simple test problem\"\"\"\n",
            "canonical_solution": "def test():\n    print('hello')\n",
            "inputs": ["1\n", "2\n"],
            "outputs": ["hello\n", "hello\n"],
            "difficulty": "introductory"
        },
        # Format 2: function-argument style (list of lists)
        {
            "prompt": "def S2N(m, n):\n    \"\"\"Sum for range\"\"\"\n",
            "canonical_solution": "def S2N(m, n):\n  return sum(i**j for i in range(m+1) for j in range(n+1))",
            "inputs": [[1, 1], [2, 3], [300, 2]],
            "outputs": [[3], [20], [9090501]],
            "difficulty": "introductory"
        },
        # Format 3: Empty inputs/outputs
        {
            "prompt": "def harder():\n    \"\"\"Harder problem\"\"\"\n",
            "canonical_solution": "def harder():\n    for i in range(10):\n        print(i)\n",
            "inputs": [],
            "outputs": [],
            "difficulty": "interview"
        },
        # Format 4: None inputs/outputs
        {
            "prompt": "def edge_case():\n    \"\"\"Edge case with None\"\"\"\n",
            "canonical_solution": "def edge_case():\n    pass\n",
            "inputs": None,
            "outputs": None,
            "difficulty": "introductory"
        },
        # Format 5: Mixed - single values in list
        {
            "prompt": "def single_arg(x):\n    \"\"\"Single arg function\"\"\"\n",
            "canonical_solution": "def single_arg(x):\n    return x * 2",
            "inputs": [5, 10, 15],
            "outputs": [10, 20, 30],
            "difficulty": "introductory"
        },
    ]
    
    # Write test JSON
    with open("/tmp/test_apps.json", "w") as f:
        json.dump(test_data, f)
    
    # Test the function
    try:
        dataset = process_apps_cccs_dataset(sample_ratio=1.0, json_path="/tmp/test_apps.json")
        print(f"\nSuccess! Dataset has {len(dataset)} rows")
        print(f"Columns: {dataset.column_names}")
        
        for i in range(len(dataset)):
            print(f"\nRow {i}:")
            print(f"  prompt: {dataset[i]['prompt'][:40]}...")
            print(f"  inputs: {dataset[i]['inputs']}")
            print(f"  outputs: {dataset[i]['outputs']}")
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


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
