import torch
import ast
import re
import json
from typing import List, Dict, Any, Optional, Tuple
import logging
from collections import Counter

NUM_GENERATIONS = 4

logger = logging.getLogger(__name__)


class CodeStructureAnalyzer:
    """
    Analyzes Python code structure using AST parsing and heuristics.
    No code execution - purely static analysis.
    """
    
    def __init__(self):
        # Common patterns that indicate well-structured code
        self.input_patterns = [
            r'\binput\s*\(',
            r'sys\.stdin',
            r'\.readline\(',
            r'\.readlines\(',
        ]
        
        self.output_patterns = [
            r'\bprint\s*\(',
            r'sys\.stdout',
            r'\breturn\b',
        ]
        
        # Control flow keywords indicating algorithmic complexity
        self.control_flow_keywords = ['if', 'else', 'elif', 'for', 'while', 'try', 'except']
        
        # Common algorithmic constructs
        self.algorithmic_patterns = [
            r'\bsorted\s*\(',
            r'\bsort\s*\(',
            r'\bmin\s*\(',
            r'\bmax\s*\(',
            r'\bsum\s*\(',
            r'\blen\s*\(',
            r'\brange\s*\(',
            r'\benumerate\s*\(',
            r'\bzip\s*\(',
            r'\bmap\s*\(',
            r'\bfilter\s*\(',
            r'\bset\s*\(',
            r'\bdict\s*\(',
            r'\blist\s*\(',
            r'\.append\s*\(',
            r'\.extend\s*\(',
            r'\.pop\s*\(',
            r'\.split\s*\(',
            r'\.join\s*\(',
            r'\.strip\s*\(',
        ]

    def parse_code(self, code: str) -> Tuple[Optional[ast.AST], Optional[str]]:
        """
        Parse code into AST. Returns (ast, None) on success or (None, error_message) on failure.
        """
        try:
            tree = ast.parse(code)
            return tree, None
        except SyntaxError as e:
            return None, f"SyntaxError: {e.msg} at line {e.lineno}"
        except Exception as e:
            return None, f"ParseError: {str(e)}"

    def analyze_ast_structure(self, tree: ast.AST) -> Dict[str, Any]:
        """
        Extract structural metrics from AST.
        """
        metrics = {
            'num_functions': 0,
            'num_classes': 0,
            'num_loops': 0,
            'num_conditionals': 0,
            'num_assignments': 0,
            'num_calls': 0,
            'num_imports': 0,
            'num_returns': 0,
            'num_try_except': 0,
            'max_nesting_depth': 0,
            'has_main_block': False,
            'function_names': [],
            'variable_names': set(),
            'called_functions': set(),
        }
        
        def walk_with_depth(node, depth=0):
            metrics['max_nesting_depth'] = max(metrics['max_nesting_depth'], depth)
            
            if isinstance(node, ast.FunctionDef):
                metrics['num_functions'] += 1
                metrics['function_names'].append(node.name)
            elif isinstance(node, ast.AsyncFunctionDef):
                metrics['num_functions'] += 1
                metrics['function_names'].append(node.name)
            elif isinstance(node, ast.ClassDef):
                metrics['num_classes'] += 1
            elif isinstance(node, (ast.For, ast.While)):
                metrics['num_loops'] += 1
            elif isinstance(node, ast.If):
                metrics['num_conditionals'] += 1
            elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                metrics['num_assignments'] += 1
                # Extract variable names
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            metrics['variable_names'].add(target.id)
            elif isinstance(node, ast.Call):
                metrics['num_calls'] += 1
                # Extract called function names
                if isinstance(node.func, ast.Name):
                    metrics['called_functions'].add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    metrics['called_functions'].add(node.func.attr)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['num_imports'] += 1
            elif isinstance(node, ast.Return):
                metrics['num_returns'] += 1
            elif isinstance(node, ast.Try):
                metrics['num_try_except'] += 1
            
            # Check for if __name__ == "__main__" pattern
            if isinstance(node, ast.If):
                try:
                    if (isinstance(node.test, ast.Compare) and
                        isinstance(node.test.left, ast.Name) and
                        node.test.left.id == '__name__'):
                        metrics['has_main_block'] = True
                except:
                    pass
            
            # Recurse with increased depth for control structures
            next_depth = depth + 1 if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, 
                                                         ast.ClassDef, ast.For, ast.While, 
                                                         ast.If, ast.Try, ast.With)) else depth
            
            for child in ast.iter_child_nodes(node):
                walk_with_depth(child, next_depth)
        
        walk_with_depth(tree)
        return metrics

    def analyze_text_patterns(self, code: str) -> Dict[str, Any]:
        """
        Analyze code using regex patterns (catches things AST might miss).
        """
        patterns = {
            'has_input': any(re.search(p, code) for p in self.input_patterns),
            'has_output': any(re.search(p, code) for p in self.output_patterns),
            'algorithmic_pattern_count': sum(1 for p in self.algorithmic_patterns if re.search(p, code)),
            'has_list_comprehension': bool(re.search(r'\[.*\bfor\b.*\bin\b.*\]', code)),
            'has_dict_comprehension': bool(re.search(r'\{.*\bfor\b.*\bin\b.*\}', code)),
            'has_lambda': bool(re.search(r'\blambda\b', code)),
            'line_count': len([l for l in code.split('\n') if l.strip() and not l.strip().startswith('#')]),
            'comment_count': len([l for l in code.split('\n') if l.strip().startswith('#')]),
            'has_docstring': bool(re.search(r'""".*?"""|\'\'\'.*?\'\'\'', code, re.DOTALL)),
        }
        return patterns


class HeuristicRewardCalculator:
    """
    Calculates rewards based on code structure heuristics.
    """
    
    def __init__(self):
        self.analyzer = CodeStructureAnalyzer()
        
        # Weights for different structural components
        self.weights = {
            'syntax_valid': 0.35,           # Major weight for valid syntax
            'has_io': 0.15,                 # Has input/output handling
            'structural_complexity': 0.20,  # Loops, conditionals, functions
            'algorithmic_patterns': 0.15,   # Uses common algorithmic constructs
            'code_completeness': 0.15,      # Non-trivial, complete code
        }

    def calculate_reward(self, code: str, problem_hints: Optional[Dict] = None) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate a reward based on code structure heuristics.
        
        Args:
            code: The generated Python code
            problem_hints: Optional hints about expected structure (e.g., expected keywords)
        
        Returns:
            Tuple of (reward, detailed_breakdown)
        """
        breakdown = {
            'syntax_score': 0.0,
            'io_score': 0.0,
            'complexity_score': 0.0,
            'algorithmic_score': 0.0,
            'completeness_score': 0.0,
            'bonus': 0.0,
            'penalty': 0.0,
        }
        
        # Handle empty or invalid input
        if not isinstance(code, str) or not code.strip():
            return -1.0, {'error': 'Empty or invalid code', **breakdown}
        
        code = code.strip()
        
        # 1. Syntax Validity (most important)
        tree, error = self.analyzer.parse_code(code)
        if tree is None:
            # Syntax error - major penalty but not maximum
            breakdown['syntax_score'] = 0.0
            breakdown['penalty'] = -0.5
            return -0.7, {'error': error, **breakdown}
        
        breakdown['syntax_score'] = 1.0
        
        # 2. Analyze structure
        ast_metrics = self.analyzer.analyze_ast_structure(tree)
        text_patterns = self.analyzer.analyze_text_patterns(code)
        
        # 3. I/O Score - does code read input and produce output?
        io_score = 0.0
        if text_patterns['has_input']:
            io_score += 0.5
        if text_patterns['has_output']:
            io_score += 0.5
        breakdown['io_score'] = io_score
        
        # 4. Structural Complexity Score
        complexity_score = 0.0
        
        # Functions/Classes
        if ast_metrics['num_functions'] > 0:
            complexity_score += 0.2
        if ast_metrics['num_functions'] > 1:
            complexity_score += 0.1
            
        # Control flow
        if ast_metrics['num_loops'] > 0:
            complexity_score += 0.25
        if ast_metrics['num_conditionals'] > 0:
            complexity_score += 0.2
        if ast_metrics['num_conditionals'] > 1:
            complexity_score += 0.1
            
        # Nesting depth (indicates non-trivial logic)
        if ast_metrics['max_nesting_depth'] >= 2:
            complexity_score += 0.1
        if ast_metrics['max_nesting_depth'] >= 3:
            complexity_score += 0.05
        
        breakdown['complexity_score'] = min(1.0, complexity_score)
        
        # 5. Algorithmic Patterns Score
        pattern_count = text_patterns['algorithmic_pattern_count']
        algorithmic_score = min(1.0, pattern_count / 5.0)  # Cap at 5 patterns
        
        # Bonus for advanced constructs
        if text_patterns['has_list_comprehension']:
            algorithmic_score = min(1.0, algorithmic_score + 0.1)
        if text_patterns['has_dict_comprehension']:
            algorithmic_score = min(1.0, algorithmic_score + 0.1)
        
        breakdown['algorithmic_score'] = algorithmic_score
        
        # 6. Code Completeness Score
        completeness_score = 0.0
        
        # Minimum line count for non-trivial code
        line_count = text_patterns['line_count']
        if line_count >= 3:
            completeness_score += 0.3
        if line_count >= 7:
            completeness_score += 0.2
        if line_count >= 15:
            completeness_score += 0.2
        
        # Has assignments (actually does something)
        if ast_metrics['num_assignments'] >= 1:
            completeness_score += 0.15
        if ast_metrics['num_assignments'] >= 3:
            completeness_score += 0.15
        
        breakdown['completeness_score'] = min(1.0, completeness_score)
        
        # 7. Problem-specific hints (if provided)
        if problem_hints:
            hint_bonus = self._check_problem_hints(code, ast_metrics, problem_hints)
            breakdown['bonus'] = hint_bonus
        
        # 8. Penalties for suspicious patterns
        penalty = 0.0
        
        # Just a pass statement or trivial code
        if line_count <= 2 and ast_metrics['num_calls'] == 0:
            penalty -= 0.3
        
        # No actual logic (just imports or pass)
        if ast_metrics['num_loops'] == 0 and ast_metrics['num_conditionals'] == 0 and ast_metrics['num_calls'] <= 1:
            penalty -= 0.2
        
        # Excessive comments relative to code (might be placeholder)
        if text_patterns['comment_count'] > line_count and line_count < 5:
            penalty -= 0.1
        
        breakdown['penalty'] = penalty
        
        # Calculate final weighted score
        weighted_score = (
            self.weights['syntax_valid'] * breakdown['syntax_score'] +
            self.weights['has_io'] * breakdown['io_score'] +
            self.weights['structural_complexity'] * breakdown['complexity_score'] +
            self.weights['algorithmic_patterns'] * breakdown['algorithmic_score'] +
            self.weights['code_completeness'] * breakdown['completeness_score'] +
            breakdown['bonus'] +
            breakdown['penalty']
        )
        
        # Map to reward range: [-1.0, 1.0]
        # Raw weighted_score is roughly [0, 1] before penalties
        # Map: 0.0 -> -0.3 (poor but valid), 0.5 -> 0.3 (decent), 1.0 -> 1.0 (excellent)
        if weighted_score < 0.2:
            reward = -0.3 + (weighted_score / 0.2) * 0.3  # [-0.3, 0.0]
        elif weighted_score < 0.5:
            reward = (weighted_score - 0.2) / 0.3 * 0.5   # [0.0, 0.5]
        else:
            reward = 0.5 + (weighted_score - 0.5) / 0.5 * 0.5  # [0.5, 1.0]
        
        reward = max(-1.0, min(1.0, reward))
        
        return reward, breakdown

    def _check_problem_hints(self, code: str, ast_metrics: Dict, hints: Dict) -> float:
        """
        Check if code matches problem-specific hints.
        """
        bonus = 0.0
        
        # Check for expected keywords
        if 'expected_keywords' in hints:
            keywords = hints['expected_keywords']
            code_lower = code.lower()
            matches = sum(1 for kw in keywords if kw.lower() in code_lower)
            if keywords:
                bonus += 0.1 * (matches / len(keywords))
        
        # Check for expected function names
        if 'expected_functions' in hints:
            expected = set(hints['expected_functions'])
            actual = set(ast_metrics['function_names'])
            if expected & actual:
                bonus += 0.1
        
        # Check for minimum complexity
        if 'min_loops' in hints:
            if ast_metrics['num_loops'] >= hints['min_loops']:
                bonus += 0.05
        
        return bonus


def extract_problem_hints(inputs: List[str], outputs: List[str]) -> Dict[str, Any]:
    """
    Extract hints about expected code structure from test inputs/outputs.
    This helps guide the heuristic without executing code.
    
    Args:
        inputs: List of test input strings. Each can be:
                - Plain string: "3\n5 0 -5\n" (stdin-style)
                - JSON-encoded: '["use ", "sword"]' or '[-10, 10]' (function args)
        outputs: List of test output strings (same formats)
    
    Returns:
        Dictionary of hints about expected code structure
    """
    hints = {}
    
    if not inputs or not outputs:
        return hints
    
    try:
        keywords = []
        
        # Try to detect the format and parse accordingly
        is_json_style = False
        has_strings = False
        has_numbers = False
        has_lists = False
        has_nested_lists = False
        
        for inp in inputs:
            if not inp:
                continue
            inp_str = str(inp).strip()
            
            # Check if it looks like JSON (starts with [ or {)
            if inp_str.startswith('[') or inp_str.startswith('{'):
                is_json_style = True
                try:
                    parsed = json.loads(inp_str)
                    
                    # Analyze the parsed structure
                    if isinstance(parsed, list):
                        has_lists = True
                        for item in parsed:
                            if isinstance(item, str):
                                has_strings = True
                            elif isinstance(item, (int, float)):
                                has_numbers = True
                            elif isinstance(item, list):
                                has_nested_lists = True
                    elif isinstance(parsed, dict):
                        keywords.extend(['dict', 'keys', 'values'])
                    elif isinstance(parsed, str):
                        has_strings = True
                    elif isinstance(parsed, (int, float)):
                        has_numbers = True
                except (json.JSONDecodeError, ValueError):
                    # Not valid JSON, treat as plain string
                    pass
            else:
                # Plain stdin-style input
                if '\n' in inp_str:
                    keywords.extend(['input', 'for', 'while', 'range'])
                    hints['min_loops'] = 1
                if any(c.isdigit() for c in inp_str):
                    has_numbers = True
                if any(c.isalpha() for c in inp_str):
                    has_strings = True
        
        # Analyze outputs similarly
        for out in outputs:
            if not out:
                continue
            out_str = str(out).strip()
            
            if out_str.startswith('[') or out_str.startswith('{'):
                try:
                    parsed = json.loads(out_str)
                    if isinstance(parsed, list):
                        has_lists = True
                        if any(isinstance(item, list) for item in parsed):
                            has_nested_lists = True
                except (json.JSONDecodeError, ValueError):
                    pass
        
        # Generate hints based on detected patterns
        if is_json_style:
            # Function-style problem
            keywords.extend(['def', 'return'])
        else:
            # stdin-style problem
            keywords.append('input')
            keywords.append('print')
        
        if has_lists:
            keywords.extend(['list', 'for', 'range', 'len'])
        
        if has_nested_lists:
            keywords.extend(['for', 'append', 'extend'])
            hints['min_loops'] = hints.get('min_loops', 0) + 1
        
        if has_strings:
            keywords.extend(['str', 'split', 'join', 'strip'])
        
        if has_numbers:
            keywords.extend(['int', 'float', 'sum', 'max', 'min'])
        
        hints['expected_keywords'] = list(set(keywords))
        
    except Exception as e:
        logger.debug(f"Error extracting problem hints: {e}")
    
    return hints


def extract_hints_from_reference(reference: str) -> Dict[str, Any]:
    """
    Extract structural hints from a reference solution.
    """
    hints = {}
    
    if not reference or not isinstance(reference, str):
        return hints
    
    try:
        analyzer = CodeStructureAnalyzer()
        tree, _ = analyzer.parse_code(reference)
        
        if tree is None:
            return hints
        
        metrics = analyzer.analyze_ast_structure(tree)
        
        # Extract hints from reference structure
        hints['expected_functions'] = metrics['function_names']
        hints['min_loops'] = metrics['num_loops']
        hints['min_conditionals'] = metrics['num_conditionals']
        hints['expected_calls'] = list(metrics['called_functions'])
        
    except Exception as e:
        logger.debug(f"Error extracting hints from reference: {e}")
    
    return hints


def unit_test_reward_function(completions, ground_truth, prompts=None, **kwargs):
    """
    Heuristic-based reward function for code generation (NO CODE EXECUTION).
    
    Evaluates code based on:
    - Compilation check (can code be parsed by Python compiler)
    - Structural properties (I/O, complexity, patterns)
    
    Dataset structure:
        - ground_truth: List[str] - one reference solution per problem
        - inputs (via kwargs): List[List[str]] - list of test inputs per problem
        - outputs (via kwargs): List[List[str]] - list of test outputs per problem
        - completions: NUM_GENERATIONS completions per problem (flattened)
    
    Reward Scale:
        +1.0: Excellent structure - compiles, good complexity, proper I/O
        +0.5: Good structure - compiles, some complexity
        +0.0: Basic structure - compiles, minimal logic
        -0.3: Compiles but trivial/incomplete
        -0.7: Compilation/syntax error
        -1.0: Empty or catastrophic failure
    
    Args:
        completions: Model generations in various TRL formats
        ground_truth: Reference solutions (used to extract structural hints)
        prompts: Problem descriptions
        **kwargs: Additional arguments (e.g., inputs, outputs for hints)
    
    Returns:
        List of torch tensors containing rewards for each completion
    """
    calculator = HeuristicRewardCalculator()
    
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
    # ground_truth: List[str] - one solution per problem
    references_raw = ground_truth  # ex: ["solution1", "solution2", ...]
    
    # ----------------------------------------------------
    # 3. Expand references to match flattened completions
    # ----------------------------------------------------
    # completions has NUM_GENERATIONS per problem, so expand ground_truth
    expanded_refs = []
    for ref in references_raw:
        expanded_refs.extend([ref] * NUM_GENERATIONS)
    # Truncate to match completions exactly
    expanded_refs = expanded_refs[:len(completion_texts)]
    
    # ----------------------------------------------------
    # 4. Extract and expand inputs/outputs from kwargs
    # ----------------------------------------------------
    # inputs: List[List[str]] - list of test case inputs per problem
    # outputs: List[List[str]] - list of test case outputs per problem
    inputs = kwargs.get('inputs', None)
    outputs = kwargs.get('outputs', None)
    
    expanded_inputs = None
    expanded_outputs = None
    if inputs is not None and outputs is not None:
        expanded_inputs = []
        expanded_outputs = []
        for i in range(len(references_raw)):
            # Each problem's inputs/outputs is a list of test cases
            problem_inputs = inputs[i] if i < len(inputs) else []
            problem_outputs = outputs[i] if i < len(outputs) else []
            # Expand for NUM_GENERATIONS
            for _ in range(NUM_GENERATIONS):
                expanded_inputs.append(problem_inputs)
                expanded_outputs.append(problem_outputs)
        expanded_inputs = expanded_inputs[:len(completion_texts)]
        expanded_outputs = expanded_outputs[:len(completion_texts)]
    
    # ----------------------------------------------------
    # 5. Calculate rewards for each completion
    # ----------------------------------------------------
    rewards = []
    
    for idx, code in enumerate(completion_texts):
        # Extract problem hints if available
        problem_hints = None
        if expanded_inputs and expanded_outputs and idx < len(expanded_inputs):
            inp = expanded_inputs[idx]  # This is List[str] of test inputs for this problem
            out = expanded_outputs[idx]  # This is List[str] of test outputs for this problem
            if inp and out:
                # inp and out are already lists of strings
                problem_hints = extract_problem_hints(inp, out)
        
        # Also extract hints from reference solution
        if idx < len(expanded_refs) and expanded_refs[idx]:
            ref_hints = extract_hints_from_reference(expanded_refs[idx])
            if problem_hints:
                problem_hints.update(ref_hints)
            else:
                problem_hints = ref_hints
        
        reward, _ = calculator.calculate_reward(code, problem_hints)
        rewards.append(reward)
    
    # ----------------------------------------------------
    # 6. Return torch tensors
    # ----------------------------------------------------
    return [torch.tensor(r, dtype=torch.float32) for r in rewards]


def code_similarity_reward_function(completions, ground_truth, prompts=None, **kwargs):
    """
    Reward function based on structural similarity to reference solution.
    Uses AST comparison rather than text matching for robustness.
    
    Reward Scale:
        +1.0: Very similar structure to reference
        +0.5: Moderately similar structure
        +0.0: Some structural overlap
        -0.5: Little structural similarity
        -1.0: No similarity or parse failure
    
    Args:
        completions: Model generations in various TRL formats
        ground_truth: Reference solutions to compare against
        prompts: Problem descriptions (not used)
        **kwargs: Additional arguments
    
    Returns:
        List of torch tensors containing rewards for each completion
    """
    analyzer = CodeStructureAnalyzer()
    
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
    # 4. Calculate similarity rewards
    # ----------------------------------------------------
    rewards = []
    
    for idx, code in enumerate(completion_texts):
        reference = expanded_refs[idx] if idx < len(expanded_refs) else None
        
        if reference is None:
            rewards.append(-1.0)
            continue
        
        similarity = calculate_structural_similarity(analyzer, code, reference)
        rewards.append(similarity)
    
    # ----------------------------------------------------
    # 5. Return torch tensors
    # ----------------------------------------------------
    return [torch.tensor(r, dtype=torch.float32) for r in rewards]


def calculate_structural_similarity(analyzer: CodeStructureAnalyzer, code: str, reference: str) -> float:
    """
    Calculate structural similarity between generated code and reference solution.
    
    Args:
        analyzer: CodeStructureAnalyzer instance
        code: Generated code
        reference: Reference solution
    
    Returns:
        Similarity score in range [-1.0, 1.0]
    """
    try:
        # Handle empty inputs
        if not isinstance(code, str) or not code.strip():
            return -1.0
        if not isinstance(reference, str) or not reference.strip():
            return -1.0
        
        code = code.strip()
        reference = reference.strip()
        
        # Parse both code snippets
        code_tree, code_err = analyzer.parse_code(code)
        ref_tree, ref_err = analyzer.parse_code(reference)
        
        # If generated code doesn't compile, heavy penalty
        if code_tree is None:
            return -0.7
        
        # If reference doesn't compile, can't compare
        if ref_tree is None:
            logger.warning(f"Reference code failed to parse: {ref_err}")
            return 0.0  # Neutral - not the model's fault
        
        # Get structural metrics
        code_metrics = analyzer.analyze_ast_structure(code_tree)
        ref_metrics = analyzer.analyze_ast_structure(ref_tree)
        
        # Get text patterns
        code_patterns = analyzer.analyze_text_patterns(code)
        ref_patterns = analyzer.analyze_text_patterns(reference)
        
        # Calculate similarity components
        similarity_score = 0.0
        
        # 1. Function structure similarity (weight: 0.15)
        if ref_metrics['num_functions'] > 0:
            func_diff = abs(code_metrics['num_functions'] - ref_metrics['num_functions'])
            func_sim = max(0, 1.0 - func_diff / max(ref_metrics['num_functions'], 1))
            similarity_score += 0.15 * func_sim
        else:
            # No functions in reference - check if generated also has none
            similarity_score += 0.15 if code_metrics['num_functions'] == 0 else 0.08
        
        # 2. Loop structure similarity (weight: 0.20)
        if ref_metrics['num_loops'] > 0:
            loop_diff = abs(code_metrics['num_loops'] - ref_metrics['num_loops'])
            loop_sim = max(0, 1.0 - loop_diff / max(ref_metrics['num_loops'], 1))
            similarity_score += 0.20 * loop_sim
        else:
            similarity_score += 0.20 if code_metrics['num_loops'] == 0 else 0.10
        
        # 3. Conditional structure similarity (weight: 0.15)
        if ref_metrics['num_conditionals'] > 0:
            cond_diff = abs(code_metrics['num_conditionals'] - ref_metrics['num_conditionals'])
            cond_sim = max(0, 1.0 - cond_diff / max(ref_metrics['num_conditionals'], 1))
            similarity_score += 0.15 * cond_sim
        else:
            similarity_score += 0.15 if code_metrics['num_conditionals'] == 0 else 0.08
        
        # 4. Called functions overlap (weight: 0.25)
        code_calls = code_metrics['called_functions']
        ref_calls = ref_metrics['called_functions']
        if ref_calls:
            overlap = len(code_calls & ref_calls)
            union = len(code_calls | ref_calls)
            jaccard = overlap / union if union > 0 else 0
            similarity_score += 0.25 * jaccard
        else:
            similarity_score += 0.25 if not code_calls else 0.12
        
        # 5. I/O pattern match (weight: 0.15)
        io_match = 0
        if code_patterns['has_input'] == ref_patterns['has_input']:
            io_match += 0.5
        if code_patterns['has_output'] == ref_patterns['has_output']:
            io_match += 0.5
        similarity_score += 0.15 * io_match
        
        # 6. Complexity similarity (weight: 0.10)
        code_complexity = (code_metrics['num_loops'] + code_metrics['num_conditionals'] + 
                          code_metrics['max_nesting_depth'])
        ref_complexity = (ref_metrics['num_loops'] + ref_metrics['num_conditionals'] + 
                         ref_metrics['max_nesting_depth'])
        if ref_complexity > 0:
            complexity_ratio = min(code_complexity, ref_complexity) / max(code_complexity, ref_complexity, 1)
            similarity_score += 0.10 * complexity_ratio
        else:
            similarity_score += 0.10 if code_complexity == 0 else 0.05
        
        # Map [0, 1] to [-0.5, 1.0] for reward scale
        # 0.0 similarity -> -0.5 reward
        # 0.5 similarity -> 0.25 reward  
        # 1.0 similarity -> 1.0 reward
        reward = similarity_score * 1.5 - 0.5
        
        return max(-1.0, min(1.0, reward))
        
    except Exception as e:
        logger.error(f"Error in calculate_structural_similarity: {e}")
        return -1.0


def compile_check_reward_function(completions, ground_truth, prompts=None, **kwargs):
    """
    Simple reward function that only checks if code compiles.
    
    Reward Scale:
        +1.0: Code compiles successfully
        -1.0: Code fails to compile (syntax error, etc.)
    
    Args:
        completions: Model generations in various TRL formats
        ground_truth: Reference solutions (not used, kept for interface compatibility)
        prompts: Problem descriptions (not used)
        **kwargs: Additional arguments
    
    Returns:
        List of torch tensors containing rewards for each completion
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
    # 2. Check compilation for each completion
    # ----------------------------------------------------
    rewards = []
    
    for idx, code in enumerate(completion_texts):
        # Handle empty code
        if not isinstance(code, str) or not code.strip():
            rewards.append(-1.0)
            continue
        
        code = code.strip()
        
        # Try to compile
        try:
            compile(code, '<string>', 'exec')
            rewards.append(1.0)  # Compiles successfully
        except SyntaxError:
            rewards.append(-1.0)  # Syntax error
        except Exception:
            rewards.append(-1.0)  # Other compilation error
    
    # ----------------------------------------------------
    # 3. Return torch tensors
    # ----------------------------------------------------
    return [torch.tensor(r, dtype=torch.float32) for r in rewards]


if __name__ == "__main__":
    # Test examples matching actual dataset structure
    # Dataset has: prompt, ground_truth, inputs (List[List[str]]), outputs (List[List[str]])
    
    test_codes = [
        # Good code - should get high reward
        """
n = int(input())
result = 0
for i in range(n):
    x = int(input())
    if x > 0:
        result += x
print(result)
""",
        # Syntax error - should get -0.7 (heuristic) or -1.0 (compile check)
        """
def foo(
    print("broken"
""",
        # Function-style solution
        """
def S2N(m, n):
    return sum(i**j for i in range(m+1) for j in range(n+1))
""",
    ]
    
    # Reference solutions (ground_truth) - one per problem
    reference_codes = [
        """
n = int(input())
total = 0
for _ in range(n):
    num = int(input())
    if num > 0:
        total += num
print(total)
""",
        """
def bar():
    print("hello")
""",
        """
def S2N(m, n):
    return sum(i**j for i in range(m+1) for j in range(n+1))
""",
    ]
    
    # inputs/outputs - List[List[str]] - one list of test cases per problem
    # Format matches what comes from dataset after normalize_test_cases()
    test_inputs = [
        # Problem 0: stdin-style
        ["3\n5 0 -5\n", "1\n0\n"],
        # Problem 1: no test cases
        [],
        # Problem 2: function-arg style (JSON encoded)
        ["[1, 1]", "[2, 3]", "[300, 2]"],
    ]
    
    test_outputs = [
        # Problem 0
        ["1\n", "0\n"],
        # Problem 1
        [],
        # Problem 2
        ["[3]", "[20]", "[9090501]"],
    ]
    
    num_problems = len(reference_codes)
    
    # Create completions: NUM_GENERATIONS per problem
    # In real training, model generates NUM_GENERATIONS completions per prompt
    completions = []
    for i in range(num_problems):
        for gen in range(NUM_GENERATIONS):
            # Use test_codes cyclically for variety
            code_idx = (i + gen) % len(test_codes)
            completions.append([{"content": test_codes[code_idx]}])
    
    # ground_truth is just the list of reference solutions (one per problem)
    ground_truth = reference_codes
    
    print("=" * 70)
    print("DATASET STRUCTURE TEST")
    print("=" * 70)
    print(f"  NUM_GENERATIONS: {NUM_GENERATIONS}")
    print(f"  num_problems: {num_problems}")
    print(f"  len(ground_truth): {len(ground_truth)}")
    print(f"  len(completions): {len(completions)} (should be {num_problems * NUM_GENERATIONS})")
    print(f"  len(inputs): {len(test_inputs)} (one list per problem)")
    print(f"  len(outputs): {len(test_outputs)} (one list per problem)")
    
    print("\n  Sample inputs structure:")
    for i, inp in enumerate(test_inputs):
        print(f"    Problem {i}: {len(inp)} test cases - {inp[:2]}{'...' if len(inp) > 2 else ''}")
    
    print("\n" + "=" * 70)
    print("1. HEURISTIC REWARD FUNCTION TEST")
    print("=" * 70)
    
    heuristic_rewards = unit_test_reward_function(
        completions, 
        ground_truth,
        inputs=test_inputs,
        outputs=test_outputs
    )
    
    print(f"\n  Results ({len(heuristic_rewards)} rewards):")
    print("-" * 50)
    for i, reward in enumerate(heuristic_rewards):
        problem_idx = i // NUM_GENERATIONS
        gen_idx = i % NUM_GENERATIONS
        code_idx = (problem_idx + gen_idx) % len(test_codes)
        code_preview = test_codes[code_idx].strip()[:35].replace('\n', ' ')
        print(f"  [prob={problem_idx}][gen={gen_idx}]: {reward.item():+.3f}  |  {code_preview}...")
    
    print("\n" + "=" * 70)
    print("2. COMPILE CHECK REWARD FUNCTION TEST")
    print("=" * 70)
    
    compile_rewards = compile_check_reward_function(completions, ground_truth)
    
    print(f"\n  Results ({len(compile_rewards)} rewards):")
    print("-" * 50)
    for i, reward in enumerate(compile_rewards):
        problem_idx = i // NUM_GENERATIONS
        gen_idx = i % NUM_GENERATIONS
        status = "COMPILES" if reward.item() > 0 else "FAILS"
        print(f"  [prob={problem_idx}][gen={gen_idx}]: {reward.item():+.3f} ({status})")
    
    print("\n" + "=" * 70)
    print("3. CODE SIMILARITY REWARD FUNCTION TEST")
    print("=" * 70)
    
    similarity_rewards = code_similarity_reward_function(completions, ground_truth)
    
    print(f"\n  Results ({len(similarity_rewards)} rewards):")
    print("-" * 50)
    for i, reward in enumerate(similarity_rewards):
        problem_idx = i // NUM_GENERATIONS
        gen_idx = i % NUM_GENERATIONS
        print(f"  [prob={problem_idx}][gen={gen_idx}]: {reward.item():+.3f}")
    
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    
    expected_len = num_problems * NUM_GENERATIONS
    all_correct = True
    
    if len(heuristic_rewards) != expected_len:
        print(f"  ✗ Heuristic rewards: got {len(heuristic_rewards)}, expected {expected_len}")
        all_correct = False
    else:
        print(f"  ✓ Heuristic rewards: {len(heuristic_rewards)} (correct)")
    
    if len(compile_rewards) != expected_len:
        print(f"  ✗ Compile rewards: got {len(compile_rewards)}, expected {expected_len}")
        all_correct = False
    else:
        print(f"  ✓ Compile rewards: {len(compile_rewards)} (correct)")
    
    if len(similarity_rewards) != expected_len:
        print(f"  ✗ Similarity rewards: got {len(similarity_rewards)}, expected {expected_len}")
        all_correct = False
    else:
        print(f"  ✓ Similarity rewards: {len(similarity_rewards)} (correct)")
    
    if all_correct:
        print("\n  ✓ All reward functions return correct number of rewards!")
    else:
        print("\n  ✗ Some reward functions have incorrect output length!")