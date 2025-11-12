# Benchmark Loaders

This directory contains loaders for different code generation benchmarks.

## PythonSaga Benchmark

### Overview

**PythonSaga** is a high-quality benchmark with 185 hand-crafted problems covering 38 programming concepts across diverse difficulty levels.

- **Paper**: "PythonSaga: Redefining the Benchmark to Evaluate Code Generating LLMs" (EMNLP 2024)
- **Authors**: Ankit Yadav, Himanshu Beniwal, Mayank Singh
- **GitHub**: https://github.com/PythonSaga/ACL2024

### Features

- **185 Problems**: Manually crafted with attention to quality
- **38 Concepts**: Balanced representation across programming concepts
- **Difficulty Distribution**: 20% Easy, 40% Medium, 40% Hard
- **Real-World Context**: Problems phrased in human-friendly language
- **Average 3.7 test cases per problem**

### Dataset Format

Each problem in PythonSaga contains:

```json
{
    "task_id": "PythonSaga/0",
    "prompt": "Function description with examples...",
    "entry_point": "function_name",
    "canonical_solution": "Reference solution code...",
    "test": "Test code with assertions..."
}
```

### Usage

```python
from evaluation.benchmarks import PythonSagaBenchmark

# Initialize and load
benchmark = PythonSagaBenchmark()
benchmark.load()  # Downloads automatically if not present

# Get statistics
stats = benchmark.get_statistics()
print(f"Total problems: {stats['total_problems']}")

# Get a specific problem
problem = benchmark.get_problem_by_index(0)
print(f"Task: {problem['task_id']}")
print(f"Entry point: {problem['entry_point']}")

# Format for ThreEvo
formatted = benchmark.format_problem_for_threevo(problem)

# Iterate over all problems
for problem in benchmark:
    print(problem['task_id'])

# Save results
results = [
    {'task_id': 'PythonSaga/0', 'final_code': '...', 'converged': True}
]
benchmark.save_results(results, 'output.jsonl')
```

### Download

The loader automatically downloads the dataset from GitHub on first use. The data is stored in:

```
data/pythonsaga/basic185.jsonl
```

Alternatively, manually download from:
- https://github.com/PythonSaga/ACL2024/tree/main/DataSet

### Comparison with Other Benchmarks

| Benchmark | Problems | Saturated? | Difficulty Balance | Concept Diversity |
|-----------|----------|------------|-------------------|-------------------|
| HumanEval | 164      | ✅ High    | ❌ Skewed        | ❌ Limited        |
| MBPP      | 974      | ✅ High    | ❌ Mostly Easy   | ❌ Limited        |
| **PythonSaga** | **185** | **✅ Low** | **✅ Balanced** | **✅ High (38 concepts)** |

### Why Focus on PythonSaga?

1. **Less Saturated**: Models haven't been heavily optimized for it yet
2. **Better Difficulty Balance**: More challenging problems than HumanEval/MBPP
3. **Concept Diversity**: Tests a wider range of programming concepts
4. **Research Potential**: Better for demonstrating novel approaches
5. **Quality**: Hand-crafted with attention to real-world context

### Performance Baseline

From the PythonSaga paper (EMNLP 2024):

**Open-Source Models:**
- Code Llama Python 13B: Best performer
- Llama 2 70B: Lower performance
- StarCoder: Lower performance

**Closed-Source Models:**
- GPT-4: Best performer
- GPT-3.5: Moderate performance
- Note: All models perform significantly worse on PythonSaga than on HumanEval/MBPP

This suggests PythonSaga is a more challenging and realistic benchmark for evaluating code generation capabilities.

## Future Benchmarks

Additional benchmark loaders can be added here:

- `humaneval.py` - HumanEval benchmark (164 problems)
- `mbpp.py` - MBPP benchmark (974 problems)
- `apps.py` - APPS benchmark (competitive programming)
- `codenet.py` - CodeNet benchmark

Each loader should follow the same interface:
- `load()` - Load the dataset
- `get_problem(task_id)` - Get specific problem
- `get_all_problems()` - Get all problems
- `format_problem_for_threevo(problem)` - Format for ThreEvo
- `save_results(results, path)` - Save evaluation results
