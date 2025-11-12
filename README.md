# ThreEvo: Three-Agent Evolutionary Framework for Code Generation

ThreEvo is a novel evolutionary framework for code generation that uses semantic feedback to improve code correctness through three-way validation.

## Core Innovation

Unlike existing systems that rely solely on execution-based validation, ThreEvo introduces a **Reasoning Agent** that independently validates both code and tests, providing rich natural language feedback to drive iterative improvement.

### Three-Way Validation

ThreEvo compares three outputs to diagnose issues:
- **Expected output** (yi) - from Tester Agent
- **Actual output** (ŷi) - from executing Coder Agent's code
- **Reasoned solution** (zi) - from Reasoning Agent independently solving the problem

This breaks the circular dependency problem where correlated errors in both code and tests would go undetected.

## Quick Start

### Prerequisites

- Python 3.8+
- Anthropic API key

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/threevo.git
cd threevo
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
cp .env.example .env
# Edit .env and add your ANTHROPIC_API_KEY
```

### Running the Examples

**Basic Example:**
```bash
python example.py
```

This will run ThreEvo on a HumanEval-style problem and demonstrate the evolutionary loop.

**PythonSaga Benchmark:**
```bash
# Show benchmark information
python example_pythonsaga.py info

# Run on a single problem (problem index 0)
python example_pythonsaga.py single 0

# Run on a batch of problems (start at 0, run 5 problems)
python example_pythonsaga.py batch 0 5
```

## Architecture

### The Three Agents

1. **Coder Agent** - Generates code solutions, evolves based on feedback about code errors
2. **Tester Agent** - Generates test suites, evolves based on feedback about test specification errors
3. **Reasoning Agent** - Independently solves problems and performs three-way diagnosis

### Evolutionary Loop

```
Generate code & tests (parallel) → Execute → Reasoning validates →
Generate semantic feedback → Evolve prompts → Repeat until convergence
```

## Project Structure

```
threevo/
├── agents/              # Three agents and orchestration logic
│   ├── base_agent.py
│   ├── coder_agent.py
│   ├── tester_agent.py
│   ├── reasoning_agent.py
│   └── orchestration.py
├── coordinator/         # Main evolutionary loop
│   ├── coordinator.py
│   └── state_manager.py
├── execution/          # Code execution with timeouts
│   └── executor.py
├── evaluation/         # Benchmark loading and metrics
│   └── benchmarks/
│       └── pythonsaga.py
├── config/             # Configuration management
│   ├── settings.py
│   └── experiment_config.yaml
├── example.py          # Basic example
├── example_pythonsaga.py  # PythonSaga benchmark example
└── requirements.txt
```

## Usage

### Basic Usage

```python
from agents import CoderAgent, TesterAgent, ReasoningAgent
from execution import CodeExecutor
from coordinator import ThreEvoCoordinator

# Initialize agents
coder = CoderAgent(api_key=your_api_key)
tester = TesterAgent(api_key=your_api_key)
reasoner = ReasoningAgent(api_key=your_api_key)
executor = CodeExecutor(timeout=10)

# Initialize coordinator
coordinator = ThreEvoCoordinator(
    coder_agent=coder,
    tester_agent=tester,
    reasoning_agent=reasoner,
    executor=executor,
    max_iterations=10
)

# Solve a problem
problem = "Write a function that returns the sum of a list of integers"
result = coordinator.solve(problem)

print(f"Converged: {result['converged']}")
print(f"Code: {result['final_code']}")
```

### Configuration

You can customize settings via YAML config:

```yaml
experiment:
  max_iterations: 10

agents:
  coder:
    model: "claude-sonnet-4-20250514"
    temperature: 0.7
  tester:
    model: "claude-sonnet-4-20250514"
    temperature: 0.7
  reasoning:
    model: "claude-sonnet-4-20250514"
    temperature: 0.0

execution:
  timeout_seconds: 10
```

## PythonSaga Benchmark

ThreEvo includes built-in support for the **PythonSaga benchmark** - a high-quality dataset of 185 hand-crafted problems covering 38 programming concepts.

### Why PythonSaga?

- **Less Saturated**: Unlike HumanEval and MBPP which are heavily benchmarked, PythonSaga is newer and less saturated
- **Balanced Difficulty**: 20% Easy, 40% Medium, 40% Hard problems
- **Concept Diversity**: 38 programming concepts with 5 problems each
- **Real-World Context**: Problems are phrased in human-friendly, contextualized language

### Using PythonSaga

```python
from evaluation.benchmarks import PythonSagaBenchmark

# Load the benchmark
benchmark = PythonSagaBenchmark()
benchmark.load()  # Automatically downloads from GitHub

# Get a problem
problem = benchmark.get_problem_by_index(0)
formatted = benchmark.format_problem_for_threevo(problem)

# Run ThreEvo on it
result = coordinator.solve(formatted)

# Save results
benchmark.save_results([result], 'results.jsonl')
```

### Dataset Source

- **Paper**: "PythonSaga: Redefining the Benchmark to Evaluate Code Generating LLMs" (EMNLP 2024)
- **GitHub**: https://github.com/PythonSaga/ACL2024
- **Authors**: Ankit Yadav, Himanshu Beniwal, Mayank Singh

## Documentation

See [CLAUDE.md](CLAUDE.md) for detailed code documentation and architecture explanation.

See [IMPLEMENTATION.md](IMPLEMENTATION.md) for the full implementation plan and research details.

## License

MIT License

## Citation

If you use ThreEvo in your research, please cite:

```bibtex
@article{threevo2024,
  title={ThreEvo: Three-Agent Evolutionary Framework for Code Generation with Semantic Feedback},
  author={Your Name},
  year={2024}
}
```