# ThreEvo: Codebase Documentation

This document provides comprehensive documentation of the ThreEvo codebase architecture, implementation details, and usage patterns.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Module Documentation](#module-documentation)
3. [Data Flow](#data-flow)
4. [Key Algorithms](#key-algorithms)
5. [Extension Points](#extension-points)
6. [Best Practices](#best-practices)

---

## Architecture Overview

ThreEvo implements a three-agent evolutionary framework with the following core components:

```
┌─────────────────────────────────────────────────────────────┐
│                    ThreEvoCoordinator                       │
│                  (Main Evolutionary Loop)                   │
└─────────────────────────────────────────────────────────────┘
                              │
                              │ orchestrates
                              ▼
        ┌──────────────────────────────────────────┐
        │                                          │
        ▼                    ▼                     ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ CoderAgent   │    │ TesterAgent  │    │ ReasoningAgent│
│ (Generates   │    │ (Generates   │    │ (Independently│
│  Code)       │    │  Tests)      │    │  Validates)   │
└──────────────┘    └──────────────┘    └──────────────┘
        │                    │                     │
        └────────────┬───────┴─────────────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │  ThreEvoOrchestration    │
        │  - Three-way validation  │
        │  - Feedback generation   │
        │  - Prompt evolution      │
        └──────────────────────────┘
                     │
                     ▼
        ┌──────────────────────────┐
        │     CodeExecutor         │
        │  (Executes & validates)  │
        └──────────────────────────┘
```

### Core Innovation: Three-Way Validation

The key innovation is comparing three outputs:

| Output Type | Source | Purpose |
|------------|--------|---------|
| **Expected** (yi) | Tester Agent | Test specification |
| **Actual** (ŷi) | Code execution | Coder Agent output |
| **Reasoned** (zi) | Reasoning Agent | Independent ground truth |

**Validation Logic:**

```
if yi == ŷi == zi → ✅ All correct
if yi == zi ≠ ŷi  → ❌ Code error (feedback to Coder)
if ŷi == zi ≠ yi  → ❌ Test error (feedback to Tester)
if yi == ŷi ≠ zi  → ⚠️  Reasoning conflict
else              → ❌ Multiple errors
```

---

## Module Documentation

### 1. Agents Module (`agents/`)

#### BaseAgent (`base_agent.py`)

Abstract base class for all agents.

**Key Methods:**

```python
class BaseAgent(ABC):
    def __init__(self, model_name, temperature, initial_prompt, api_key)
        # Initialize LLM and agent state

    @abstractmethod
    def generate(self, problem: str) -> Any
        # Generate output based on problem

    def _call_llm(self, prompt: str, system_prompt: Optional[str]) -> str
        # Call LLM with prompt

    def save_state(self) -> Dict[str, Any]
        # Save agent state for checkpointing

    def load_state(self, state: Dict[str, Any])
        # Restore agent from checkpoint
```

**State Management:**

Each agent maintains:
- `prompt`: Current evolved prompt
- `history`: List of past generations
- `feedback_cache`: Accumulated feedback for evolution

#### CoderAgent (`coder_agent.py`)

Generates code solutions from problem specifications.

**Key Features:**
- Extracts code from markdown-formatted responses
- Maintains history of generated code
- Default prompt emphasizes correctness and edge case handling

**Code Extraction:**

The agent automatically handles responses in these formats:
- Plain code
- Markdown code blocks (```python ... ```)
- Markdown code blocks (``` ... ```)

**Example Usage:**

```python
coder = CoderAgent(api_key="your_key")
code = coder.generate("Write a function to reverse a string")
# Returns: Clean Python code string
```

#### TesterAgent (`tester_agent.py`)

Generates comprehensive test suites.

**Key Features:**
- Generates tests in JSON format
- Automatically parses test specifications
- Emphasizes edge cases and boundary conditions

**Test Format:**

```json
[
    {"input": [1, 2, 3], "expected": 6},
    {"input": [], "expected": 0},
    {"input": [-1, 1], "expected": 0}
]
```

**Example Usage:**

```python
tester = TesterAgent(api_key="your_key")
tests = tester.generate("Write a function to sum integers")
# Returns: List of (input, expected) tuples
```

#### ReasoningAgent (`reasoning_agent.py`)

Independently solves problems through chain-of-thought reasoning.

**Key Features:**
- Uses temperature=0.0 for deterministic reasoning
- Chain-of-thought prompting for step-by-step analysis
- Extracts final answer from reasoning trace

**Reasoning Process:**

1. Understand the problem
2. Analyze the input
3. Determine the correct approach
4. Walk through the solution step-by-step
5. Provide final answer with "FINAL ANSWER:" marker

**Example Usage:**

```python
reasoner = ReasoningAgent(api_key="your_key")
solution = reasoner.solve(problem="Sum integers", test_input=[1, 2, 3])
# Returns: 6 (with reasoning trace in history)
```

#### ThreEvoOrchestration (`orchestration.py`)

Handles three-way validation, feedback generation, and prompt evolution.

**Key Methods:**

```python
class ThreEvoOrchestration:
    def validate_three_way(expected, actual, reasoned) -> Dict
        # Performs three-way comparison
        # Returns: {'type': error_type, 'target': agent_to_feedback}

    def generate_feedback(diagnosis, problem, test_input, ...) -> Dict
        # Generates semantic feedback
        # Returns: {'coder_feedback': [...], 'tester_feedback': [...]}

    def evolve_prompt(current_prompt, feedback_history, agent_type) -> str
        # Evolves prompt based on error patterns
        # Returns: Updated prompt string
```

**Validation Types:**

- `correct`: All three outputs match
- `code_error`: Code is wrong, test is right
- `test_error`: Test specification is wrong
- `both_errors`: Both code and test are wrong
- `reasoning_conflict`: Rare case requiring review

**Feedback Structure:**

Feedback is rich and semantic, including:
- What went wrong
- Why it's wrong
- The correct output
- How to fix it
- Context (problem, code, inputs)

**Prompt Evolution:**

Extracts patterns from feedback:
- Edge case handling issues
- Empty input handling
- Negative number handling
- Boundary errors
- Type handling problems

Generates additional guidance to add to prompts.

---

### 2. Execution Module (`execution/`)

#### CodeExecutor (`executor.py`)

Executes code with timeout and error handling.

**Key Features:**
- Subprocess-based execution
- Configurable timeout (default 10s)
- JSON-based input/output
- Comprehensive error handling

**Execution Flow:**

1. Create temporary directory
2. Write code with execution wrapper
3. Execute with subprocess + timeout
4. Parse JSON output
5. Clean up temporary files

**Error Handling:**

Returns error dictionary for:
- Execution errors (exceptions)
- Timeouts
- Parsing failures

**Example Usage:**

```python
executor = CodeExecutor(timeout=10)
result = executor.execute(code, test_input=[1, 2, 3])
# Returns: 6 or {'error': 'error message'}
```

**Batch Execution:**

```python
results = executor.execute_batch(code, [input1, input2, input3])
# Returns: List of results
```

---

### 3. Coordinator Module (`coordinator/`)

#### ThreEvoCoordinator (`coordinator.py`)

Main evolutionary loop coordinator.

**Initialization:**

```python
coordinator = ThreEvoCoordinator(
    coder_agent=coder,
    tester_agent=tester,
    reasoning_agent=reasoner,
    executor=executor,
    max_iterations=10
)
```

**Main Solve Loop:**

```python
def solve(problem: str) -> Dict:
    for iteration in range(max_iterations):
        # 1. Generate code and tests in parallel
        code = coder.generate(problem)
        tests = tester.generate(problem)

        # 2. Execute code on all tests
        execution_results = [executor.execute(code, inp) for inp, exp in tests]

        # 3. Reasoning validation with three-way comparison
        for result in execution_results:
            reasoned = reasoner.solve(problem, result['input'])
            diagnosis = orchestration.validate_three_way(
                expected=result['expected'],
                actual=result['actual'],
                reasoned=reasoned
            )

            # 4. Generate semantic feedback
            feedback = orchestration.generate_feedback(
                diagnosis, problem, result['input'], ...
            )

        # 5. Check convergence
        if all_tests_pass:
            break

        # 6. Evolve prompts based on feedback
        if coder_feedback:
            coder.prompt = orchestration.evolve_prompt(...)
        if tester_feedback:
            tester.prompt = orchestration.evolve_prompt(...)

    return {
        'final_code': code,
        'final_tests': tests,
        'iterations': iteration,
        'converged': all_tests_pass,
        'history': history
    }
```

**Return Value:**

```python
{
    'final_code': str,           # Final generated code
    'final_tests': List[Tuple],  # Final test suite
    'iterations': int,           # Number of iterations taken
    'converged': bool,           # Whether solution converged
    'history': List[Dict]        # Full iteration history
}
```

**Checkpointing:**

```python
coordinator.save_checkpoint('checkpoint.json')
coordinator.load_checkpoint('checkpoint.json')
```

#### StateManager (`state_manager.py`)

Manages experiment results and history.

**Key Methods:**

```python
state_manager = StateManager(results_dir='results/')

# Save individual result
state_manager.save_result(problem_id='problem_1', result=result)

# Load result
result = state_manager.load_result('results/problem_1_20240101_120000.json')

# Get all results for a problem
files = state_manager.get_all_results(problem_id='problem_1')

# Save summary
state_manager.save_summary(results, filename='summary.json')
```

**Result File Format:**

```json
{
    "problem_id": "problem_1",
    "timestamp": "20240101_120000",
    "result": {
        "final_code": "...",
        "converged": true,
        "iterations": 3
    },
    "metadata": {}
}
```

---

### 4. Config Module (`config/`)

#### Settings (`settings.py`)

Configuration management with YAML support.

**Usage:**

```python
# Load default settings
settings = Settings()

# Load from file
settings = Settings(config_path='config/experiment_config.yaml')

# Get values with dot notation
model = settings.get('agents.coder.model')
timeout = settings.get('execution.timeout_seconds', default=10)

# Set values
settings.set('experiment.max_iterations', 15)

# Save to file
settings.save_to_file('new_config.yaml')
```

**Configuration Structure:**

```yaml
experiment:
  name: "experiment_name"
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

storage:
  results_dir: "results/"
  save_intermediate: true
```

---

## Data Flow

### Complete Example Flow

```
1. Problem Definition
   └─> "Write a function to sum a list of integers"

2. Parallel Generation (Iteration 1)
   ├─> CoderAgent generates:
   │   def solution(lst):
   │       return sum(lst)
   │
   └─> TesterAgent generates:
       [([1,2,3], 6), ([], 0), ([-1,1], 0)]

3. Execution
   ├─> Test 1: execute(code, [1,2,3]) → 6
   ├─> Test 2: execute(code, []) → 0
   └─> Test 3: execute(code, [-1,1]) → 0

4. Reasoning Validation (for each test)
   ├─> Test 1:
   │   ├─> ReasoningAgent.solve(problem, [1,2,3]) → 6
   │   ├─> validate_three_way(6, 6, 6) → 'correct'
   │   └─> No feedback
   │
   ├─> Test 2:
   │   ├─> ReasoningAgent.solve(problem, []) → 0
   │   ├─> validate_three_way(0, 0, 0) → 'correct'
   │   └─> No feedback
   │
   └─> Test 3:
       ├─> ReasoningAgent.solve(problem, [-1,1]) → 0
       ├─> validate_three_way(0, 0, 0) → 'correct'
       └─> No feedback

5. Convergence Check
   └─> All tests passed → CONVERGED ✓

6. Return Result
   └─> {
         'final_code': 'def solution(lst): return sum(lst)',
         'converged': True,
         'iterations': 1
       }
```

### Error Case Flow

```
Scenario: Code error detected

1. Execution produces wrong output
   execute(code, [1,2,3]) → 7  (WRONG)

2. Reasoning validates
   ReasoningAgent.solve(problem, [1,2,3]) → 6  (CORRECT)

3. Three-way validation
   validate_three_way(expected=6, actual=7, reasoned=6)
   └─> expected == reasoned ≠ actual
   └─> diagnosis: 'code_error', target: 'coder'

4. Generate feedback
   "CODE ERROR DETECTED:
    Your code returned 7 but should return 6.
    Review your logic for off-by-one errors..."

5. Evolve coder prompt
   Add: "Pay special attention to boundary conditions"

6. Next iteration with updated prompt
```

---

## Key Algorithms

### Three-Way Validation Algorithm

```python
def validate_three_way(expected, actual, reasoned):
    """
    Compare three outputs to diagnose error source

    Truth table:
    exp  act  rea  → diagnosis
    ───  ───  ───  ────────────────
     ✓    ✓    ✓  → correct
     ✓    ✗    ✓  → code_error
     ✗    ✓    ✓  → test_error
     ✓    ✓    ✗  → reasoning_conflict
     ✗    ✗    *  → both_errors or unclear
    """
    matches = {
        'exp_act': expected == actual,
        'exp_rea': expected == reasoned,
        'act_rea': actual == reasoned
    }

    if all(matches.values()):
        return 'correct'
    elif matches['exp_rea'] and not matches['exp_act']:
        return 'code_error'
    elif matches['act_rea'] and not matches['exp_act']:
        return 'test_error'
    elif matches['exp_act'] and not matches['act_rea']:
        return 'reasoning_conflict'
    else:
        return 'both_errors'
```

### Prompt Evolution Algorithm

```python
def evolve_prompt(current_prompt, feedback_history, agent_type):
    """
    Evolve prompt based on error patterns

    Steps:
    1. Extract error patterns from feedback
    2. Identify common mistakes
    3. Generate specific guidance
    4. Append to current prompt
    """
    # Pattern extraction
    patterns = []
    feedback_text = " ".join(feedback_history).lower()

    if "edge case" in feedback_text:
        patterns.append("edge_case_handling")
    if "empty" in feedback_text:
        patterns.append("empty_input")
    # ... more pattern detection

    # Generate guidance
    guidance = []
    for pattern in patterns:
        if pattern == "edge_case_handling":
            guidance.append("Pay special attention to edge cases")
        # ... more guidance generation

    # Append to prompt
    evolved_prompt = f"{current_prompt}\n\nAdditional guidance:\n{guidance}"
    return evolved_prompt
```

---

## Extension Points

### 1. Custom Agents

Create custom agents by extending `BaseAgent`:

```python
from agents.base_agent import BaseAgent

class CustomAgent(BaseAgent):
    def generate(self, problem: str):
        # Your custom generation logic
        prompt = self._build_prompt(problem)
        output = self._call_llm(prompt)
        return output

    def _build_prompt(self, problem: str):
        # Custom prompt building
        return f"{self.prompt}\n\nProblem: {problem}"
```

### 2. Custom Validation Logic

Extend `ThreEvoOrchestration` for custom validation:

```python
from agents.orchestration import ThreEvoOrchestration

class CustomOrchestration(ThreEvoOrchestration):
    def validate_three_way(self, expected, actual, reasoned):
        # Custom validation logic
        # E.g., fuzzy matching, semantic similarity, etc.
        return super().validate_three_way(expected, actual, reasoned)
```

### 3. Custom Executors

Create specialized executors for different languages:

```python
from execution.executor import CodeExecutor

class JavaScriptExecutor(CodeExecutor):
    def execute(self, code: str, test_input: Any):
        # Execute JavaScript code with Node.js
        # Return results in same format
        pass
```

### 4. Benchmark Integration

Add custom benchmarks:

```python
class CustomBenchmark:
    def load_problems(self) -> List[Dict]:
        # Load your benchmark problems
        return [
            {'id': 'p1', 'description': '...', 'tests': [...]},
            # ...
        ]

    def evaluate(self, problem_id: str, solution: str) -> Dict:
        # Evaluate solution on benchmark
        pass
```

---

## Best Practices

### 1. Error Handling

Always check for execution errors:

```python
result = executor.execute(code, test_input)
if isinstance(result, dict) and 'error' in result:
    print(f"Execution error: {result['error']}")
    # Handle error appropriately
```

### 2. Checkpointing

Save checkpoints during long experiments:

```python
for i, problem in enumerate(problems):
    result = coordinator.solve(problem)

    if i % 10 == 0:
        coordinator.save_checkpoint(f'checkpoint_{i}.json')
```

### 3. Configuration Management

Use YAML configs for reproducibility:

```python
# Save config with results
settings.save_to_file('results/experiment_config.yaml')
```

### 4. Logging

Use LangSmith for detailed tracing:

```python
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'threevo-experiments'
```

### 5. Prompt Engineering

Provide clear, specific prompts:

```python
# Good: Specific problem with examples
problem = """
Write a function 'solution' that takes a list of integers
and returns their sum.

Examples:
- solution([1, 2, 3]) → 6
- solution([]) → 0
"""

# Bad: Vague problem
problem = "Sum some numbers"
```

### 6. Test Quality

Ensure test specifications are correct:

```python
# The TesterAgent might generate wrong expected outputs
# The ReasoningAgent catches these via three-way validation
# But good initial prompts help
```

---

## Troubleshooting

### Common Issues

**Issue: Agent not generating valid output**
- Check API key is set correctly
- Verify prompt format
- Review LLM temperature settings

**Issue: Code execution timing out**
- Increase timeout: `CodeExecutor(timeout=30)`
- Check for infinite loops in generated code

**Issue: Tests not parsing correctly**
- Review TesterAgent output format
- Check JSON parsing in `_parse_test_suite`

**Issue: Not converging**
- Increase max_iterations
- Review feedback quality
- Check if problem is well-specified

---

## Performance Optimization

### 1. Batch Processing

Process multiple problems in parallel:

```python
from concurrent.futures import ThreadPoolExecutor

def solve_problem(problem):
    coordinator = ThreEvoCoordinator(...)
    return coordinator.solve(problem)

with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(solve_problem, problems))
```

### 2. Caching

Cache reasoning results for common inputs:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_reasoning(problem, test_input):
    return reasoner.solve(problem, test_input)
```

### 3. Early Stopping

Stop if no improvement after N iterations:

```python
no_improvement_count = 0
for iteration in range(max_iterations):
    # ... solve ...
    if not improved:
        no_improvement_count += 1
    else:
        no_improvement_count = 0

    if no_improvement_count >= 3:
        break  # Early stop
```

---

## Testing

### Unit Tests

```python
# tests/unit/test_agents.py
def test_coder_agent():
    coder = CoderAgent(api_key=test_key)
    code = coder.generate("Return 42")
    assert "def solution" in code

# tests/unit/test_orchestration.py
def test_three_way_validation():
    orch = ThreEvoOrchestration()
    diagnosis = orch.validate_three_way(5, 5, 5)
    assert diagnosis['type'] == 'correct'
```

### Integration Tests

```python
# tests/integration/test_coordinator.py
def test_full_pipeline():
    coordinator = ThreEvoCoordinator(...)
    result = coordinator.solve("Sum integers")
    assert result['converged']
    assert 'final_code' in result
```

---

This documentation provides a complete reference for understanding and extending the ThreEvo codebase. For more details on the research motivation and experimental design, see [IMPLEMENTATION.md](IMPLEMENTATION.md).
