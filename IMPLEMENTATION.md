# ThreEvo: Implementation Outline for Claude Code

## Project Overview

ThreEvo is a three-agent evolutionary framework for code generation that uses semantic feedback to improve code correctness. Unlike existing systems that rely solely on execution-based validation, ThreEvo introduces a Reasoning Agent that independently validates both code and tests, providing rich natural language feedback to drive iterative improvement.

### Core Innovation
- **Three-way validation**: Compares actual output (code execution), expected output (test specification), and reasoned solution (independent problem-solving)
- **Rich semantic feedback**: Natural language explanations of errors, not just pass/fail signals
- **Prompt evolution**: Agents iteratively improve their prompts based on historical failures and feedback
- **Break circular dependency**: Reasoning Agent serves as independent ground truth to catch correlated errors in both code and tests

---

## System Architecture

### 1. Three Agent System

#### **Coder Agent (AC)**
- **Input**: Problem specification P, evolved prompt πC
- **Output**: Generated code c = fθ(P, πC)
- **Responsibilities**:
  - Generate code solutions from natural language problem specifications
  - Maintain and evolve its prompt based on feedback from previous iterations
  - Learn from semantic feedback about code errors, logic flaws, and edge cases
  - Track historical failures and incorporate corrective strategies

#### **Tester Agent (AT)**
- **Input**: Problem specification P, evolved prompt πT
- **Output**: Test suite T = gϕ(P, πT) where each test ti = (xi, yi)
- **Responsibilities**:
  - Generate comprehensive test suites including edge cases
  - Create test inputs (xi) and expected outputs (yi)
  - Evolve prompts to generate better test coverage based on past failures
  - Discover corner cases through evolutionary process
  - Learn to write correct test specifications

#### **Reasoning Agent (AR)**
- **Input**: Test input xi, expected output yi, actual output ŷi, code c, problem P
- **Output**: 
  - Boolean correctness signal bi ∈ {correct, incorrect}
  - Detailed natural language feedback fi
- **Responsibilities**:
  - Independently solve each test case: zi = hψ(xi, P)
  - Perform three-way consistency validation
  - Diagnose error sources (code bug, test spec bug, or both)
  - Generate rich semantic feedback explaining:
    - What went wrong
    - Why it's wrong
    - How to fix it
    - Step-by-step correct logic
  - Act as independent ground truth validator

### 2. Three-Way Validation Logic

The Reasoning Agent compares three outputs to diagnose issues:

| yi (expected) | ŷi (actual) | zi (reasoned) | Diagnosis | Action |
|---------------|-------------|---------------|-----------|--------|
| ✓ | ✓ | ✓ | All correct | Accept solution |
| ✓ | ✗ | ✓ | Code error | Feedback to Coder |
| ✗ | ✓ | ✓ | Test spec error | Feedback to Tester |
| ✗ | ✗ | ✓ | Both code and test errors | Feedback to both agents |
| varies | varies | varies | Conflict requires review | Human review or deeper analysis |

### 3. Evolutionary Loop

```
WHILE not converged AND iterations < MAX_ITERATIONS:
    1. Coder Agent generates code c using current prompt πC
    2. Tester Agent generates test suite T using current prompt πT
       (These happen in parallel)
    
    3. Execute code on test inputs:
       FOR each test ti = (xi, yi) in T:
           ŷi = execute(c, xi)
    
    4. Reasoning Agent validates:
       FOR each test ti:
           zi = independently_solve(xi, P)
           bi, fi = compare_and_diagnose(yi, ŷi, zi, c)
    
    5. Prompt Evolution:
       πC_new = evolve_prompt(πC, feedback_for_coder)
       πT_new = evolve_prompt(πT, feedback_for_tester)
    
    6. Check convergence:
       IF all tests pass AND reasoning agent confirms correctness:
           BREAK
```

---

## Implementation Stack

### Core Technologies
- **Python**: Primary implementation language
- **LangChain**: Multi-agent orchestration and coordination
- **LangSmith**: Detailed logging and trace analysis

### Infrastructure Requirements
- GPU access for LLM API calls (or API keys for Claude, GPT, etc.)
- Standard development machine for orchestration
- Basic execution environment with timeouts and resource monitoring

---

## Directory Structure

```
threevo/
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # Abstract base class for all agents
│   ├── coder_agent.py         # Coder Agent implementation
│   ├── tester_agent.py        # Tester Agent implementation
│   ├── reasoning_agent.py     # Reasoning Agent implementation
│   └── orchestration.py       # Three-way validation, feedback generation, and prompt evolution
│
├── execution/
│   ├── __init__.py
│   ├── executor.py            # Test execution logic with timeouts
│   └── safety.py              # Resource limits and security
│
├── coordinator/
│   ├── __init__.py
│   ├── coordinator.py         # Main evolutionary loop coordinator
│   └── state_manager.py       # Track iteration state and history
│
├── evaluation/
│   ├── __init__.py
│   ├── benchmarks/
│   │   ├── humaneval.py
│   │   ├── mbpp.py
│   │   └── pythonsaga.py
│   ├── metrics.py             # Pass@k, convergence, semantic correctness
│   └── baselines.py           # Baseline model comparisons
│
├── config/
│   ├── __init__.py
│   ├── settings.py            # Configuration management
│   └── experiment_config.yaml # Experiment parameters
│
├── experiments/
│   ├── run_experiment.py      # Main experiment runner
│   ├── ablation_studies.py    # Ablation study configurations
│   └── baseline_comparison.py # Baseline experiment runner
│
├── utils/
│   ├── __init__.py
│   ├── logging_config.py      # LangSmith logging setup
│   ├── dataset_loader.py      # Load benchmark datasets
│   └── visualization.py       # Result visualization
│
├── tests/
│   ├── unit/
│   ├── integration/
│   └── end_to_end/
│
├── requirements.txt
├── setup.py
└── README.md
```

---

## Key Components Implementation Details

### 1. Base Agent (agents/base_agent.py)

```python
class BaseAgent:
    """Abstract base class for all agents in ThreEvo"""
    
    def __init__(self, model_name: str, initial_prompt: str):
        self.model_name = model_name
        self.prompt = initial_prompt
        self.history = []
        self.feedback_cache = []
    
    def generate(self, problem: str) -> Any:
        """Generate output based on current prompt"""
        raise NotImplementedError
    
    def evolve_prompt(self, feedback: List[str]) -> str:
        """Update prompt based on accumulated feedback"""
        raise NotImplementedError
    
    def save_state(self) -> dict:
        """Save agent state for checkpointing"""
        raise NotImplementedError
    
    def load_state(self, state: dict):
        """Restore agent from checkpoint"""
        raise NotImplementedError
```

### 2. Coder Agent (agents/coder_agent.py)

```python
class CoderAgent(BaseAgent):
    """Generates code solutions from problem specifications"""
    
    def generate(self, problem: str) -> str:
        """
        Generate code using current evolved prompt
        
        Returns:
            str: Python code as string
        """
        prompt = self._build_generation_prompt(problem)
        code = self._call_llm(prompt)
        self.history.append({
            'problem': problem,
            'code': code,
            'prompt': self.prompt
        })
        return code
    
    def evolve_prompt(self, feedback: List[dict]) -> str:
        """
        Update prompt based on semantic feedback
        
        Args:
            feedback: List of dicts containing:
                - error_type: Type of error
                - explanation: Why the code failed
                - correction_strategy: How to fix it
        
        Returns:
            Updated prompt string
        """
        # Extract patterns from failures
        # Identify common mistakes
        # Generate improved instructions
        # Add edge case handling guidance
        pass
    
    def _build_generation_prompt(self, problem: str) -> str:
        """Combine evolved prompt with current problem"""
        return f"{self.prompt}\n\nProblem: {problem}\n\nCode:"
```

### 3. Tester Agent (agents/tester_agent.py)

```python
class TesterAgent(BaseAgent):
    """Generates test suites from problem specifications"""
    
    def generate(self, problem: str) -> List[Tuple[Any, Any]]:
        """
        Generate test cases including edge cases
        
        Returns:
            List of (input, expected_output) tuples
        """
        prompt = self._build_generation_prompt(problem)
        test_suite = self._call_llm(prompt)
        parsed_tests = self._parse_test_suite(test_suite)
        
        self.history.append({
            'problem': problem,
            'tests': parsed_tests,
            'prompt': self.prompt
        })
        return parsed_tests
    
    def evolve_prompt(self, feedback: List[dict]) -> str:
        """
        Update prompt based on test specification errors
        
        Args:
            feedback: List of dicts containing:
                - incorrect_expectation: What was wrong
                - correct_output: What should have been expected
                - reasoning: Why the test was wrong
        """
        # Learn from incorrect test specifications
        # Improve edge case generation
        # Refine output specification accuracy
        pass
```

### 4. Reasoning Agent (agents/reasoning_agent.py)

```python
class ReasoningAgent(BaseAgent):
    """Independently solves problems through reasoning"""

    def solve(self, problem: str, test_input: Any) -> Any:
        """
        Independently solve the test case through reasoning

        Uses chain-of-thought prompting to work through the problem

        Returns:
            The reasoned solution output
        """
        prompt = f"""
        Problem: {problem}
        Input: {test_input}

        Think step-by-step to solve this problem:
        1. What is the problem asking for?
        2. What is the correct approach?
        3. What should the output be?

        Solution:
        """
        return self._call_llm(prompt)
```

### 5. Orchestration Logic (agents/orchestration.py)

```python
from typing import Any, Tuple, List, Dict

class ThreEvoOrchestration:
    """Handles three-way validation, feedback generation, and prompt evolution"""

    def validate_three_way(
        self,
        expected: Any,
        actual: Any,
        reasoned: Any
    ) -> Dict[str, Any]:
        """
        Perform three-way validation logic

        Returns diagnostic information about error type and target
        """
        matches = {
            'expected_actual': self._compare(expected, actual),
            'expected_reasoned': self._compare(expected, reasoned),
            'actual_reasoned': self._compare(actual, reasoned)
        }

        if all(matches.values()):
            return {'type': 'correct', 'target': None}
        elif matches['expected_reasoned'] and matches['actual_reasoned']:
            return {'type': 'test_error', 'target': 'tester'}
        elif matches['expected_actual'] and not matches['actual_reasoned']:
            return {'type': 'reasoning_conflict', 'target': 'review'}
        else:
            return self._detailed_diagnosis(expected, actual, reasoned)

    def generate_feedback(
        self,
        diagnosis: Dict[str, Any],
        problem: str,
        test_input: Any,
        expected: Any,
        actual: Any,
        reasoned: Any,
        code: str
    ) -> Dict[str, str]:
        """
        Generate rich semantic feedback for both agents

        Returns:
            dict with 'coder_feedback' and 'tester_feedback' keys
        """
        feedback = {'coder_feedback': [], 'tester_feedback': []}

        if diagnosis['type'] == 'correct':
            return feedback

        elif diagnosis['type'] == 'code_error':
            feedback['coder_feedback'].append(
                self._explain_code_error(problem, test_input, expected, actual, reasoned, code)
            )

        elif diagnosis['type'] == 'test_error':
            feedback['tester_feedback'].append(
                self._explain_test_error(problem, test_input, expected, reasoned)
            )

        elif diagnosis['type'] == 'both_errors':
            coder_fb, tester_fb = self._explain_both_errors(
                problem, test_input, expected, actual, reasoned, code
            )
            feedback['coder_feedback'].append(coder_fb)
            feedback['tester_feedback'].append(tester_fb)

        return feedback

    def evolve_prompt(
        self,
        current_prompt: str,
        feedback_history: List[str],
        agent_type: str
    ) -> str:
        """
        Evolve agent prompt based on accumulated feedback

        Args:
            current_prompt: Current prompt string
            feedback_history: List of feedback from recent iterations
            agent_type: 'coder' or 'tester'

        Returns:
            Updated prompt string
        """
        # Extract patterns from failures
        # Identify common mistakes
        # Generate improved instructions
        # Add edge case handling guidance
        pass

    def _compare(self, a: Any, b: Any) -> bool:
        """Compare two outputs for equality"""
        return a == b

    def _detailed_diagnosis(self, expected: Any, actual: Any, reasoned: Any) -> Dict[str, Any]:
        """Perform detailed diagnosis when simple matching fails"""
        # Could be code error, test error, or both
        # Use heuristics or additional LLM call to determine
        pass

    def _explain_code_error(self, problem, test_input, expected, actual, reasoned, code) -> str:
        """Generate explanation for code errors"""
        pass

    def _explain_test_error(self, problem, test_input, expected, reasoned) -> str:
        """Generate explanation for test specification errors"""
        pass

    def _explain_both_errors(self, problem, test_input, expected, actual, reasoned, code) -> Tuple[str, str]:
        """Generate explanations when both code and test have errors"""
        pass
```

### 6. Coordinator (coordinator/coordinator.py)

```python
from agents.orchestration import ThreEvoOrchestration

class ThreEvoCoordinator:
    """Main evolutionary loop coordinator"""

    def __init__(
        self,
        coder_agent: CoderAgent,
        tester_agent: TesterAgent,
        reasoning_agent: ReasoningAgent,
        executor: CodeExecutor,
        max_iterations: int = 10
    ):
        self.coder = coder_agent
        self.tester = tester_agent
        self.reasoner = reasoning_agent
        self.executor = executor
        self.max_iterations = max_iterations
        self.orchestration = ThreEvoOrchestration()
    
    def solve(self, problem: str) -> dict:
        """
        Main evolutionary loop
        
        Returns:
            dict containing:
                - final_code: The solution
                - iterations: Number of iterations
                - all_feedback: Feedback history
                - converged: Whether solution converged
        """
        iteration = 0
        converged = False
        
        while iteration < self.max_iterations and not converged:
            # Step 1: Parallel generation
            code = self.coder.generate(problem)
            tests = self.tester.generate(problem)
            
            # Step 2: Execute code on tests
            execution_results = []
            for test_input, expected_output in tests:
                actual_output = self.executor.execute(code, test_input)
                execution_results.append({
                    'input': test_input,
                    'expected': expected_output,
                    'actual': actual_output
                })
            
            # Step 3: Reasoning validation
            coder_feedback = []
            tester_feedback = []
            all_correct = True

            for result in execution_results:
                # Reasoning agent independently solves the test case
                reasoned_solution = self.reasoner.solve(problem, result['input'])

                # Three-way validation
                diagnosis = self.orchestration.validate_three_way(
                    expected=result['expected'],
                    actual=result['actual'],
                    reasoned=reasoned_solution
                )

                if diagnosis['type'] != 'correct':
                    all_correct = False

                    # Generate semantic feedback
                    feedback = self.orchestration.generate_feedback(
                        diagnosis=diagnosis,
                        problem=problem,
                        test_input=result['input'],
                        expected=result['expected'],
                        actual=result['actual'],
                        reasoned=reasoned_solution,
                        code=code
                    )

                    coder_feedback.extend(feedback['coder_feedback'])
                    tester_feedback.extend(feedback['tester_feedback'])
            
            # Step 4: Check convergence
            if all_correct:
                converged = True
                break
            
            # Step 5: Evolve prompts using orchestration
            if coder_feedback:
                new_prompt = self.orchestration.evolve_prompt(
                    current_prompt=self.coder.prompt,
                    feedback_history=coder_feedback,
                    agent_type='coder'
                )
                self.coder.prompt = new_prompt

            if tester_feedback:
                new_prompt = self.orchestration.evolve_prompt(
                    current_prompt=self.tester.prompt,
                    feedback_history=tester_feedback,
                    agent_type='tester'
                )
                self.tester.prompt = new_prompt
            
            iteration += 1
        
        return {
            'final_code': code,
            'iterations': iteration,
            'converged': converged,
            'final_tests': tests
        }
```

### 7. Code Executor (execution/executor.py)

```python
import subprocess
import tempfile
import json
import os
import sys
from typing import Any

class CodeExecutor:
    """Execute code with timeout and resource limits"""

    def __init__(self, timeout: int = 10):
        self.timeout = timeout

    def execute(self, code: str, test_input: Any) -> Any:
        """
        Execute code with test input

        Returns:
            Output from code execution or error message
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "solution.py")

            # Write code that includes the test input execution
            full_code = f"""
import json
import sys

{code}

# Execute with test input
test_input = {repr(test_input)}
try:
    result = solution(test_input)  # Assuming function is named 'solution'
    print(json.dumps({{'result': result}}))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
"""

            with open(code_path, 'w') as f:
                f.write(full_code)

            # Execute with timeout
            try:
                result = subprocess.run(
                    [sys.executable, code_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=tmpdir
                )

                # Parse output
                if result.returncode == 0:
                    output = json.loads(result.stdout.strip())
                    return output.get('result') if 'result' in output else output
                else:
                    return {'error': result.stderr}

            except subprocess.TimeoutExpired:
                return {'error': 'Execution timeout'}
            except Exception as e:
                return {'error': str(e)}
```

---

## Evaluation Framework

### Benchmarks to Implement

1. **HumanEval** (164 problems)
   - Hand-written programming problems
   - Function signatures and docstrings provided
   - Tests fundamental algorithms and data structures

2. **MBPP** (974 problems)
   - Entry-level programming problems
   - Basic programming knowledge assessment

3. **PythonSaga** (185 problems)
   - 38 programming concepts across 3 difficulty levels
   - Balanced concept diversity

### Metrics to Track

```python
class EvaluationMetrics:
    """Calculate and track evaluation metrics"""
    
    def pass_at_k(self, solutions: List[str], k: int) -> float:
        """
        Percentage of problems with at least one correct solution in k samples
        
        Tests consistency and reliability of the framework
        """
        pass
    
    def iterations_to_convergence(
        self, 
        problem_results: List[dict]
    ) -> dict:
        """
        Number of evolutionary cycles to reach correct solutions
        
        Returns statistics: mean, median, std, distribution
        """
        pass
    
    def semantic_correctness_rate(
        self,
        solutions: List[str],
        test_suites: List[List[tuple]]
    ) -> float:
        """
        Percentage passing all tests including edge cases discovered during evolution
        
        Demonstrates handling of corner cases
        """
        pass
```

### Baseline Comparisons

Compare against:
1. GPT-5 direct generation
2. Claude 4.5 Sonnet direct generation
3. Gemini 2.5 Pro direct generation
4. CoCoEvo
5. AgentCoder
6. LLMLOOP

### Ablation Studies

Test these variants:
1. **ThreEvo without Reasoning Agent**: Only Coder and Tester with execution feedback
2. **ThreEvo without evolutionary prompts**: Fixed prompts throughout
3. **ThreEvo with execution-only feedback**: Binary pass/fail signals instead of semantic feedback
4. **ThreEvo with different base models**: Swap Claude Sonnet 4.5 with GPT-4, Gemini, etc.

---

## Configuration Management

### experiment_config.yaml

```yaml
experiment:
  name: "threevo_humaneval_baseline"
  benchmark: "humaneval"  # humaneval, mbpp, pythonsaga
  max_iterations: 10
  num_problems: 164  # null for all problems

agents:
  coder:
    model: "claude-sonnet-4-20250514"
    temperature: 0.7
    initial_prompt: "You are an expert programmer..."
  
  tester:
    model: "claude-sonnet-4-20250514"
    temperature: 0.7
    initial_prompt: "You are an expert test designer..."
  
  reasoning:
    model: "claude-sonnet-4-20250514"
    temperature: 0.0  # Deterministic reasoning
    initial_prompt: "You are a reasoning expert..."

execution:
  timeout_seconds: 10
  parallel_executions: 4

evolution:
  enable_prompt_evolution: true
  feedback_aggregation: "weighted"  # weighted, latest, all
  learning_rate: 0.1

storage:
  results_dir: "results/"
  save_intermediate: true
  checkpoint_frequency: 5  # Save every N iterations

logging:
  langsmith_project: "threevo-experiments"
  log_level: "INFO"
  save_traces: true
```

---

## Implementation Timeline (4 Weeks)

### Week 1: Core Architecture
**Days 1-2: Agent Framework**
- [ ] Implement BaseAgent abstract class
- [ ] Implement CoderAgent with basic generation
- [ ] Implement TesterAgent with basic test generation
- [ ] Implement ReasoningAgent with independent problem solving
- [ ] Set up LangChain integration for LLM calls

**Days 3-4: Execution Environment**
- [ ] Implement CodeExecutor with subprocess and timeouts
- [ ] Set up timeout handling and error catching
- [ ] Test basic code execution pipeline
- [ ] Handle execution errors and timeouts gracefully

**Days 5-7: Orchestration**
- [ ] Implement ThreEvoCoordinator with main evolutionary loop
- [ ] Implement state management and checkpointing to files
- [ ] Test end-to-end flow with simple examples
- [ ] Add result persistence to JSON/file storage

### Week 2: Orchestration and Evolution
**Days 8-9: Orchestration Logic**
- [ ] Implement ThreEvoOrchestration class in agents/orchestration.py
- [ ] Implement three-way validation logic (validate_three_way)
- [ ] Implement feedback generation (generate_feedback)
- [ ] Design feedback format and structure
- [ ] Test validation and feedback on examples

**Days 10-11: Prompt Evolution**
- [ ] Implement prompt evolution mechanism (evolve_prompt)
- [ ] Create failure pattern extraction
- [ ] Build prompt update strategies
- [ ] Integrate orchestration with coordinator
- [ ] Test evolution over multiple iterations

**Days 12-14: Storage and Logging**
- [ ] Implement file-based storage for results (JSON/pickle)
- [ ] Create result serialization and deserialization
- [ ] Integrate LangSmith logging
- [ ] Create debugging and visualization tools

### Week 3: Evaluation
**Days 15-16: Benchmark Integration**
- [ ] Load and parse HumanEval dataset
- [ ] Load and parse MBPP dataset
- [ ] Load and parse PythonSaga dataset
- [ ] Implement dataset-specific preprocessing

**Days 17-18: Metrics and Baselines**
- [ ] Implement Pass@k calculation
- [ ] Implement iterations to convergence tracking
- [ ] Implement semantic correctness rate
- [ ] Set up baseline model comparisons

**Days 19-21: Experiments**
- [ ] Run full evaluation on HumanEval
- [ ] Run full evaluation on MBPP
- [ ] Run full evaluation on PythonSaga
- [ ] Run ablation studies
- [ ] Run baseline comparisons
- [ ] Collect comprehensive results

### Week 4: Analysis and Documentation
**Days 22-24: Analysis**
- [ ] Analyze results across benchmarks
- [ ] Compare against baselines
- [ ] Analyze ablation study results
- [ ] Create visualizations and plots
- [ ] Identify strengths and weaknesses

**Days 25-28: Paper Writing**
- [ ] Write introduction and related work
- [ ] Write methodology section
- [ ] Write results section
- [ ] Write discussion and conclusion
- [ ] Create tables and figures
- [ ] Proofread and finalize

---

## Key Design Decisions

### 1. LLM Selection
- **Primary**: Claude Sonnet 4.5 for all agents
- **Rationale**: State-of-the-art reasoning and code generation
- **Ablation**: Test with GPT-4, Gemini to validate model-agnostic approach

### 2. Feedback Format
Use structured natural language feedback:
```json
{
  "correctness": false,
  "error_type": "code_error",
  "diagnosis": "The code fails to handle negative numbers correctly",
  "explanation": "When the input is negative, the code returns None instead of the absolute value",
  "correct_approach": "Add a check: if x < 0: return -x",
  "reasoning_trace": "Step 1: ... Step 2: ..."
}
```

### 3. Prompt Evolution Strategy
- Accumulate feedback patterns
- Identify common failure modes
- Add specific guidance for edge cases
- Maintain rolling window of last N failures
- Use LLM to synthesize new prompt from old prompt + feedback

### 4. Convergence Criteria
Solution converges when:
1. All test cases pass (expected == actual)
2. Reasoning Agent confirms correctness (expected == reasoned == actual)
3. OR maximum iterations reached

### 5. Safety and Security
- Timeout on all executions (default 10s)
- Code execution in isolated temporary directories
- Limited to Python standard library unless specified
- No network access (not enforced by default - use caution with untrusted code)
- Basic error catching and resource monitoring

---

## Testing Strategy

### Unit Tests
- Test each agent independently
- Test validation logic with known inputs
- Test prompt evolution mechanism
- Test executor with various code samples

### Integration Tests
- Test agent communication via message bus
- Test full evolutionary loop with simple problems
- Test convergence detection
- Test error handling and recovery

### End-to-End Tests
- Run on small subset of benchmark problems
- Verify metric calculations
- Test baseline comparisons
- Validate logging and storage

---

## Monitoring and Debugging

### LangSmith Integration
- Log all LLM calls with prompts and responses
- Track reasoning traces
- Monitor token usage
- Analyze failure patterns

### Metrics to Monitor During Development
- Average iterations to convergence
- Feedback quality (manual review)
- Prompt evolution effectiveness
- Execution success rate
- Error types distribution

### Debugging Tools
- Visualize evolution trajectory
- Inspect prompt changes over iterations
- Review validation decisions
- Replay failed attempts

---

## Common Pitfalls and Solutions

### 1. Circular Errors
**Problem**: Both code and test are wrong in complementary ways
**Solution**: Reasoning Agent breaks the loop by providing independent ground truth

### 2. Prompt Drift
**Problem**: Prompts become too long or unfocused
**Solution**: Implement prompt summarization, keep fixed structure

### 3. Execution Failures
**Problem**: Code crashes or times out
**Solution**: Capture stderr and exceptions, provide as feedback, graceful timeout handling with clear error messages

### 4. Inconsistent Reasoning
**Problem**: Reasoning Agent gives conflicting feedback
**Solution**: Use temperature=0 for deterministic reasoning, add consistency checks

### 5. Slow Convergence
**Problem**: Taking too many iterations
**Solution**: Improve feedback quality, better prompt evolution, curriculum learning

---

## Success Criteria

The implementation is successful if:
1. ✅ System runs end-to-end on all three benchmarks
2. ✅ Three-way validation correctly diagnoses error sources
3. ✅ Semantic feedback improves over iterations
4. ✅ Pass@k scores competitive with or better than baselines
5. ✅ Converges in fewer iterations than execution-only systems
6. ✅ Catches test specification errors that other systems miss
7. ✅ All ablation studies completed
8. ✅ Results reproducible and well-documented

---

## Future Extensions (Post-Initial Implementation)

1. **Multi-language Support**: Extend beyond Python
2. **Human-in-the-Loop**: Allow human feedback integration
3. **Meta-learning**: Learn prompt evolution strategies
4. **Parallel Exploration**: Try multiple solutions simultaneously
5. **Transfer Learning**: Use learnings from one problem on related problems
6. **Formal Verification**: Integrate SMT solvers for mathematical correctness
7. **Real-world Datasets**: Test on production code repositories

---

## Quick Start Commands

```bash
# Setup environment
pip install -r requirements.txt

# Run single experiment
python experiments/run_experiment.py --config config/experiment_config.yaml

# Run full evaluation
python experiments/run_experiment.py --benchmark humaneval --all
python experiments/run_experiment.py --benchmark mbpp --all
python experiments/run_experiment.py --benchmark pythonsaga --all

# Run ablation studies
python experiments/ablation_studies.py --study no_reasoning
python experiments/ablation_studies.py --study no_evolution
python experiments/ablation_studies.py --study execution_only

# Compare baselines
python experiments/baseline_comparison.py --all

# Visualize results
python utils/visualization.py --results results/humaneval/ --output plots/
```

---

## References

Key papers to implement and compare:
- [1] Chen et al. 2021 - Codex (for HumanEval benchmark)
- [2] Cao et al. 2023 - CoCoEvo (baseline)
- [3] Huang et al. 2024 - GenX (concepts)
- [4] Zhang et al. 2024 - LLMLOOP (baseline)
- [5] Huang et al. 2024 - AgentCoder (baseline)
- [8] Shinn et al. 2023 - Reflexion (verbal reinforcement learning)
- [9] Wei et al. 2022 - Chain-of-Thought (reasoning approach)

---

This implementation outline provides Claude Code with a comprehensive understanding of the ThreEvo system, its architecture, and a clear path to implementation. The outline emphasizes the core innovation (three-way validation with semantic feedback), provides detailed component specifications, and includes a realistic timeline with specific deliverables.