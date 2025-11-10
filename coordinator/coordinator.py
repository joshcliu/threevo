"""
ThreEvo Coordinator

Main evolutionary loop coordinator that orchestrates the three agents.
"""

from typing import Dict, List, Any
from agents.coder_agent import CoderAgent
from agents.tester_agent import TesterAgent
from agents.reasoning_agent import ReasoningAgent
from agents.orchestration import ThreEvoOrchestration
from execution.executor import CodeExecutor


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
        """
        Initialize the ThreEvo coordinator.

        Args:
            coder_agent: Agent that generates code
            tester_agent: Agent that generates tests
            reasoning_agent: Agent that independently validates
            executor: Code executor
            max_iterations: Maximum number of evolutionary iterations
        """
        self.coder = coder_agent
        self.tester = tester_agent
        self.reasoner = reasoning_agent
        self.executor = executor
        self.max_iterations = max_iterations
        self.orchestration = ThreEvoOrchestration()

    def solve(self, problem: str) -> Dict[str, Any]:
        """
        Main evolutionary loop.

        Iteratively generates code and tests, validates them through three-way
        comparison, generates semantic feedback, and evolves prompts until
        convergence or max iterations reached.

        Args:
            problem: Problem specification string

        Returns:
            Dictionary containing:
                - final_code: The solution code
                - final_tests: The test suite
                - iterations: Number of iterations taken
                - converged: Whether solution converged
                - history: Full iteration history
        """
        iteration = 0
        converged = False
        history = []

        print(f"Starting ThreEvo on problem...")
        print(f"Max iterations: {self.max_iterations}\n")

        while iteration < self.max_iterations and not converged:
            print(f"--- Iteration {iteration + 1}/{self.max_iterations} ---")

            # Step 1: Parallel generation
            print("Generating code...")
            code = self.coder.generate(problem)

            print("Generating tests...")
            tests = self.tester.generate(problem)

            if not tests:
                print("Warning: No tests generated. Stopping.")
                break

            # Step 2: Execute code on tests
            print(f"Executing code on {len(tests)} test cases...")
            execution_results = []
            for test_input, expected_output in tests:
                actual_output = self.executor.execute(code, test_input)
                execution_results.append({
                    'input': test_input,
                    'expected': expected_output,
                    'actual': actual_output
                })

            # Step 3: Reasoning validation
            print("Performing three-way validation...")
            coder_feedback = []
            tester_feedback = []
            all_correct = True
            validation_results = []

            for result in execution_results:
                # Check if execution failed
                if isinstance(result['actual'], dict) and 'error' in result['actual']:
                    all_correct = False
                    error_msg = result['actual']['error']
                    coder_feedback.append(
                        f"Execution error on input {result['input']}: {error_msg}"
                    )
                    validation_results.append({
                        'input': result['input'],
                        'diagnosis': 'execution_error',
                        'feedback': error_msg
                    })
                    continue

                # Reasoning agent independently solves the test case
                reasoned_solution = self.reasoner.solve(problem, result['input'])

                # Three-way validation
                diagnosis = self.orchestration.validate_three_way(
                    expected=result['expected'],
                    actual=result['actual'],
                    reasoned=reasoned_solution
                )

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

                if diagnosis['type'] != 'correct':
                    all_correct = False

                coder_feedback.extend(feedback['coder_feedback'])
                tester_feedback.extend(feedback['tester_feedback'])

                validation_results.append({
                    'input': result['input'],
                    'expected': result['expected'],
                    'actual': result['actual'],
                    'reasoned': reasoned_solution,
                    'diagnosis': diagnosis['type'],
                    'feedback': feedback
                })

            # Step 4: Check convergence
            if all_correct:
                print(f"✓ All tests passed! Converged in {iteration + 1} iterations.")
                converged = True
            else:
                print(f"✗ {len(coder_feedback)} code errors, {len(tester_feedback)} test errors")

            # Save iteration history
            history.append({
                'iteration': iteration + 1,
                'code': code,
                'tests': tests,
                'execution_results': execution_results,
                'validation_results': validation_results,
                'coder_feedback': coder_feedback,
                'tester_feedback': tester_feedback,
                'converged': converged
            })

            # Break if converged
            if converged:
                break

            # Step 5: Evolve prompts using orchestration
            if coder_feedback:
                print("Evolving coder prompt...")
                new_prompt = self.orchestration.evolve_prompt(
                    current_prompt=self.coder.prompt,
                    feedback_history=coder_feedback,
                    agent_type='coder'
                )
                self.coder.prompt = new_prompt

            if tester_feedback:
                print("Evolving tester prompt...")
                new_prompt = self.orchestration.evolve_prompt(
                    current_prompt=self.tester.prompt,
                    feedback_history=tester_feedback,
                    agent_type='tester'
                )
                self.tester.prompt = new_prompt

            iteration += 1
            print()

        if not converged:
            print(f"Did not converge after {self.max_iterations} iterations.")

        return {
            'final_code': code,
            'final_tests': tests,
            'iterations': iteration,
            'converged': converged,
            'history': history
        }

    def save_checkpoint(self, filepath: str) -> None:
        """
        Save current state to checkpoint file.

        Args:
            filepath: Path to save checkpoint
        """
        import json

        checkpoint = {
            'coder_state': self.coder.save_state(),
            'tester_state': self.tester.save_state(),
            'reasoner_state': self.reasoner.save_state(),
        }

        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def load_checkpoint(self, filepath: str) -> None:
        """
        Load state from checkpoint file.

        Args:
            filepath: Path to checkpoint file
        """
        import json

        with open(filepath, 'r') as f:
            checkpoint = json.load(f)

        self.coder.load_state(checkpoint['coder_state'])
        self.tester.load_state(checkpoint['tester_state'])
        self.reasoner.load_state(checkpoint['reasoner_state'])
