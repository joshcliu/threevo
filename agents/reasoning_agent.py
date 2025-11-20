"""
Reasoning Agent

Independently solves problems through chain-of-thought reasoning.
"""

from typing import Any, List
from .base_agent import BaseAgent


class ReasoningAgent(BaseAgent):
    """Independently solves problems through reasoning"""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        temperature: float = 0.0,  # Deterministic reasoning
        initial_prompt: str = None,
        api_key: str = None
    ):
        """
        Initialize the Reasoning Agent.

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature (default 0.0 for deterministic reasoning)
            initial_prompt: Initial system prompt (uses default if not provided)
            api_key: API key for the LLM service
        """
        if initial_prompt is None:
            initial_prompt = self._get_default_prompt()

        super().__init__(
            model_name=model_name,
            temperature=temperature,
            initial_prompt=initial_prompt,
            api_key=api_key
        )

    def generate(self, problem: str) -> Any:
        """
        Not used for ReasoningAgent. Use solve() instead.

        Args:
            problem: Problem specification string

        Raises:
            NotImplementedError: This method is not used for ReasoningAgent
        """
        raise NotImplementedError("ReasoningAgent uses solve() method instead of generate()")

    def solve(self, problem: str, test_input: Any) -> Any:
        """
        Independently solve the test case through reasoning.

        Uses chain-of-thought prompting to work through the problem.

        Args:
            problem: Problem specification string
            test_input: Input for the test case

        Returns:
            The reasoned solution output
        """
        prompt = self._build_reasoning_prompt(problem, test_input)
        response = self._call_llm(prompt)

        # Extract the final answer from the reasoning trace
        solution = self._extract_solution(response)

        # Save to history
        self.history.append({
            'problem': problem,
            'test_input': test_input,
            'reasoning': response,
            'solution': solution
        })

        # Debug: print reasoning result
        print(f"\n=== ReasoningAgent (Test #{len(self.history)}) ===")
        print(f"Input: {test_input} -> Solution: {solution}")
        print("=" * 50)

        return solution

    def solve_batch(self, problem: str, test_inputs: List[Any]) -> List[Any]:
        """
        Solve multiple test cases in a single call.

        More efficient than calling solve() multiple times since the problem
        and code don't change within an iteration.

        Args:
            problem: Problem specification string
            test_inputs: List of inputs for test cases

        Returns:
            List of reasoned solutions corresponding to each input
        """
        prompt = self._build_batch_reasoning_prompt(problem, test_inputs)
        response = self._call_llm(prompt)

        # Extract solutions for each test case
        solutions = self._extract_batch_solutions(response, len(test_inputs))

        # Save to history
        self.history.append({
            'problem': problem,
            'test_inputs': test_inputs,
            'reasoning': response,
            'solutions': solutions
        })

        # Debug: print reasoning results
        print(f"\n=== ReasoningAgent (Batch of {len(test_inputs)} tests) ===")
        for inp, sol in zip(test_inputs, solutions):
            print(f"  Input: {inp} -> Solution: {sol}")
        print("=" * 50)

        return solutions

    def _build_reasoning_prompt(self, problem: str, test_input: Any) -> str:
        """
        Build the reasoning prompt with chain-of-thought structure.

        Args:
            problem: Problem specification string
            test_input: Input for the test case

        Returns:
            Full reasoning prompt
        """
        return f"""
Problem: {problem}

Input: {test_input}

Think step-by-step to solve this problem:

1. What is the problem asking for?
2. What is the input and what form does it take?
3. What is the correct approach or algorithm to solve this?
4. Walk through the solution step by step with the given input
5. What should the final output be?

Provide your reasoning, then on a new line write "FINAL ANSWER:" followed by just the output value.

Solution:
""".strip()

    def _build_batch_reasoning_prompt(self, problem: str, test_inputs: List[Any]) -> str:
        """
        Build reasoning prompt for multiple test cases at once.

        Args:
            problem: Problem specification string
            test_inputs: List of test inputs

        Returns:
            Full batch reasoning prompt
        """
        inputs_str = "\n".join([f"  Test {i+1}: {inp}" for i, inp in enumerate(test_inputs)])

        return f"""
Problem: {problem}

You need to solve this problem for multiple test inputs. For each test input, determine the correct output.

Test Inputs:
{inputs_str}

For each test input, think through the solution and provide the answer.

Format your response as:
Test 1: [answer]
Test 2: [answer]
...

Be precise and only output the answer value (number, list, string, etc.) for each test.

Solutions:
""".strip()

    def _extract_solution(self, response: str) -> Any:
        """
        Extract the final solution from the reasoning trace.

        Args:
            response: Full reasoning response from LLM

        Returns:
            Extracted solution value
        """
        # Look for "FINAL ANSWER:" marker
        if "FINAL ANSWER:" in response:
            parts = response.split("FINAL ANSWER:")
            if len(parts) > 1:
                answer_str = parts[1].strip()
                # Try to parse as Python literal
                return self._parse_answer(answer_str)

        # If no marker found, try to extract from end of response
        lines = response.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            return self._parse_answer(last_line)

        return None

    def _parse_answer(self, answer_str: str) -> Any:
        """
        Parse answer string into Python value.

        Args:
            answer_str: String representation of answer

        Returns:
            Parsed Python value
        """
        # Remove common prefixes
        answer_str = answer_str.replace("Output:", "").replace("Result:", "").strip()

        # Try to evaluate as Python literal
        try:
            import ast
            return ast.literal_eval(answer_str)
        except (ValueError, SyntaxError):
            # If it fails, return as string
            return answer_str

    def _extract_batch_solutions(self, response: str, num_tests: int) -> List[Any]:
        """
        Extract solutions for multiple test cases from batch response.

        Args:
            response: Full reasoning response from LLM
            num_tests: Number of test cases expected

        Returns:
            List of extracted solutions
        """
        solutions = []
        lines = response.strip().split('\n')

        # Look for "Test N: [answer]" pattern
        for i in range(1, num_tests + 1):
            pattern = f"Test {i}:"
            for line in lines:
                if pattern in line:
                    # Extract answer after the pattern
                    answer_str = line.split(pattern, 1)[1].strip()
                    solution = self._parse_answer(answer_str)
                    solutions.append(solution)
                    break
            else:
                # If pattern not found, append None
                solutions.append(None)

        return solutions

    @staticmethod
    def _get_default_prompt() -> str:
        """
        Get the default system prompt for the Reasoning Agent.

        Returns:
            Default system prompt string
        """
        return """You are an expert reasoning agent. Your task is to independently solve problems through careful step-by-step analysis.

Key guidelines:
1. Think through the problem systematically
2. Break down complex problems into smaller steps
3. Verify your logic at each step
4. Consider edge cases and special conditions
5. Provide clear reasoning for your conclusions
6. Be precise and accurate in your final answer

Your reasoning serves as an independent ground truth to validate both code and test specifications."""
