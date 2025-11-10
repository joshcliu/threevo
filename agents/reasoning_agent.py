"""
Reasoning Agent

Independently solves problems through chain-of-thought reasoning.
"""

from typing import Any
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

        return solution

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
