"""
Tester Agent

Generates test suites from problem specifications.
"""

import json
from typing import Any, List, Tuple, Dict
from .base_agent import BaseAgent


class TesterAgent(BaseAgent):
    """Generates test suites from problem specifications"""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        initial_prompt: str = None,
        api_key: str = None
    ):
        """
        Initialize the Tester Agent.

        Args:
            model_name: Name of the LLM model to use
            temperature: Sampling temperature for generation
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

    def generate(self, problem: str) -> List[Tuple[Any, Any]]:
        """
        Generate test cases including edge cases.

        Args:
            problem: Problem specification string

        Returns:
            List of (input, expected_output) tuples
        """
        prompt = self._build_generation_prompt(problem)
        test_suite_str = self._call_llm(prompt)

        # Parse test suite
        parsed_tests = self._parse_test_suite(test_suite_str)

        # Save to history
        self.history.append({
            'problem': problem,
            'tests': parsed_tests,
            'prompt': self.prompt
        })

        return parsed_tests

    def _build_generation_prompt(self, problem: str) -> str:
        """
        Combine evolved prompt with current problem.

        Args:
            problem: Problem specification string

        Returns:
            Full prompt for test generation
        """
        return f"""
{problem}

Generate a comprehensive test suite for this problem. Include:
1. Basic test cases
2. Edge cases (empty inputs, boundaries, special values)
3. Corner cases (large inputs, negative numbers, etc.)

IMPORTANT: Return the tests in VALID JSON format as a list of objects.
- Each object must have "input" and "expected" fields
- Use only valid JSON syntax (no Python expressions like [100] * 10)
- Arrays must be fully written out: use [100, 100, 100] not [100] * 3
- All values must be valid JSON types (numbers, strings, arrays, objects, booleans, null)

Example format:
[
    {{"input": [1, 2, 3], "expected": 6}},
    {{"input": [], "expected": 0}},
    {{"input": [-1, -2], "expected": -3}}
]

Tests:
""".strip()

    def _parse_test_suite(self, test_suite_str: str) -> List[Tuple[Any, Any]]:
        """
        Parse test suite string into list of (input, expected) tuples.

        Args:
            test_suite_str: Raw test suite string from LLM

        Returns:
            List of (input, expected_output) tuples
        """
        try:
            # Extract JSON from markdown if present
            if "```json" in test_suite_str:
                parts = test_suite_str.split("```json")
                if len(parts) > 1:
                    json_str = parts[1].split("```")[0].strip()
                    test_suite_str = json_str
            elif "```" in test_suite_str:
                parts = test_suite_str.split("```")
                if len(parts) > 1:
                    json_str = parts[1].split("```")[0].strip()
                    test_suite_str = json_str

            # Parse JSON
            tests_data = json.loads(test_suite_str)

            # Convert to list of tuples
            parsed_tests = []
            for test in tests_data:
                test_input = test.get('input')
                expected = test.get('expected')
                parsed_tests.append((test_input, expected))

            return parsed_tests

        except json.JSONDecodeError as e:
            print(f"Error parsing test suite: {e}")
            print(f"Raw output (first 500 chars): {test_suite_str[:500]}...")
            print("Note: Make sure the output is valid JSON, not Python code!")
            # Return empty list if parsing fails
            return []
        except Exception as e:
            print(f"Unexpected error parsing tests: {e}")
            print(f"Raw output (first 500 chars): {test_suite_str[:500]}...")
            return []

    @staticmethod
    def _get_default_prompt() -> str:
        """
        Get the default system prompt for the Tester Agent.

        Returns:
            Default system prompt string
        """
        return """You are an expert test designer. Your task is to create comprehensive test suites that thoroughly validate code correctness.

Key guidelines:
1. Cover basic functionality with simple test cases
2. Include edge cases (empty inputs, boundaries, None, zero, etc.)
3. Include corner cases (large inputs, negative values, special characters)
4. Think about what could break the code
5. Ensure expected outputs are CORRECT according to the problem specification
6. Generate diverse test cases that cover different scenarios
7. ALWAYS output VALID JSON only (no Python expressions like [100] * 10)
8. Write out arrays fully: use [100, 100, 100] instead of [100] * 3

Focus on correctness of expected outputs and comprehensive coverage."""
