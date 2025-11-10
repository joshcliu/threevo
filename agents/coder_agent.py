"""
Coder Agent

Generates code solutions from problem specifications.
"""

from typing import Any, Dict
from .base_agent import BaseAgent


class CoderAgent(BaseAgent):
    """Generates code solutions from problem specifications"""

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-20250514",
        temperature: float = 0.7,
        initial_prompt: str = None,
        api_key: str = None
    ):
        """
        Initialize the Coder Agent.

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

    def generate(self, problem: str) -> str:
        """
        Generate code using current evolved prompt.

        Args:
            problem: Problem specification string

        Returns:
            Generated Python code as string
        """
        prompt = self._build_generation_prompt(problem)
        code = self._call_llm(prompt)

        # Extract code from markdown if present
        code = self._extract_code(code)

        # Save to history
        self.history.append({
            'problem': problem,
            'code': code,
            'prompt': self.prompt
        })

        return code

    def _build_generation_prompt(self, problem: str) -> str:
        """
        Combine evolved prompt with current problem.

        Args:
            problem: Problem specification string

        Returns:
            Full prompt for code generation
        """
        return f"""
{problem}

Generate a Python function that solves this problem. Return only the code, properly formatted.
The function should be named 'solution' and handle the input as specified.

Code:
""".strip()

    def _extract_code(self, response: str) -> str:
        """
        Extract code from LLM response, removing markdown formatting.

        Args:
            response: Raw LLM response

        Returns:
            Cleaned code string
        """
        # Remove markdown code blocks if present
        if "```python" in response:
            parts = response.split("```python")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()
        elif "```" in response:
            parts = response.split("```")
            if len(parts) > 1:
                code = parts[1].split("```")[0]
                return code.strip()

        return response.strip()

    @staticmethod
    def _get_default_prompt() -> str:
        """
        Get the default system prompt for the Coder Agent.

        Returns:
            Default system prompt string
        """
        return """You are an expert Python programmer. Your task is to write clean, correct, and efficient code.

Key guidelines:
1. Write clear, readable code with proper variable names
2. Handle edge cases carefully (empty inputs, negative numbers, etc.)
3. Follow Python best practices
4. Ensure your code is bug-free and handles all specified requirements
5. Think step-by-step about the problem before coding
6. Test your logic mentally before finalizing the solution

Focus on correctness first, then efficiency."""
