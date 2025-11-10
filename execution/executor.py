"""
Code Executor

Executes code with timeout and error handling.
"""

import subprocess
import tempfile
import json
import os
import sys
from typing import Any, Dict


class CodeExecutor:
    """Execute code with timeout and resource limits"""

    def __init__(self, timeout: int = 10):
        """
        Initialize the code executor.

        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout

    def execute(self, code: str, test_input: Any) -> Any:
        """
        Execute code with test input.

        Args:
            code: Python code string to execute
            test_input: Input value to pass to the solution function

        Returns:
            Output from code execution or error dictionary
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            code_path = os.path.join(tmpdir, "solution.py")

            # Write code that includes the test input execution
            full_code = self._build_execution_code(code, test_input)

            with open(code_path, 'w', encoding='utf-8') as f:
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
                    return self._parse_output(result.stdout)
                else:
                    return {'error': result.stderr or 'Unknown execution error'}

            except subprocess.TimeoutExpired:
                return {'error': f'Execution timeout ({self.timeout}s)'}
            except Exception as e:
                return {'error': f'Execution failed: {str(e)}'}

    def _build_execution_code(self, code: str, test_input: Any) -> str:
        """
        Build the full execution code including error handling.

        Args:
            code: User's solution code
            test_input: Input to test

        Returns:
            Full code string with execution wrapper
        """
        return f"""
import json
import sys

# User's solution code
{code}

# Execute with test input
test_input = {repr(test_input)}
try:
    result = solution(test_input)
    print(json.dumps({{'result': result}}))
except Exception as e:
    print(json.dumps({{'error': str(e)}}))
    sys.exit(1)
""".strip()

    def _parse_output(self, stdout: str) -> Any:
        """
        Parse execution output.

        Args:
            stdout: Standard output from execution

        Returns:
            Parsed result or error dictionary
        """
        try:
            # Try to parse as JSON
            output = json.loads(stdout.strip())

            # If there's an error field, return the whole dict
            if 'error' in output:
                return output

            # Otherwise return the result
            return output.get('result')

        except json.JSONDecodeError:
            # If JSON parsing fails, return raw output
            return {'error': f'Failed to parse output: {stdout}'}

    def execute_batch(self, code: str, test_inputs: list) -> list:
        """
        Execute code with multiple test inputs.

        Args:
            code: Python code string to execute
            test_inputs: List of input values

        Returns:
            List of outputs corresponding to each input
        """
        results = []
        for test_input in test_inputs:
            result = self.execute(code, test_input)
            results.append(result)
        return results
