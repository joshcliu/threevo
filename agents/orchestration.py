"""
Orchestration Logic

Handles three-way validation, feedback generation, and prompt evolution.
"""

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
        Perform three-way validation logic.

        Compares three outputs to diagnose issues:
        - expected: Output from test specification (Tester Agent)
        - actual: Output from code execution (Coder Agent)
        - reasoned: Output from independent reasoning (Reasoning Agent)

        Args:
            expected: Expected output from test specification
            actual: Actual output from code execution
            reasoned: Reasoned solution from independent analysis

        Returns:
            Dictionary with 'type' (error type) and 'target' (which agent to provide feedback to)
        """
        matches = {
            'expected_actual': self._compare(expected, actual),
            'expected_reasoned': self._compare(expected, reasoned),
            'actual_reasoned': self._compare(actual, reasoned)
        }

        # All three match - correct solution
        if all(matches.values()):
            return {'type': 'correct', 'target': None}

        # Expected matches reasoned, but not actual - code error
        elif matches['expected_reasoned'] and not matches['expected_actual']:
            return {'type': 'code_error', 'target': 'coder'}

        # Actual matches reasoned, but not expected - test specification error
        elif matches['actual_reasoned'] and not matches['expected_actual']:
            return {'type': 'test_error', 'target': 'tester'}

        # Expected matches actual, but not reasoned - reasoning conflict (rare)
        elif matches['expected_actual'] and not matches['actual_reasoned']:
            return {'type': 'reasoning_conflict', 'target': 'review'}

        # Nothing matches - could be multiple errors
        else:
            return self._detailed_diagnosis(expected, actual, reasoned, matches)

    def generate_feedback(
        self,
        diagnosis: Dict[str, Any],
        problem: str,
        test_input: Any,
        expected: Any,
        actual: Any,
        reasoned: Any,
        code: str
    ) -> Dict[str, List[str]]:
        """
        Generate rich semantic feedback for both agents.

        Args:
            diagnosis: Diagnosis from validate_three_way
            problem: Problem specification
            test_input: Test input value
            expected: Expected output from test
            actual: Actual output from code
            reasoned: Reasoned solution
            code: Generated code

        Returns:
            Dictionary with 'coder_feedback' and 'tester_feedback' keys
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
        Evolve agent prompt based on accumulated feedback.

        Args:
            current_prompt: Current prompt string
            feedback_history: List of feedback from recent iterations
            agent_type: 'coder' or 'tester'

        Returns:
            Updated prompt string
        """
        if not feedback_history:
            return current_prompt

        # Extract common patterns from feedback
        patterns = self._extract_patterns(feedback_history)

        # Generate additional guidance based on patterns
        additional_guidance = self._generate_guidance(patterns, agent_type)

        # Combine with current prompt
        evolved_prompt = f"""{current_prompt}

Additional guidance based on recent errors:
{additional_guidance}
""".strip()

        return evolved_prompt

    def _compare(self, a: Any, b: Any) -> bool:
        """
        Compare two outputs for equality.

        Args:
            a: First value
            b: Second value

        Returns:
            True if values are equal, False otherwise
        """
        # Handle None cases
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False

        # Direct equality check
        try:
            return a == b
        except Exception:
            # If comparison fails, convert to string and compare
            return str(a) == str(b)

    def _detailed_diagnosis(
        self,
        expected: Any,
        actual: Any,
        reasoned: Any,
        matches: Dict[str, bool]
    ) -> Dict[str, Any]:
        """
        Perform detailed diagnosis when simple matching fails.

        Args:
            expected: Expected output
            actual: Actual output
            reasoned: Reasoned solution
            matches: Dictionary of match results

        Returns:
            Diagnosis dictionary
        """
        # If nothing matches, likely both code and test have errors
        if not any(matches.values()):
            return {'type': 'both_errors', 'target': 'both'}

        # Default to review if unclear
        return {'type': 'unclear', 'target': 'review'}

    def _explain_code_error(
        self,
        problem: str,
        test_input: Any,
        expected: Any,
        actual: Any,
        reasoned: Any,
        code: str
    ) -> str:
        """
        Generate explanation for code errors.

        Args:
            problem: Problem specification
            test_input: Test input
            expected: Expected output
            actual: Actual output
            reasoned: Reasoned solution
            code: Generated code

        Returns:
            Feedback string
        """
        return f"""
CODE ERROR DETECTED:

Test Input: {test_input}
Expected Output: {expected}
Your Code Output: {actual}
Correct Output (from reasoning): {reasoned}

Your code produced incorrect output. The test specification is correct.

Analysis:
- Your code returned {actual} but should return {reasoned}
- Review your logic, especially edge case handling
- Consider: Are you handling all input types correctly? Are there off-by-one errors?

Problem: {problem}

Code that failed:
{code}

Fix the code to handle this case correctly.
""".strip()

    def _explain_test_error(
        self,
        problem: str,
        test_input: Any,
        expected: Any,
        reasoned: Any
    ) -> str:
        """
        Generate explanation for test specification errors.

        Args:
            problem: Problem specification
            test_input: Test input
            expected: Expected output (incorrect)
            reasoned: Reasoned solution (correct)

        Returns:
            Feedback string
        """
        return f"""
TEST SPECIFICATION ERROR DETECTED:

Test Input: {test_input}
Your Expected Output: {expected}
Correct Output (from reasoning): {reasoned}

Your test's expected output is INCORRECT. The code is actually producing the right answer.

Analysis:
- You specified {expected} as the expected output
- But the correct output should be {reasoned}
- Review the problem specification carefully
- Ensure you understand what the function should return

Problem: {problem}

Fix the test specification to expect the correct output.
""".strip()

    def _explain_both_errors(
        self,
        problem: str,
        test_input: Any,
        expected: Any,
        actual: Any,
        reasoned: Any,
        code: str
    ) -> Tuple[str, str]:
        """
        Generate explanations when both code and test have errors.

        Args:
            problem: Problem specification
            test_input: Test input
            expected: Expected output
            actual: Actual output
            reasoned: Reasoned solution
            code: Generated code

        Returns:
            Tuple of (coder_feedback, tester_feedback)
        """
        coder_feedback = f"""
CODE ERROR DETECTED:

Test Input: {test_input}
Your Code Output: {actual}
Correct Output (from reasoning): {reasoned}

Your code produced incorrect output.

Problem: {problem}

Code that failed:
{code}

Fix the code to handle this case correctly.
""".strip()

        tester_feedback = f"""
TEST SPECIFICATION ERROR DETECTED:

Test Input: {test_input}
Your Expected Output: {expected}
Correct Output (from reasoning): {reasoned}

Your test's expected output is INCORRECT.

Problem: {problem}

Fix the test specification to expect the correct output.
""".strip()

        return coder_feedback, tester_feedback

    def _extract_patterns(self, feedback_history: List[str]) -> List[str]:
        """
        Extract common error patterns from feedback history.

        Args:
            feedback_history: List of feedback strings

        Returns:
            List of identified patterns
        """
        patterns = []

        # Simple pattern detection based on keywords
        feedback_text = " ".join(feedback_history).lower()

        if "edge case" in feedback_text:
            patterns.append("edge_case_handling")
        if "empty" in feedback_text:
            patterns.append("empty_input_handling")
        if "negative" in feedback_text:
            patterns.append("negative_number_handling")
        if "boundary" in feedback_text or "off-by-one" in feedback_text:
            patterns.append("boundary_errors")
        if "type" in feedback_text or "None" in feedback_text:
            patterns.append("type_handling")

        return patterns

    def _generate_guidance(self, patterns: List[str], agent_type: str) -> str:
        """
        Generate additional guidance based on error patterns.

        Args:
            patterns: List of identified patterns
            agent_type: 'coder' or 'tester'

        Returns:
            Additional guidance string
        """
        guidance_lines = []

        if agent_type == 'coder':
            if "edge_case_handling" in patterns:
                guidance_lines.append("- Pay special attention to edge cases")
            if "empty_input_handling" in patterns:
                guidance_lines.append("- Always handle empty inputs correctly")
            if "negative_number_handling" in patterns:
                guidance_lines.append("- Consider how negative numbers should be handled")
            if "boundary_errors" in patterns:
                guidance_lines.append("- Check for off-by-one errors and boundary conditions")
            if "type_handling" in patterns:
                guidance_lines.append("- Handle None and different types carefully")

        elif agent_type == 'tester':
            if "edge_case_handling" in patterns:
                guidance_lines.append("- Include more edge case tests")
            if "empty_input_handling" in patterns:
                guidance_lines.append("- Verify expected outputs for empty inputs")
            if "boundary_errors" in patterns:
                guidance_lines.append("- Add tests for boundary values")

        if not guidance_lines:
            guidance_lines.append("- Review previous errors and avoid similar mistakes")

        return "\n".join(guidance_lines)
