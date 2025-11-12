"""
Example Usage of ThreEvo

Demonstrates how to use the ThreEvo framework to solve a simple problem.
"""

import os
from dotenv import load_dotenv
from agents import CoderAgent, TesterAgent, ReasoningAgent
from execution import CodeExecutor
from coordinator import ThreEvoCoordinator
from config import Settings

# Load environment variables from .env file
load_dotenv()


def main():
    """Run a simple example problem through ThreEvo"""

    # Define a real problem from HumanEval benchmark
    # HumanEval/0: has_close_elements
    problem = """
Write a function called 'solution' that checks if in a given list of numbers,
any two numbers are closer to each other than a given threshold.

The function should take two parameters:
1. numbers: A list of float numbers
2. threshold: A float representing the minimum distance threshold

Return True if any two numbers in the list are closer than the threshold, False otherwise.

Examples:
- Input: numbers=[1.0, 2.0, 3.0], threshold=0.5
  Output: False
  (All numbers are at least 1.0 apart, which is greater than 0.5)

- Input: numbers=[1.0, 2.8, 3.0, 4.0, 5.0, 2.0], threshold=0.3
  Output: True
  (2.8 and 3.0 are 0.2 apart, which is less than 0.3)

- Input: numbers=[1.0, 2.0, 3.9, 4.0, 5.0, 2.2], threshold=0.3
  Output: True
  (2.0 and 2.2 are 0.2 apart, also 3.9 and 4.0 are 0.1 apart)

- Input: numbers=[], threshold=0.5
  Output: False
  (Empty list has no pairs)

- Input: numbers=[1.0], threshold=0.5
  Output: False
  (Single element has no pairs)

Note: The function should be named 'solution' and take a tuple/list containing
both parameters: solution((numbers, threshold))
"""

    # Get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Load settings
    settings = Settings()

    # Initialize agents
    print("Initializing agents...")
    coder = CoderAgent(
        model_name=settings.get('agents.coder.model'),
        temperature=settings.get('agents.coder.temperature'),
        api_key=api_key
    )

    tester = TesterAgent(
        model_name=settings.get('agents.tester.model'),
        temperature=settings.get('agents.tester.temperature'),
        api_key=api_key
    )

    reasoner = ReasoningAgent(
        model_name=settings.get('agents.reasoning.model'),
        temperature=settings.get('agents.reasoning.temperature'),
        api_key=api_key
    )

    # Initialize executor
    executor = CodeExecutor(timeout=settings.get('execution.timeout_seconds'))

    # Initialize coordinator
    coordinator = ThreEvoCoordinator(
        coder_agent=coder,
        tester_agent=tester,
        reasoning_agent=reasoner,
        executor=executor,
        max_iterations=settings.get('experiment.max_iterations')
    )

    # Solve the problem
    print("\n" + "="*60)
    print("PROBLEM:")
    print("="*60)
    print(problem)
    print("="*60 + "\n")

    result = coordinator.solve(problem)

    # Display results
    print("\n" + "="*60)
    print("RESULTS:")
    print("="*60)
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")

    print("\n" + "="*60)
    print("FINAL CODE:")
    print("="*60)
    print(result['final_code'])
    print("="*60)

    print("\n" + "="*60)
    print("GENERATED TEST CASES:")
    print("="*60)
    for i, (test_input, expected) in enumerate(result['final_tests'], 1):
        print(f"Test {i}:")
        print(f"  Input:    {test_input}")
        print(f"  Expected: {expected}")
    print("="*60)

    # Show reasoning agent's validations from last iteration
    if result['history']:
        last_iteration = result['history'][-1]
        print("\n" + "="*60)
        print("REASONING AGENT VALIDATIONS (Last Iteration):")
        print("="*60)

        if 'validation_results' in last_iteration:
            for i, validation in enumerate(last_iteration['validation_results'], 1):
                print(f"\nTest {i}:")
                print(f"  Input:         {validation['input']}")
                print(f"  Expected:      {validation['expected']}")
                print(f"  Actual:        {validation['actual']}")
                print(f"  Reasoned:      {validation['reasoned']}")
                print(f"  Diagnosis:     {validation['diagnosis']}")

                # Show feedback if any
                if validation.get('feedback'):
                    coder_fb = validation['feedback'].get('coder_feedback', [])
                    tester_fb = validation['feedback'].get('tester_feedback', [])

                    if coder_fb:
                        print(f"  Coder Feedback: {coder_fb[0][:100]}...")
                    if tester_fb:
                        print(f"  Tester Feedback: {tester_fb[0][:100]}...")

        print("="*60)

    print(f"\n✓ Solution {'converged' if result['converged'] else 'did not converge'} after {result['iterations']} iteration(s)")
    print("="*60)


if __name__ == "__main__":
    main()
