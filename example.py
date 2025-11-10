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

    # Define a simple problem
    problem = """
Write a function called 'solution' that takes a list of integers and returns their sum.

Example:
- Input: [1, 2, 3]
- Output: 6

- Input: []
- Output: 0

- Input: [-1, 1, -2, 2]
- Output: 0
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
    print("\nFinal Code:")
    print("-" * 60)
    print(result['final_code'])
    print("-" * 60)
    print(f"\nNumber of Tests: {len(result['final_tests'])}")
    print("="*60)


if __name__ == "__main__":
    main()
