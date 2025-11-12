"""
Example: Using ThreEvo with PythonSaga Benchmark

Demonstrates how to run ThreEvo on problems from the PythonSaga benchmark.
"""

import os
from dotenv import load_dotenv
from agents import CoderAgent, TesterAgent, ReasoningAgent
from execution import CodeExecutor
from coordinator import ThreEvoCoordinator
from config import Settings
from evaluation.benchmarks import PythonSagaBenchmark

# Load environment variables from .env file
load_dotenv()


def run_single_problem(problem_index: int = 0):
    """
    Run ThreEvo on a single problem from PythonSaga.

    Args:
        problem_index: Index of the problem to solve (0-184)
    """
    # Load PythonSaga benchmark
    print("Loading PythonSaga benchmark...")
    benchmark = PythonSagaBenchmark()

    try:
        benchmark.load()
    except Exception as e:
        print(f"Error loading benchmark: {e}")
        print("\nTo use PythonSaga benchmark:")
        print("1. Download from: https://github.com/PythonSaga/ACL2024")
        print("2. Place prompt.jsonl in data/pythonsaga/")
        print("3. Or the loader will attempt to download automatically")
        return

    # Get a specific problem
    problem = benchmark.get_problem_by_index(problem_index)
    formatted_problem = benchmark.format_problem_for_threevo(problem)

    # Get API key from environment
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Load settings
    settings = Settings()

    # Initialize agents
    print("\nInitializing agents...")
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

    # Display problem
    print("\n" + "="*80)
    print("PYTHONSAGA PROBLEM:")
    print("="*80)
    print(formatted_problem)
    print("="*80 + "\n")

    # Solve the problem
    result = coordinator.solve(formatted_problem)

    # Display results
    print("\n" + "="*80)
    print("RESULTS:")
    print("="*80)
    print(f"Task ID: {problem['task_id']}")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iterations']}")

    print("\n" + "="*80)
    print("FINAL CODE:")
    print("="*80)
    print(result['final_code'])
    print("="*80)

    print("\n" + "="*80)
    print("GENERATED TEST CASES:")
    print("="*80)
    for i, (test_input, expected) in enumerate(result['final_tests'], 1):
        print(f"Test {i}:")
        print(f"  Input:    {test_input}")
        print(f"  Expected: {expected}")
    print("="*80)

    # Show reasoning agent's validations from last iteration
    if result['history']:
        last_iteration = result['history'][-1]
        print("\n" + "="*80)
        print("REASONING AGENT VALIDATIONS (Last Iteration):")
        print("="*80)

        if 'validation_results' in last_iteration:
            for i, validation in enumerate(last_iteration['validation_results'], 1):
                print(f"\nTest {i}:")
                print(f"  Input:         {validation['input']}")
                print(f"  Expected:      {validation['expected']}")
                print(f"  Actual:        {validation['actual']}")
                print(f"  Reasoned:      {validation['reasoned']}")
                print(f"  Diagnosis:     {validation['diagnosis']}")

        print("="*80)

    print(f"\n✓ Solution {'converged' if result['converged'] else 'did not converge'} after {result['iterations']} iteration(s)")
    print("="*80)

    # Save result
    result_data = {
        'task_id': problem['task_id'],
        'final_code': result['final_code'],
        'converged': result['converged'],
        'iterations': result['iterations']
    }

    benchmark.save_results([result_data], 'results/pythonsaga_result.jsonl')


def run_multiple_problems(start_index: int = 0, count: int = 5):
    """
    Run ThreEvo on multiple problems from PythonSaga.

    Args:
        start_index: Starting problem index
        count: Number of problems to solve
    """
    # Load PythonSaga benchmark
    print("Loading PythonSaga benchmark...")
    benchmark = PythonSagaBenchmark()

    try:
        benchmark.load()
    except Exception as e:
        print(f"Error loading benchmark: {e}")
        return

    # Get API key
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable not set")
        return

    # Initialize settings
    settings = Settings()

    # Results storage
    all_results = []

    # Process each problem
    for i in range(start_index, min(start_index + count, len(benchmark))):
        print(f"\n{'='*80}")
        print(f"PROBLEM {i+1}/{min(start_index + count, len(benchmark))}")
        print(f"{'='*80}\n")

        problem = benchmark[i]
        formatted_problem = benchmark.format_problem_for_threevo(problem)

        # Initialize fresh agents for each problem
        coder = CoderAgent(api_key=api_key)
        tester = TesterAgent(api_key=api_key)
        reasoner = ReasoningAgent(api_key=api_key)
        executor = CodeExecutor(timeout=settings.get('execution.timeout_seconds'))

        coordinator = ThreEvoCoordinator(
            coder_agent=coder,
            tester_agent=tester,
            reasoning_agent=reasoner,
            executor=executor,
            max_iterations=settings.get('experiment.max_iterations')
        )

        # Solve
        result = coordinator.solve(formatted_problem)

        # Save result
        result_data = {
            'task_id': problem['task_id'],
            'final_code': result['final_code'],
            'converged': result['converged'],
            'iterations': result['iterations']
        }
        all_results.append(result_data)

        print(f"\n✓ Problem {i}: {'CONVERGED' if result['converged'] else 'NOT CONVERGED'} ({result['iterations']} iterations)")

    # Save all results
    benchmark.save_results(all_results, 'results/pythonsaga_batch_results.jsonl')

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY:")
    print(f"{'='*80}")
    print(f"Total problems: {len(all_results)}")
    print(f"Converged: {sum(1 for r in all_results if r['converged'])}")
    print(f"Average iterations: {sum(r['iterations'] for r in all_results) / len(all_results):.2f}")
    print(f"{'='*80}")


def show_benchmark_info():
    """Display information about the PythonSaga benchmark."""
    benchmark = PythonSagaBenchmark()

    try:
        benchmark.load()
    except Exception as e:
        print(f"Error loading benchmark: {e}")
        return

    print("\n" + "="*80)
    print("PYTHONSAGA BENCHMARK INFORMATION")
    print("="*80)

    stats = benchmark.get_statistics()
    print(f"\nTotal Problems: {stats['total_problems']}")
    print(f"Programming Concepts: {stats['concepts']}")
    print(f"Difficulty Distribution:")
    for difficulty, pct in stats['difficulty_distribution'].items():
        print(f"  - {difficulty}: {pct}")
    print(f"Average Test Cases per Problem: {stats['avg_test_cases']}")

    print("\n" + "="*80)
    print("SAMPLE PROBLEMS:")
    print("="*80)

    # Show first 3 problems
    for i in range(min(3, len(benchmark))):
        problem = benchmark[i]
        print(f"\n[{i}] {problem['task_id']}")
        print(f"Entry Point: {problem['entry_point']}")
        # Show first 200 chars of prompt
        prompt = problem.get('prompt', '')
        print(f"Prompt: {prompt[:200]}...")

    print("\n" + "="*80)


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "info":
            show_benchmark_info()
        elif command == "single":
            problem_index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            run_single_problem(problem_index)
        elif command == "batch":
            start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
            count = int(sys.argv[3]) if len(sys.argv) > 3 else 5
            run_multiple_problems(start, count)
        else:
            print("Unknown command. Use: info, single [index], or batch [start] [count]")
    else:
        # Default: run single problem
        print("Running single problem from PythonSaga...")
        print("Usage: python example_pythonsaga.py [info|single|batch] [args]")
        print()
        run_single_problem(0)
