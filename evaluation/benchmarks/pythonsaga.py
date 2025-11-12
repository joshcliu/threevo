"""
PythonSaga Benchmark Loader

Loads and manages the PythonSaga benchmark dataset (185 problems, 38 concepts).

Dataset Source: https://github.com/PythonSaga/ACL2024
Paper: "PythonSaga: Redefining the Benchmark to Evaluate Code Generating LLMs"
        (Findings of EMNLP 2024)
"""

import json
import os
from typing import List, Dict, Any, Optional
import urllib.request
import tempfile


class PythonSagaBenchmark:
    """
    Loader for PythonSaga benchmark dataset.

    PythonSaga contains 185 hand-crafted problems with balanced representation
    of 38 programming concepts across diverse difficulty levels (Easy: 20%,
    Medium: 40%, Hard: 40%).

    Dataset Format:
        - task_id: Unique identifier (e.g., "PythonSaga/15")
        - prompt: Problem description with function signature and examples
        - entry_point: Function name to implement
        - canonical_solution: Reference solution
        - test: Test cases for validation
    """

    DATASET_URL = "https://raw.githubusercontent.com/PythonSaga/ACL2024/main/DataSet/basic185.jsonl"
    SAMPLE_INPUT_URL = "https://raw.githubusercontent.com/PythonSaga/ACL2024/main/DataSet/sample_input.jsonl"
    SAMPLE_OUTPUT_URL = "https://raw.githubusercontent.com/PythonSaga/ACL2024/main/DataSet/sample_output.jsonl"

    def __init__(self, data_dir: str = "data/pythonsaga"):
        """
        Initialize the PythonSaga benchmark loader.

        Args:
            data_dir: Directory to store downloaded dataset files
        """
        self.data_dir = data_dir
        self.problems: List[Dict[str, Any]] = []
        self.loaded = False

        # Create data directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def load(self, local_path: Optional[str] = None, force_download: bool = False) -> None:
        """
        Load the PythonSaga dataset.

        Args:
            local_path: Path to local JSONL file (if already downloaded)
            force_download: Force re-download even if local file exists
        """
        if local_path and os.path.exists(local_path) and not force_download:
            # Load from local file
            self._load_from_file(local_path)
        else:
            # Download and load from GitHub
            self._download_and_load()

        self.loaded = True
        print(f"Loaded {len(self.problems)} problems from PythonSaga benchmark")

    def _download_and_load(self) -> None:
        """Download dataset from GitHub and load it."""
        dataset_path = os.path.join(self.data_dir, "basic185.jsonl")

        # Download if not exists
        if not os.path.exists(dataset_path):
            print(f"Downloading PythonSaga dataset from {self.DATASET_URL}...")
            try:
                urllib.request.urlretrieve(self.DATASET_URL, dataset_path)
                print(f"Dataset downloaded to {dataset_path}")
            except Exception as e:
                print(f"Error downloading dataset: {e}")
                print("Please download manually from: https://github.com/PythonSaga/ACL2024")
                raise

        # Load from downloaded file
        self._load_from_file(dataset_path)

    def _load_from_file(self, filepath: str) -> None:
        """
        Load problems from JSONL file.

        Args:
            filepath: Path to JSONL file
        """
        self.problems = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    problem = json.loads(line)
                    self.problems.append(problem)

    def get_problem(self, task_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific problem by task ID.

        Args:
            task_id: Problem task ID (e.g., "PythonSaga/0")

        Returns:
            Problem dictionary or None if not found
        """
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        for problem in self.problems:
            if problem.get('task_id') == task_id:
                return problem

        return None

    def get_problem_by_index(self, index: int) -> Dict[str, Any]:
        """
        Get a problem by index.

        Args:
            index: Problem index (0-184)

        Returns:
            Problem dictionary
        """
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        if index < 0 or index >= len(self.problems):
            raise IndexError(f"Index {index} out of range (0-{len(self.problems)-1})")

        return self.problems[index]

    def get_all_problems(self) -> List[Dict[str, Any]]:
        """
        Get all problems in the dataset.

        Returns:
            List of all problem dictionaries
        """
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        return self.problems

    def get_problems_by_difficulty(self, difficulty: str) -> List[Dict[str, Any]]:
        """
        Get problems filtered by difficulty level.

        Args:
            difficulty: One of "Easy", "Medium", "Hard"

        Returns:
            List of problems matching the difficulty
        """
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        # Note: PythonSaga doesn't explicitly label difficulty in the dataset
        # This would need to be added manually or inferred
        # For now, return all problems with a warning
        print(f"Warning: Difficulty filtering not yet implemented for PythonSaga")
        return self.problems

    def format_problem_for_threevo(self, problem: Dict[str, Any]) -> str:
        """
        Format a PythonSaga problem for ThreEvo.

        Converts the PythonSaga problem format into a clear problem specification
        string suitable for the ThreEvo framework.

        Args:
            problem: Problem dictionary from PythonSaga

        Returns:
            Formatted problem string
        """
        task_id = problem.get('task_id', 'Unknown')
        prompt = problem.get('prompt', '')
        entry_point = problem.get('entry_point', 'solution')

        # Format the problem description
        formatted_problem = f"""Task ID: {task_id}

{prompt}

Note: The function should be named '{entry_point}'.
"""

        return formatted_problem.strip()

    def get_test_cases(self, problem: Dict[str, Any]) -> List[tuple]:
        """
        Extract test cases from a problem.

        Note: PythonSaga test cases are in Python code format, not simple
        input/output pairs. This method attempts to parse them if possible.

        Args:
            problem: Problem dictionary

        Returns:
            List of (input, expected_output) tuples (may be empty if parsing fails)
        """
        # PythonSaga tests are in code format, not structured data
        # For now, return empty list - let the Tester Agent generate tests
        # In a full implementation, you could parse the test code

        test_code = problem.get('test', '')

        # TODO: Parse test code to extract test cases
        # This would require parsing Python assert statements

        return []

    def save_results(self, results: List[Dict[str, Any]], output_path: str) -> None:
        """
        Save evaluation results in PythonSaga format.

        Args:
            results: List of result dictionaries
            output_path: Path to save results (JSONL format)
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                # Format: {"task_id": "...", "completion": "...", "passed": bool}
                output = {
                    'task_id': result.get('task_id', ''),
                    'completion': result.get('final_code', ''),
                    'passed': result.get('converged', False)
                }
                f.write(json.dumps(output) + '\n')

        print(f"Results saved to {output_path}")

    def get_statistics(self) -> Dict[str, Any]:
        """
        Get dataset statistics.

        Returns:
            Dictionary containing dataset statistics
        """
        if not self.loaded:
            raise RuntimeError("Dataset not loaded. Call load() first.")

        return {
            'total_problems': len(self.problems),
            'concepts': 38,  # As stated in paper
            'difficulty_distribution': {
                'Easy': '20%',
                'Medium': '40%',
                'Hard': '40%'
            },
            'avg_test_cases': 3.7  # As stated in paper
        }

    def __len__(self) -> int:
        """Return number of problems in dataset."""
        return len(self.problems)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get problem by index using bracket notation."""
        return self.get_problem_by_index(index)

    def __iter__(self):
        """Iterate over problems."""
        return iter(self.problems)
