"""
State Manager

Tracks iteration state and history for experiments.
"""

import json
import os
from typing import Dict, Any, List
from datetime import datetime


class StateManager:
    """Manages experiment state and history"""

    def __init__(self, results_dir: str = "results"):
        """
        Initialize the state manager.

        Args:
            results_dir: Directory to store results
        """
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)

    def save_result(
        self,
        problem_id: str,
        result: Dict[str, Any],
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Save experiment result to file.

        Args:
            problem_id: Unique identifier for the problem
            result: Result dictionary from coordinator
            metadata: Additional metadata to save

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{problem_id}_{timestamp}.json"
        filepath = os.path.join(self.results_dir, filename)

        output = {
            'problem_id': problem_id,
            'timestamp': timestamp,
            'result': result,
            'metadata': metadata or {}
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

        return filepath

    def load_result(self, filepath: str) -> Dict[str, Any]:
        """
        Load result from file.

        Args:
            filepath: Path to result file

        Returns:
            Result dictionary
        """
        with open(filepath, 'r') as f:
            return json.load(f)

    def get_all_results(self, problem_id: str = None) -> List[str]:
        """
        Get paths to all result files.

        Args:
            problem_id: Optional filter by problem ID

        Returns:
            List of file paths
        """
        files = []
        for filename in os.listdir(self.results_dir):
            if filename.endswith('.json'):
                if problem_id is None or filename.startswith(problem_id):
                    files.append(os.path.join(self.results_dir, filename))
        return sorted(files)

    def save_summary(
        self,
        results: List[Dict[str, Any]],
        filename: str = "summary.json"
    ) -> str:
        """
        Save summary of multiple results.

        Args:
            results: List of result dictionaries
            filename: Name of summary file

        Returns:
            Path to saved summary file
        """
        filepath = os.path.join(self.results_dir, filename)

        summary = {
            'total_problems': len(results),
            'converged': sum(1 for r in results if r.get('converged', False)),
            'avg_iterations': sum(r.get('iterations', 0) for r in results) / len(results) if results else 0,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        return filepath
