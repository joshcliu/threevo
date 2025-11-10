"""
Settings

Configuration management for ThreEvo.
"""

import os
import yaml
from typing import Dict, Any, Optional


class Settings:
    """Configuration settings for ThreEvo experiments"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings.

        Args:
            config_path: Path to YAML config file (optional)
        """
        self.config = self._load_default_config()

        if config_path and os.path.exists(config_path):
            self.load_from_file(config_path)

    def _load_default_config(self) -> Dict[str, Any]:
        """
        Load default configuration.

        Returns:
            Default config dictionary
        """
        return {
            'experiment': {
                'name': 'threevo_experiment',
                'max_iterations': 10,
            },
            'agents': {
                'coder': {
                    'model': 'claude-sonnet-4-20250514',
                    'temperature': 0.7,
                },
                'tester': {
                    'model': 'claude-sonnet-4-20250514',
                    'temperature': 0.7,
                },
                'reasoning': {
                    'model': 'claude-sonnet-4-20250514',
                    'temperature': 0.0,
                }
            },
            'execution': {
                'timeout_seconds': 10,
            },
            'storage': {
                'results_dir': 'results/',
                'save_intermediate': True,
            }
        }

    def load_from_file(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to YAML config file
        """
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f)

        # Merge with defaults
        self._merge_config(self.config, file_config)

    def _merge_config(self, base: Dict, update: Dict) -> None:
        """
        Recursively merge update dict into base dict.

        Args:
            base: Base configuration dictionary
            update: Update configuration dictionary
        """
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., 'agents.coder.model')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key path.

        Args:
            key: Dot-separated key path (e.g., 'agents.coder.model')
            value: Value to set
        """
        keys = key.split('.')
        config = self.config

        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]

        config[keys[-1]] = value

    def save_to_file(self, config_path: str) -> None:
        """
        Save configuration to YAML file.

        Args:
            config_path: Path to save YAML config file
        """
        with open(config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
