import yaml
from pathlib import Path
from typing import Any, Dict

class ConfigLoader:
    @staticmethod
    def load(config_path: str) -> Dict[str, Any]:
        """
        Loads and returns configuration from a YAML file.

        Args:
            config_path (str): Path to the YAML configuration file.

        Returns:
            Dict[str, Any]: A dictionary of configuration values.

        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file {config_path} does not exist.")
        with config_file.open("r") as f:
            config: Dict[str, Any] = yaml.safe_load(f)
        return config
