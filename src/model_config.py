"""
Configuration loader for the Leukemia Detection project.
Parses the YAML file and exposes configuration sections.
"""

import yaml
from pathlib import Path
from typing import Any, Dict
import os
import logging

# Ensure the YAML file is in the same directory as this script
CONFIG_FILE = Path(__file__).parent / "config.yaml"
from pathlib import Path
import yaml

class ModelConfig:
    """Configuration class for model parameters."""

    def __init__(self, config_path=None):
        if config_path is None:
            # Try to find config.yaml in root or src/
            possible_paths = [
                Path(__file__).parent.parent / "config.yaml",  # root/config.yaml
                Path(__file__).parent / "config.yaml",         # src/config.yaml
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = path
                    break
            else:
                raise FileNotFoundError("Configuration file not found in expected locations.")
        else:
            config_path = Path(config_path)
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
class ModelConfig:
    """Configuration class to load and manage config.yaml settings."""

    def __init__(self, config_path: str = None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
@property
def data(self) -> Dict[str, Any]:
        return self.config.get("data", {})

@property
def model(self) -> Dict[str, Any]:
        return self.config.get("model", {})

@property
def training(self) -> Dict[str, Any]:
        return self.config.get("training", {})

@property
def augmentation(self) -> Dict[str, Any]:
        return self.config.get("augmentation", {})

@property
def evaluation(self) -> Dict[str, Any]:
        return self.config.get("evaluation", {})

@property
def paths(self) -> Dict[str, Any]:
        return self.config.get("paths", {})

@property
def inference(self) -> Dict[str, Any]:
        return self.config.get("inference", {})

@property
def logging(self) -> Dict[str, Any]:
        return self.config.get("logging", {})

def get_model_params(self) -> Dict[str, Any]:
        model_cfg = self.model
        return {
            "architecture": model_cfg.get("architecture"),
            "input_shape": tuple(model_cfg.get("input_shape", [224, 224, 3])),
            "num_classes": model_cfg.get("num_classes", 2),
            "dropout_rate": model_cfg.get("dropout_rate", 0.3),
            "pretrained_weights": model_cfg.get("pretrained_weights", None),
            "fine_tune_layers": model_cfg.get("fine_tune_layers", 0),
        }

def get_training_params(self) -> Dict[str, Any]:
        train_cfg = self.training
        return {
            "epochs": train_cfg.get("epochs", 50),
            "initial_learning_rate": train_cfg.get("initial_learning_rate", 0.001),
            "fine_tune_learning_rate": train_cfg.get("fine_tune_learning_rate", 0.0001),
            "early_stopping_patience": train_cfg.get("early_stopping_patience", 10),
            "reduce_lr_patience": train_cfg.get("reduce_lr_patience", 5),
            "reduce_lr_factor": train_cfg.get("reduce_lr_factor", 0.5),
            "optimizer": train_cfg.get("optimizer", "adam"),
            "loss_function": train_cfg.get("loss_function", "binary_crossentropy"),
            "metrics": train_cfg.get("metrics", ["accuracy"]),
        }

def get_data_params(self) -> Dict[str, Any]:
        data_cfg = self.data
        return {
            "dataset_path": data_cfg.get("dataset_path"),
            "processed_path": data_cfg.get("processed_path"),
            "image_size": tuple(data_cfg.get("image_size", [224, 224])),
            "batch_size": data_cfg.get("batch_size", 32),
            "validation_split": data_cfg.get("validation_split", 0.2),
            "test_split": data_cfg.get("test_split", 0.1),
            "random_seed": data_cfg.get("random_seed", 42),
        }

def create_directories(self):
        """Create all necessary directories from config."""
        from pathlib import Path

        paths = self.paths
        directories = [
            paths.get("model_save_path"),
            paths.get("checkpoint_path"),
            paths.get("export_path"),
            paths.get("results_path"),
            paths.get("logs_path"),
            paths.get("plots_path"),
            f"{self.data.get('processed_path')}train/healthy/",
            f"{self.data.get('processed_path')}train/leukemia/",
            f"{self.data.get('processed_path')}validation/healthy/",
            f"{self.data.get('processed_path')}validation/leukemia/",
            f"{self.data.get('processed_path')}test/healthy/",
            f"{self.data.get('processed_path')}test/leukemia/",
        ]

        for path in directories:
            if path:
                Path(path).mkdir(parents=True, exist_ok=True)

def update(self, section: str, key: str, value: Any):
        """Update a specific configuration value."""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
        else:
            raise KeyError(f"Section '{section}' or key '{key}' not found in config.")

def save(self, save_path: str = None):
        """Save the updated config back to file."""
        if save_path is None:
            save_path = Path(__file__).parent / "config.yaml"
        else:
            save_path = Path(save_path)

        with open(save_path, "w") as f:
            yaml.safe_dump(self.config, f, default_flow_style=False)


# Optional global instance for convenience
config = ModelConfig()