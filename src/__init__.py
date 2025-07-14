"""
Leukemia Detection Package

This package contains modules for data preprocessing, model training,
evaluation, configuration management, utilities, and inference.

Modules:
- data_preprocessing: Functions and classes for loading, preparing, and preprocessing the dataset.
- model_architecture: Contains custom CNN and transfer learning models for classification.
- training: Training script to train models using the specified configuration.
- evaluation: Evaluation utilities for computing metrics and visualizing results.
- data_augmentation: Augmentation pipeline constructed from configuration.
- utils: Miscellaneous utility functions including logging, visualization, and saving/loading tools.
- model_config: Configuration handler that parses config.yaml and provides access to parameters.

Usage:
Import specific components as needed. For example:

    from src.model_architecture import LeukemiaDetectionModel
    from src.utils import setup_logging, plot_training_history
    from src.training import run_training_pipeline
"""

# Optional: Expose commonly used classes/functions at package level
from .model_config import ModelConfig
from .model_architecture import LeukemiaDetectionModel, CustomCNN
from .training import run_training_pipeline
from .evaluation import evaluate_model
from .utils import setup_logging, set_random_seeds, get_device_info
<<<<<<< HEAD
from .data_augmentation import DataAugmentor
from .data_preprocessing import preprocess_data, load_dataset
=======
from .data_augmentation import DataAugmentor
>>>>>>> e211ebe (commit)
