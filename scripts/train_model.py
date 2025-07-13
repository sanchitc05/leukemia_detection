"""
Script to train a leukemia detection model.
Supports transfer learning, fine-tuning, and automatic logging.
"""

import logging
import tensorflow as tf
from src.model_config import config
from src.utils import (
    setup_logging,
    set_random_seeds,
    save_model_history,
    create_experiment_folder
)
from src.data_preprocessing import create_tf_datasets
from src.model_architecture import LeukemiaDetectionModel
from pathlib import Path

# Setup logging
setup_logging(log_file="results/logs/training.log")
logger = logging.getLogger(__name__)


def main():
    logger.info("Initializing training pipeline...")

    # Reproducibility
    set_random_seeds(config.data_config["random_seed"])

    # Create experiment folder
    experiment_dir = create_experiment_folder(config.paths_config["results_path"])
    logger.info(f"Created experiment folder at: {experiment_dir}")

    # Load dataset
    train_ds, val_ds, _ = create_tf_datasets(config)

    # Build model
    model_wrapper = LeukemiaDetectionModel(config)
    model = model_wrapper.build_model()

    # Callbacks
    callbacks = model_wrapper.create_callbacks(
        checkpoint_path=str(Path(experiment_dir) / "checkpoint.keras"),
        logs_path=str(Path(experiment_dir) / "tensorboard")
    )

    # Train model
    logger.info("Starting initial training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training_config["epochs"],
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    history_path = Path(experiment_dir) / "history.json"
    save_model_history(history, str(history_path))
    logger.info(f"Saved training history to {history_path}")

    # Enable fine-tuning (optional)
    model_wrapper.enable_fine_tuning()

    logger.info("Starting fine-tuning phase...")
    fine_tune_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training_config["epochs"],
        callbacks=callbacks,
        verbose=1
    )

    # Save fine-tuned model
    model_save_path = Path(config.paths_config["model_save_path"]) / f"model_{Path(experiment_dir).name}"
    model_wrapper.save_model(str(model_save_path), format=config.inference_config["model_format"])
    logger.info(f"Saved fine-tuned model to {model_save_path}")


if __name__ == "__main__":
    main()