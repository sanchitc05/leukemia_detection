"""
Script to evaluate a trained leukemia detection model on the test dataset.
Supports multiclass classification evaluation with metrics and visualization.
"""

import logging
import numpy as np
import tensorflow as tf
from pathlib import Path
from src.utils import (
    setup_logging,
    load_metrics,
    save_metrics,
    plot_confusion_matrix,
    calculate_metrics,
    create_experiment_folder,
    load_model_history
)
from src.data_preprocessing import create_tf_datasets
from src.model_architecture import LeukemiaDetectionModel
from src.model_config import config
from src.evaluation import evaluate_model_performance

# Setup logging
setup_logging(log_file="results/logs/evaluation.log")
logger = logging.getLogger(__name__)


def main():
    logger.info("Starting model evaluation...")

    # Load test dataset
    _, _, test_ds = create_tf_datasets(config)
    class_names = ["Healthy", "Leukemia"] if config.model_config["num_classes"] == 2 else list(range(config.model_config["num_classes"]))
    logger.info(f"Loaded test dataset with {len(test_ds)} batches.")

    # Load trained model
    model_path = Path(config.paths_config["model_save_path"])
    latest_model = sorted(model_path.glob("*"), key=lambda x: x.stat().st_mtime)[-1]
    logger.info(f"Loading model from: {latest_model}")
    model = tf.keras.models.load_model(latest_model)

    # Evaluate the model
    y_true = []
    y_pred = []
    y_prob = []

    for batch in test_ds:
        images, labels = batch
        preds = model.predict(images)
        if config.model_config["num_classes"] == 2:
            predicted_labels = (preds > config.evaluation_config["threshold"]).astype("int32").flatten()
        else:
            predicted_labels = np.argmax(preds, axis=1)
        
        y_true.extend(labels.numpy())
        y_pred.extend(predicted_labels)
        y_prob.extend(preds if config.model_config["num_classes"] > 2 else preds.flatten())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)

    # Calculate metrics
    metrics = calculate_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_pred_proba=y_prob if config.model_config["num_classes"] == 2 else None
    )

    # Create experiment folder and save
    exp_path = create_experiment_folder(config.paths_config["results_path"])
    metrics_path = Path(exp_path) / "metrics.json"
    cm_path = Path(exp_path) / "confusion_matrix.png"
    history_path = Path(exp_path) / "history.json"

    save_metrics(metrics, metrics_path)
    logger.info(f"Saved metrics to {metrics_path}")

    plot_confusion_matrix(y_true, y_pred, class_names=class_names, save_path=cm_path)
    logger.info(f"Saved confusion matrix to {cm_path}")

    # Plot training history if available
    model_history_files = list(Path(config.paths_config["logs_path"]).glob("*.json"))
    if model_history_files:
        from src.utils import plot_training_history
        history_dict = load_model_history(model_history_files[-1])
        plot_training_history(history_dict, save_path=Path(exp_path) / "training_history.png")

    logger.info("Model evaluation complete.")


if __name__ == "__main__":
    main()