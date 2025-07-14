<<<<<<< HEAD
import os
import tensorflow as tf
from config.model_config import ModelConfig
from models.model_architecture import LeukemiaDetectionModel, CustomCNN
from utils.utils import (
    setup_logging, set_random_seeds, get_device_info,
    validate_data_structure, create_experiment_folder, 
    save_model_history, plot_training_history,
    calculate_metrics, save_metrics, plot_confusion_matrix
)
import logging
from pathlib import Path
import json


def main():
    # Load config
    config = ModelConfig()
    config.create_directories()

    # Setup logging and seed
    setup_logging(config.config['logging']['level'], config.config['logging']['file_path'])
    logger = logging.getLogger(__name__)
    set_random_seeds(config.get_data_params()['random_seed'])
    logger.info("Configuration loaded and seeds set.")

    # Validate directory structure
    if not validate_data_structure(config.get_data_params()['processed_path']):
        logger.error("Invalid data directory structure. Exiting...")
        return

    # Print device info
    device_info = get_device_info()
    logger.info(f"Device Info: {device_info}")

    # Create experiment folder
    experiment_path = create_experiment_folder(config.paths_config['results_path'])
    logger.info(f"Experiment folder created at: {experiment_path}")

    # Load datasets
    data_params = config.get_data_params()
    logger.info("Loading datasets...")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_params['processed_path'], 'train'),
        label_mode='binary',
        image_size=data_params['image_size'],
        batch_size=data_params['batch_size'],
        seed=data_params['random_seed']
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_params['processed_path'], 'validation'),
        label_mode='binary',
        image_size=data_params['image_size'],
        batch_size=data_params['batch_size'],
        seed=data_params['random_seed']
    )

    test_ds = tf.keras.utils.image_dataset_from_directory(
        os.path.join(data_params['processed_path'], 'test'),
        label_mode='binary',
        image_size=data_params['image_size'],
        batch_size=data_params['batch_size'],
        shuffle=False
    )

    class_names = train_ds.class_names

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    # Build model
    model_type = config.model_config['architecture']
    logger.info(f"Building model: {model_type}")
    
    if model_type == 'custom_cnn':
        model = CustomCNN(config.get_model_params()['input_shape'], config.get_model_params()['num_classes']).build_model()
    else:
        model_builder = LeukemiaDetectionModel(config)
        model = model_builder.build_model()

    # Setup callbacks
    callbacks = model_builder.create_callbacks(
        checkpoint_path=os.path.join(config.paths_config['checkpoint_path'], 'best_model.h5'),
        logs_path=config.paths_config['logs_path']
    )

    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config.training_config['epochs'],
        callbacks=callbacks
    )

    # Save history
    history_path = os.path.join(experiment_path, 'history.json')
    save_model_history(history, history_path)
    plot_training_history(json.load(open(history_path)), save_path=os.path.join(experiment_path, 'training_history.png'))

    # Fine-tuning
    if model_type != 'custom_cnn':
        logger.info("Starting fine-tuning...")
        model_builder.enable_fine_tuning()
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=config.training_config['epochs'],
            callbacks=callbacks
        )
        fine_tune_path = os.path.join(experiment_path, 'fine_tune_history.json')
        save_model_history(fine_tune_history, fine_tune_path)
        plot_training_history(json.load(open(fine_tune_path)), save_path=os.path.join(experiment_path, 'fine_tune_history.png'))

    # Evaluate
    logger.info("Evaluating model...")
    y_true = tf.concat([y for x, y in test_ds], axis=0).numpy()
    y_pred_proba = model.predict(test_ds).squeeze()
    y_pred = (y_pred_proba >= config.evaluation_config['threshold']).astype(int)

    metrics = calculate_metrics(y_true, y_pred, y_pred_proba)
    metrics_path = os.path.join(experiment_path, 'metrics.json')
    save_metrics(metrics, metrics_path)

    plot_confusion_matrix(y_true, y_pred, class_names, save_path=os.path.join(experiment_path, 'confusion_matrix.png'))

    # Save model
    export_format = config.inference_config['model_format']
    save_path = os.path.join(config.paths_config['model_save_path'], f"leukemia_model.{export_format if export_format == 'h5' else ''}")
    model_builder.save_model(save_path, format=export_format)
    logger.info(f"Model saved to {save_path}")


if __name__ == "__main__":
    main()
=======
"""
Inference module for the Leukemia Detection project.
Supports prediction on single images or batches.
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import List, Union
import cv2
import os

from .model_config import ModelConfig
from .utils import get_class_weights

# Load configuration
config = ModelConfig()
data_params = config.get_data_params()
inference_config = config.inference_config()
class_names = ["healthy", "leukemia"]  # Update if multiclass

def preprocess_image(image_path: str, image_size: tuple) -> np.ndarray:
    """
    Load and preprocess an image for prediction.
    """
    image = cv2.imread(image_path)
    image = cv2.resize(image, image_size)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype("float32") / 255.0
    return image


def load_trained_model(model_path: str = None) -> tf.keras.Model:
    """
    Load a trained Keras model.
    """
    if model_path is None:
        model_path = Path(config.paths_config['checkpoint_path']) / "best_model.h5"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return tf.keras.models.load_model(model_path)


def predict_image(image_path: str, model: tf.keras.Model, threshold: float = 0.5) -> dict:
    """
    Predict class for a single image.
    """
    input_shape = tuple(data_params['image_size'])
    image = preprocess_image(image_path, input_shape)
    image_batch = np.expand_dims(image, axis=0)  # Add batch dimension

    predictions = model.predict(image_batch)

    if predictions.shape[1] == 1:
        # Binary classification
        label_idx = int(predictions[0][0] > threshold)
    else:
        # Multiclass classification
        label_idx = np.argmax(predictions[0])

    return {
        "image_path": image_path,
        "predicted_class": class_names[label_idx],
        "confidence": float(np.max(predictions[0]))
    }


def predict_batch(image_paths: List[str], model: tf.keras.Model, threshold: float = 0.5) -> List[dict]:
    """
    Predict class labels for a batch of images.
    """
    input_shape = tuple(data_params['image_size'])
    results = []

    for image_path in image_paths:
        result = predict_image(image_path, model, threshold)
        results.append(result)

    return results


def infer_from_directory(directory_path: str, model: tf.keras.Model) -> List[dict]:
    """
    Predict on all images in a directory.
    """
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp")
    image_paths = [
        str(Path(directory_path) / f)
        for f in os.listdir(directory_path)
        if f.lower().endswith(image_extensions)
    ]

    return predict_batch(image_paths, model, inference_config['confidence_threshold'])
>>>>>>> e211ebe (commit)
