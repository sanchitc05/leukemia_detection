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