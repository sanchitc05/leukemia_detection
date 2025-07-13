"""
Script to run inference on new image(s) using a trained model.
Supports batch predictions and confidence filtering.
"""

import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from pathlib import Path
from src.model_config import config
from src.utils import setup_logging

# Setup logging
setup_logging(log_file="results/logs/predict.log")
import logging
logger = logging.getLogger(__name__)


def load_and_preprocess_image(img_path, target_size):
    """Load and preprocess a single image."""
    img = Image.open(img_path).convert("RGB")
    img = img.resize(target_size)
    img = np.array(img) / 255.0
    return img


def predict_single(model, image_path, class_names):
    """Run prediction on a single image."""
    input_shape = tuple(config.model_config["input_shape"][:2])
    image = load_and_preprocess_image(image_path, input_shape)
    image_batch = np.expand_dims(image, axis=0)
    preds = model.predict(image_batch)

    if config.model_config["num_classes"] == 2:
        label = int(preds[0] > config.inference_config["confidence_threshold"])
        prob = float(preds[0])
    else:
        label = np.argmax(preds[0])
        prob = float(np.max(preds[0]))

    logger.info(f"Prediction: {class_names[label]} (Confidence: {prob:.4f})")
    return class_names[label], prob


def main():
    parser = argparse.ArgumentParser(description="Run inference on image(s).")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to image or folder of images")
    args = parser.parse_args()

    input_path = Path(args.image_dir)
    assert input_path.exists(), f"Path not found: {input_path}"

    # Load model
    model_dir = Path(config.paths_config["model_save_path"])
    latest_model = sorted(model_dir.glob("*"), key=os.path.getmtime)[-1]
    model = tf.keras.models.load_model(latest_model)
    logger.info(f"Loaded model from {latest_model}")

    class_names = ["Healthy", "Leukemia"] if config.model_config["num_classes"] == 2 else list(range(config.model_config["num_classes"]))

    # Batch prediction or single image
    image_paths = list(input_path.glob("*")) if input_path.is_dir() else [input_path]

    for img_path in image_paths:
        logger.info(f"Predicting for: {img_path.name}")
        label, prob = predict_single(model, img_path, class_names)
        print(f"{img_path.name} → {label} ({prob:.2%})")


if __name__ == "__main__":
    main()