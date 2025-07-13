"""
Utility functions for the Leukemia Detection project.
Contains helper functions for data processing, visualization, and model utilities.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import json
from datetime import datetime


def setup_logging(log_level: str = "INFO", log_file: str = "results/logs/training.log"):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Path to log file
    """
    # Create logs directory if it doesn't exist
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )


def set_random_seeds(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_device_info():
    """
    Get information about available devices (GPU/CPU).
    
    Returns:
        Dict containing device information
    """
    device_info = {
        'gpu_available': tf.config.list_physical_devices('GPU'),
        'gpu_count': len(tf.config.list_physical_devices('GPU')),
        'cpu_count': len(tf.config.list_physical_devices('CPU')),
        'tensorflow_version': tf.__version__
    }
    
    if device_info['gpu_available']:
        for gpu in device_info['gpu_available']:
            device_info['gpu_name'] = tf.config.experimental.get_device_details(gpu)
    
    return device_info


def save_model_history(history: tf.keras.callbacks.History, save_path: str):
    """
    Save training history to JSON file.
    
    Args:
        history: Keras training history object
        save_path: Path to save the history
    """
    history_dict = {
        'epoch': list(range(1, len(history.history['loss']) + 1)),
        **{key: [float(val) for val in values] for key, values in history.history.items()}
    }
    
    with open(save_path, 'w') as f:
        json.dump(history_dict, f, indent=2)


def load_model_history(load_path: str) -> Dict:
    """
    Load training history from JSON file.
    
    Args:
        load_path: Path to load the history
        
    Returns:
        Dictionary containing training history
    """
    with open(load_path, 'r') as f:
        history_dict = json.load(f)
    
    return history_dict


def plot_training_history(history: Dict, save_path: str = None):
    """
    Plot training history (loss and metrics).
    
    Args:
        history: Training history dictionary
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training History', fontsize=16)
    
    # Plot loss
    axes[0, 0].plot(history['epoch'], history['loss'], 'b-', label='Training Loss')
    axes[0, 0].plot(history['epoch'], history['val_loss'], 'r-', label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot accuracy
    axes[0, 1].plot(history['epoch'], history['accuracy'], 'b-', label='Training Accuracy')
    axes[0, 1].plot(history['epoch'], history['val_accuracy'], 'r-', label='Validation Accuracy')
    axes[0, 1].set_title('Model Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Plot precision if available
    if 'precision' in history:
        axes[1, 0].plot(history['epoch'], history['precision'], 'b-', label='Training Precision')
        axes[1, 0].plot(history['epoch'], history['val_precision'], 'r-', label='Validation Precision')
        axes[1, 0].set_title('Model Precision')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Plot recall if available
    if 'recall' in history:
        axes[1, 1].plot(history['epoch'], history['recall'], 'b-', label='Training Recall')
        axes[1, 1].plot(history['epoch'], history['val_recall'], 'r-', label='Validation Recall')
        axes[1, 1].set_title('Model Recall')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Recall')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         class_names: List[str], save_path: str = None):
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        save_path: Path to save the plot
    """
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_pred_proba: np.ndarray = None) -> Dict:
    """
    Calculate comprehensive evaluation metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
        
    Returns:
        Dictionary containing various metrics
    """
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, classification_report
    )
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
    
    # Detailed classification report
    metrics['classification_report'] = classification_report(
        y_true, y_pred, target_names=['Healthy', 'Leukemia'], output_dict=True
    )
    
    return metrics


def save_metrics(metrics: Dict, save_path: str):
    """
    Save metrics to JSON file.
    
    Args:
        metrics: Metrics dictionary
        save_path: Path to save the metrics
    """
    # Convert numpy types to native Python types for JSON serialization
    serializable_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, np.ndarray):
            serializable_metrics[key] = value.tolist()
        elif isinstance(value, (np.int64, np.float64)):
            serializable_metrics[key] = value.item()
        else:
            serializable_metrics[key] = value
    
    with open(save_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)


def load_metrics(load_path: str) -> Dict:
    """
    Load metrics from JSON file.
    
    Args:
        load_path: Path to load the metrics
        
    Returns:
        Dictionary containing metrics
    """
    with open(load_path, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def create_experiment_folder(base_path: str = "results/experiments") -> str:
    """
    Create a new experiment folder with timestamp.
    
    Args:
        base_path: Base path for experiments
        
    Returns:
        Path to the created experiment folder
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_path = Path(base_path) / f"experiment_{timestamp}"
    experiment_path.mkdir(parents=True, exist_ok=True)
    
    return str(experiment_path)


def get_class_weights(y_train: np.ndarray) -> Dict:
    """
    Calculate class weights for imbalanced datasets.
    
    Args:
        y_train: Training labels
        
    Returns:
        Dictionary containing class weights
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y_train)
    class_weights = compute_class_weight(
        'balanced', classes=classes, y=y_train
    )
    
    return {i: weight for i, weight in enumerate(class_weights)}


def display_sample_images(dataset: tf.data.Dataset, class_names: List[str], 
                         num_samples: int = 16):
    """
    Display sample images from dataset.
    
    Args:
        dataset: TensorFlow dataset
        class_names: List of class names
        num_samples: Number of samples to display
    """
    plt.figure(figsize=(12, 12))
    
    for i, (image, label) in enumerate(dataset.take(num_samples)):
        plt.subplot(4, 4, i + 1)
        plt.imshow(image.numpy().astype("uint8"))
        plt.title(f"Class: {class_names[label.numpy()]}")
        plt.axis("off")
    
    plt.tight_layout()
    plt.show()


def get_model_summary(model: tf.keras.Model) -> str:
    """
    Get detailed model summary as string.
    
    Args:
        model: Keras model
        
    Returns:
        Model summary as string
    """
    import io
    
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    
    return summary_string


def print_system_info():
    """Print system and environment information."""
    device_info = get_device_info()
    
    print("=" * 50)
    print("SYSTEM INFORMATION")
    print("=" * 50)
    print(f"TensorFlow Version: {device_info['tensorflow_version']}")
    print(f"GPU Available: {bool(device_info['gpu_available'])}")
    print(f"GPU Count: {device_info['gpu_count']}")
    print(f"CPU Count: {device_info['cpu_count']}")
    
    if device_info['gpu_available']:
        print(f"GPU Details: {device_info.get('gpu_name', 'N/A')}")
    
    print("=" * 50)


def validate_data_structure(data_path: str) -> bool:
    """
    Validate the data directory structure.
    
    Args:
        data_path: Path to data directory
        
    Returns:
        True if structure is valid, False otherwise
    """
    required_dirs = [
        'train/healthy', 'train/leukemia',
        'validation/healthy', 'validation/leukemia',
        'test/healthy', 'test/leukemia'
    ]
    
    for dir_name in required_dirs:
        full_path = Path(data_path) / dir_name
        if not full_path.exists():
            logging.error(f"Required directory not found: {full_path}")
            return False
    
    return True