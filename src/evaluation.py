import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Rescaling
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import json

from config.model_config import ModelConfig
from utils import (
    load_model_history,
    plot_training_history,
    plot_confusion_matrix,
    calculate_metrics,
    save_metrics,
)

# Load config
config = ModelConfig()
model_params = config.get_model_params()
data_params = config.get_data_params()
paths = config.paths_config
eval_config = config.evaluation_config

def evaluate_model():
    """
    Evaluate trained leukemia detection model (binary or multiclass).
    """
    num_classes = model_params['num_classes']
    image_size = data_params['image_size']
    batch_size = data_params['batch_size']
    threshold = eval_config['threshold']

    # Load test dataset
    test_dir = os.path.join(data_params['processed_path'], 'test')
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    class_names = test_ds.class_names
    print(f"Class names: {class_names}")

    # Normalization layer
    normalization_layer = Rescaling(1. / 255)
    test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

    # Load trained model
    model_path = os.path.join(paths['model_save_path'], "best_model")
    model = load_model(model_path)

    # Predict
    y_true = np.concatenate([y.numpy() for _, y in test_ds])
    y_pred_proba = model.predict(test_ds)

    # Classification type handling
    if num_classes == 2:
        y_pred = (y_pred_proba > threshold).astype(int).flatten()
    else:
        y_pred = np.argmax(y_pred_proba, axis=1)

    # Metrics
    metrics = calculate_metrics(y_true, y_pred, y_pred_proba if num_classes == 2 else None)

    # Save metrics
    metrics_save_path = os.path.join(paths['results_path'], "evaluation_metrics.json")
    save_metrics(metrics, metrics_save_path)

    # Confusion matrix
    cm_plot_path = os.path.join(paths['plots_path'], "confusion_matrix.png")
    plot_confusion_matrix(y_true, y_pred, class_names, save_path=cm_plot_path)

    print("Evaluation complete. Metrics and plots saved.")

    # Optional: Load and plot training history
    history_path = os.path.join(paths['results_path'], "history.json")
    if os.path.exists(history_path):
        history = load_model_history(history_path)
        plot_training_history(history, save_path=os.path.join(paths['plots_path'], "training_history.png"))

if __name__ == "__main__":
    evaluate_model()