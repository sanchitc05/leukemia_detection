"""
Training script for the Leukemia Detection project.
Handles data loading, model training, evaluation, and saving results.
"""

import os
import logging
import tensorflow as tf
import numpy as np

from config.model_config import ModelConfig
from src.model_architecture import LeukemiaDetectionModel, CustomCNN
from src.utils import (
    setup_logging, set_random_seeds, create_experiment_folder,
    save_model_history, plot_training_history, calculate_metrics,
    save_metrics, display_sample_images, validate_data_structure
)
from src.data_loader import DataLoader

# Step 1: Setup
config = ModelConfig()
log_cfg = config.config['logging']
setup_logging(log_cfg.get("level", "INFO"), log_cfg.get("file_path", "results/logs/training.log"))
logger = logging.getLogger(__name__)

set_random_seeds(config.get_data_params()['random_seed'])
config.create_directories()

# Step 2: Validate Data Structure
if not validate_data_structure(config.get_data_params()['processed_path']):
    raise ValueError("Data structure is invalid. Please check directory structure.")

# Step 3: Prepare Experiment Folder
experiment_dir = create_experiment_folder(config.paths_config['results_path'])
model_path = os.path.join(experiment_dir, "model.h5")
history_path = os.path.join(experiment_dir, "history.json")
metrics_path = os.path.join(experiment_dir, "metrics.json")
plot_path = os.path.join(experiment_dir, "training_plot.png")
tensorboard_logs = os.path.join(experiment_dir, "logs")

# Step 4: Load Data
logger.info("Loading and preprocessing dataset...")
data_params = config.get_data_params()
data_loader = DataLoader(
    dataset_path=data_params["dataset_path"],
    processed_path=data_params["processed_path"],
    image_size=tuple(data_params["image_size"]),
    seed=data_params["random_seed"]
)
train_ds, val_ds, test_ds, class_names, y_test = data_loader.load_datasets()
logger.info(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}, Test size: {len(test_ds)}")

# Step 5: Build Model
model_config = config.get_model_params()
train_config = config.get_training_params()
logger.info("Building model...")

if model_config['architecture'] == 'custom_cnn':
    custom_model = CustomCNN(
        input_shape=model_config['input_shape'],
        num_classes=model_config['num_classes']
    )
    model = custom_model.build_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=train_config['initial_learning_rate']),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=['accuracy']
    )
else:
    detection_model = LeukemiaDetectionModel(config)
    model = detection_model.build_model()

# Step 6: Callbacks
if model_config['architecture'] != 'custom_cnn':
    callbacks = detection_model.create_callbacks(
        checkpoint_path=os.path.join(experiment_dir, "checkpoint.keras"),
        logs_path=tensorboard_logs
    )
else:
    callbacks = []

# Step 7: Training
logger.info("Starting training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=train_config['epochs'],
    callbacks=callbacks
)

# Save training history
save_model_history(history, history_path)
plot_training_history(history.history, save_path=plot_path)

# Step 8: Fine-tuning (if using pretrained model)
if model_config['architecture'] != 'custom_cnn':
    logger.info("Starting fine-tuning...")
    detection_model.enable_fine_tuning()
    fine_tune_epochs = train_config['epochs'] // 2

    history_fine = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        callbacks=callbacks
    )

    save_model_history(history_fine, history_path.replace(".json", "_fine.json"))
    plot_training_history(history_fine.history, save_path=plot_path.replace(".png", "_fine.png"))

# Step 9: Evaluation
logger.info("Evaluating on test set...")
y_pred_proba = model.predict(test_ds).flatten()
y_pred = (y_pred_proba >= config.evaluation_config['threshold']).astype(int)

metrics = calculate_metrics(np.array(y_test), y_pred, y_pred_proba)
save_metrics(metrics, metrics_path)

# Step 10: Save Model
model_format = config.inference_config['model_format']
logger.info(f"Saving model to: {model_path} in format: {model_format}")

if model_format == "saved_model":
    model.save(model_path.replace(".h5", ""))
else:
    model.save(model_path, save_format='h5')

logger.info("Training complete. Experiment saved at: %s", experiment_dir)
"""
    if model_path is None:
        model_path = Path(config.paths_config['checkpoint_path']) / "best_model.h5"

    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    return tf.keras.models.load_model(model_path)
"""