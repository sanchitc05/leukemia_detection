<<<<<<< HEAD
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
from src.data_preprocessing import DataLoader

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
data_loader = DataLoader(config)
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
=======
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
>>>>>>> e211ebe (commit)
