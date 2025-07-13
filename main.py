import os
import logging
from src.model_config import ModelConfig
from utils import set_random_seed, configure_logging, create_dirs
from data_preprocessing import prepare_data, augment_data
from model_architecture import build_custom_cnn, build_transfer_model
import training
import evaluation

def main():
    """
    Main script to run the full pipeline for Leukemia Detection project.
    """
    # =============================
    # 1. Load Configuration
    # =============================
    config = ModelConfig()  # Load model configuration (e.g., from YAML or arguments)
    
    # =============================
    # 2. Set Random Seed and Logging
    # =============================
    # Set random seeds for reproducibility
    set_random_seed(config.seed)
    # Configure logging
    configure_logging(
        log_file=config.paths.log_file, 
        level=config.logging.level
    )
    logger = logging.getLogger(__name__)
    logger.info("Configuration and logging set up successfully.")
    
    # =============================
    # 3. Create Required Directories
    # =============================
    # Ensure output directories exist
    output_dirs = [
        config.paths.model_dir,
        config.paths.history_dir,
        config.paths.metrics_dir,
        config.paths.plots_dir,
        config.paths.log_dir
    ]
    create_dirs(output_dirs)
    logger.info("Required directories created or already exist.")
    
    # =============================
    # 4. Prepare Datasets
    # =============================
    logger.info("Preparing datasets...")
    # Example: prepare_data might load images and split into train/val/test
    train_dataset, val_dataset, test_dataset = prepare_data(config.data)
    logger.info(f"Datasets prepared: {len(train_dataset)} training samples, "
                f"{len(val_dataset)} validation samples, {len(test_dataset)} test samples.")
    
    # =============================
    # 5. Data Augmentation (Optional)
    # =============================
    if config.data.augmentation and config.data.augmentation.enabled:
        logger.info("Applying data augmentation to training data...")
        train_dataset = augment_data(train_dataset, config.data.augmentation)
        logger.info("Data augmentation applied.")
    
    # =============================
    # 6. Build Model Architecture
    # =============================
    logger.info("Building the model architecture...")
    if config.model.use_transfer_learning:
        model = build_transfer_model(config.model)
        logger.info("Transfer learning model built.")
    else:
        model = build_custom_cnn(config.model)
        logger.info("Custom CNN model built.")
    
    # =============================
    # 7. Train the Model
    # =============================
    logger.info("Starting model training...")
    history = training.train_model(model, train_dataset, val_dataset, config.training)
    logger.info("Model training completed.")
    
    # =============================
    # 8. Evaluate the Model
    # =============================
    logger.info("Evaluating the model on test set...")
    metrics = evaluation.evaluate_model(model, test_dataset, config.evaluation)
    logger.info(f"Evaluation metrics: {metrics}")
    
    # =============================
    # 9. Save Model, History, Metrics, and Plots
    # =============================
    # Save the trained model
    model_save_path = os.path.join(config.paths.model_dir, config.model.filename)
    logger.info(f"Saving trained model to {model_save_path}...")
    model.save(model_save_path)
    
    # Save training history
    history_save_path = os.path.join(config.paths.history_dir, config.training.history_file)
    logger.info(f"Saving training history to {history_save_path}...")
    training.save_history(history, history_save_path)
    
    # Save evaluation metrics
    metrics_save_path = os.path.join(config.paths.metrics_dir, config.evaluation.metrics_file)
    logger.info(f"Saving evaluation metrics to {metrics_save_path}...")
    evaluation.save_metrics(metrics, metrics_save_path)
    
    # Plot and save relevant plots (e.g., loss/accuracy curves, confusion matrix)
    logger.info("Generating and saving plots...")
    evaluation.plot_training_history(history, os.path.join(config.paths.plots_dir, config.plots.history))
    evaluation.plot_confusion_matrix(model, test_dataset, os.path.join(config.paths.plots_dir, config.plots.confusion_matrix))
    
    logger.info("Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()