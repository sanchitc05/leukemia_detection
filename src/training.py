import logging
from src.model_config import ModelConfig
from src.model_architecture import LeukemiaDetectionModel
from src.data_loader import DataLoader
from src.utils import (
    setup_logging, print_system_info, create_experiment_folder,
    plot_training_history, save_model_history
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from pathlib import Path

def run_training_pipeline():
    # Setup logging
    setup_logging(log_level="INFO")
    logger = logging.getLogger(__name__)

    # Print system info
    print_system_info()

    # Load config
    config = ModelConfig()
    model_params = config.get_model_params()
    training_params = config.get_training_params()
    data_params = config.get_data_params()
    paths = config.paths

    # Load data
    loader = DataLoader(
        dataset_path=data_params["dataset_path"],
        processed_path=data_params["processed_path"],
        image_size=tuple(data_params["image_size"]),
        seed=data_params["random_seed"]
    )

    train_ds, val_ds, test_ds, class_names = loader.get_tf_datasets(
        batch_size=data_params["batch_size"]
    )

    # Augmentation pipeline
    from src.data_augmentation import get_augmentation_pipeline
    augmenter = get_augmentation_pipeline(config.augmentation_config)
    train_ds = train_ds.map(lambda x, y: (augmenter(x, training=True), y))

    # Setup paths
    experiment_path = create_experiment_folder()
    checkpoint_path = Path(paths["checkpoint_path"]) / "best_model.keras"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = Path(experiment_path) / "history.json"
    plot_path = Path(experiment_path) / "training_plot.png"

    # Build and compile model
    model = LeukemiaDetectionModel(config).build_model()
    model.compile(
        optimizer=training_params["optimizer"],
        loss=training_params["loss_function"],
        metrics=training_params["metrics"]
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=training_params["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=training_params["reduce_lr_factor"],
            patience=training_params["reduce_lr_patience"],
            verbose=1
        ),
        ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=training_params["epochs"],
        callbacks=callbacks
    )

    # Save history
    save_model_history(history, str(history_path))
    plot_training_history(
        {
            'epoch': list(range(1, len(history.history['loss']) + 1)),
            **history.history
        },
        save_path=str(plot_path)
    )

    logger.info(f"✅ Training complete.")
    logger.info(f"📦 Best model saved at: {checkpoint_path}")
    logger.info(f"📊 Training history saved at: {history_path}")
    logger.info(f"📈 Training plot saved at: {plot_path}")
