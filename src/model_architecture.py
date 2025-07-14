<<<<<<< HEAD
=======

>>>>>>> e211ebe (commit)
"""
Model architecture module for Leukemia Detection project.
Contains CNN model definitions and transfer learning implementations.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, applications
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall, AUC
import logging
from typing import Dict, Tuple, Optional, List
from config.model_config import ModelConfig


class LeukemiaDetectionModel:
    """Model class for leukemia detection using CNN and transfer learning."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.model_config = config.get_model_params()
        self.training_config = config.get_training_params()
        self.logger = logging.getLogger(__name__)
        
        self.model = None
        self.base_model = None
        
    def build_model(self) -> tf.keras.Model:
        """
        Build the complete model architecture.
        
        Returns:
            Compiled Keras model
        """
        self.logger.info(f"Building model with architecture: {self.model_config['architecture']}")
        
        # Create base model
        self.base_model = self._create_base_model()
        
        # Create complete model
        self.model = self._create_complete_model()
        
        # Compile model
        self._compile_model()
        
        self.logger.info("Model built successfully")
        return self.model
    
    def _create_base_model(self) -> tf.keras.Model:
        """Create the base model (pretrained backbone)."""
        input_shape = self.model_config['input_shape']
        pretrained_weights = self.model_config['pretrained_weights']
        
        if self.model_config['architecture'] == 'efficientnet_b0':
            base_model = applications.EfficientNetB0(
                input_shape=input_shape,
                include_top=False,
                weights=pretrained_weights
            )
        elif self.model_config['architecture'] == 'efficientnet_b1':
            base_model = applications.EfficientNetB1(
                input_shape=input_shape,
                include_top=False,
                weights=pretrained_weights
            )
        elif self.model_config['architecture'] == 'resnet50':
            base_model = applications.ResNet50(
                input_shape=input_shape,
                include_top=False,
                weights=pretrained_weights
            )
        elif self.model_config['architecture'] == 'densenet121':
            base_model = applications.DenseNet121(
                input_shape=input_shape,
                include_top=False,
                weights=pretrained_weights
            )
        elif self.model_config['architecture'] == 'vgg16':
            base_model = applications.VGG16(
                input_shape=input_shape,
                include_top=False,
                weights=pretrained_weights
            )
        elif self.model_config['architecture'] == 'mobilenet_v2':
            base_model = applications.MobileNetV2(
                input_shape=input_shape,
                include_top=False,
                weights=pretrained_weights
            )
        else:
            raise ValueError(f"Unsupported architecture: {self.model_config['architecture']}")
        
        # Freeze base model initially
        base_model.trainable = False
        
        return base_model
    
    def _create_complete_model(self) -> tf.keras.Model:
        """Create the complete model with custom head."""
        inputs = keras.Input(shape=self.model_config['input_shape'])
        
        # Data augmentation layers (applied during training)
        x = layers.RandomRotation(0.1)(inputs)
        x = layers.RandomZoom(0.1)(x)
        x = layers.RandomFlip('horizontal')(x)
        
        # Base model
        x = self.base_model(x, training=False)
        
        # Custom head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.model_config['dropout_rate'])(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.model_config['dropout_rate'])(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(self.model_config['dropout_rate'] / 2)(x)
        
        # Output layer
        if self.model_config['num_classes'] == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.model_config['num_classes'], activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model
    
    def _compile_model(self):
        """Compile the model with optimizer, loss, and metrics."""
        # Optimizer
        if self.training_config['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=self.training_config['initial_learning_rate'])
        elif self.training_config['optimizer'] == 'sgd':
            optimizer = SGD(learning_rate=self.training_config['initial_learning_rate'])
        else:
            raise ValueError(f"Unsupported optimizer: {self.training_config['optimizer']}")
        
        # Loss function
        if self.training_config['loss_function'] == 'binary_crossentropy':
            loss = BinaryCrossentropy()
        elif self.training_config['loss_function'] == 'categorical_crossentropy':
            loss = keras.losses.CategoricalCrossentropy()
        else:
            raise ValueError(f"Unsupported loss function: {self.training_config['loss_function']}")
        
        # Metrics
        metrics = ['accuracy']
        if 'precision' in self.training_config['metrics']:
            metrics.append(Precision(name='precision'))
        if 'recall' in self.training_config['metrics']:
            metrics.append(Recall(name='recall'))
        if 'auc' in self.training_config['metrics']:
            metrics.append(AUC(name='auc'))
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )
    
    def enable_fine_tuning(self, learning_rate: Optional[float] = None):
        """
        Enable fine-tuning by unfreezing top layers of base model.
        
        Args:
            learning_rate: Learning rate for fine-tuning
        """
        if learning_rate is None:
            learning_rate = self.training_config['fine_tune_learning_rate']
        
        # Unfreeze top layers
        self.base_model.trainable = True
        
        # Freeze bottom layers
        fine_tune_at = len(self.base_model.layers) - self.model_config['fine_tune_layers']
        
        for layer in self.base_model.layers[:fine_tune_at]:
            layer.trainable = False
        
        # Recompile with lower learning rate
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss=self.model.loss,
            metrics=self.model.metrics
        )
        
        self.logger.info(f"Fine-tuning enabled with learning rate: {learning_rate}")
        self.logger.info(f"Unfrozen layers: {self.model_config['fine_tune_layers']}")
    
    def get_model_summary(self) -> str:
        """Get model summary as string."""
        if self.model is None:
            return "Model not built yet"
        
        import io
        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
        
        return summary_string
    
    def count_parameters(self) -> Dict[str, int]:
        """Count trainable and non-trainable parameters."""
        if self.model is None:
            return {"error": "Model not built yet"}
        
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.trainable_weights])
        non_trainable_params = sum([tf.keras.backend.count_params(w) for w in self.model.non_trainable_weights])
        
        return {
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": non_trainable_params,
            "total_parameters": trainable_params + non_trainable_params
        }
    
    def save_model(self, save_path: str, format: str = 'tf'):
        """
        Save the trained model.
        
        Args:
            save_path: Path to save the model
            format: Save format ('tf' for SavedModel, 'h5' for HDF5)
        """
        if self.model is None:
            raise ValueError("Model not built yet")
        
        if format == 'tf':
            self.model.save(save_path)
        elif format == 'h5':
            self.model.save(save_path, save_format='h5')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Model saved to {save_path}")
    
    def load_model(self, load_path: str):
        """
        Load a trained model.
        
        Args:
            load_path: Path to load the model from
        """
        self.model = keras.models.load_model(load_path)
        self.logger.info(f"Model loaded from {load_path}")
    
    def create_callbacks(self, checkpoint_path: str, logs_path: str) -> List[tf.keras.callbacks.Callback]:
        """
        Create training callbacks.
        
        Args:
            checkpoint_path: Path to save checkpoints
            logs_path: Path to save logs
            
        Returns:
            List of callbacks
        """
        callbacks = []
        
        # Model checkpoint
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1
        )
        callbacks.append(checkpoint_callback)
        
        # Early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.training_config['early_stopping_patience'],
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stopping)
        
        # Learning rate reduction
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=self.training_config['reduce_lr_factor'],
            patience=self.training_config['reduce_lr_patience'],
            min_lr=1e-7,
            verbose=1
        )
        callbacks.append(reduce_lr)
        
        # TensorBoard
        tensorboard = tf.keras.callbacks.TensorBoard(
            log_dir=logs_path,
            histogram_freq=1,
            write_graph=True,
            write_images=True,
            update_freq='epoch'
        )
        callbacks.append(tensorboard)
        
        # Custom callback for learning rate scheduling
        lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
            self._lr_schedule,
            verbose=1
        )
        # callbacks.append(lr_scheduler)  # Optional
        
        return callbacks
    
    def _lr_schedule(self, epoch: int, lr: float) -> float:
        """
        Learning rate schedule function.
        
        Args:
            epoch: Current epoch
            lr: Current learning rate
            
        Returns:
            New learning rate
        """
        if epoch < 10:
            return lr
        else:
            return lr * tf.math.exp(-0.1)


class CustomCNN:
    """Custom CNN architecture for leukemia detection."""
    
    def __init__(self, input_shape: Tuple[int, int, int], num_classes: int):
        """
        Initialize custom CNN.
        
        Args:
            input_shape: Input shape (height, width, channels)
            num_classes: Number of output classes
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        
    def build_model(self) -> tf.keras.Model:
        """Build custom CNN model."""
        inputs = keras.Input(shape=self.input_shape)
        
        # First convolutional block
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Second convolutional block
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Third convolutional block
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Fourth convolutional block
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        x = layers.Dropout(0.25)(x)
        
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(x)
        
        # Dense layers
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        x = layers.Dense(256, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.5)(x)
        
        # Output layer
        if self.num_classes == 2:
            outputs = layers.Dense(1, activation='sigmoid')(x)
        else:
            outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        
        model = keras.Model(inputs, outputs)
        
        return model


def create_ensemble_model(models: List[tf.keras.Model], 
                         input_shape: Tuple[int, int, int]) -> tf.keras.Model:
    """
    Create an ensemble model from multiple trained models.
    
    Args:
        models: List of trained models
        input_shape: Input shape
        
    Returns:
        Ensemble model
    """
    inputs = keras.Input(shape=input_shape)
    
    # Get predictions from all models
    predictions = []
    for i, model in enumerate(models):
        # Freeze the model
        model.trainable = False
        pred = model(inputs)
        predictions.append(pred)
    
    # Average predictions
    if len(predictions) > 1:
        averaged = layers.Average()(predictions)
    else:
        averaged = predictions[0]