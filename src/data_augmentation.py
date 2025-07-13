import tensorflow as tf
from tensorflow import Tensor

# Assuming ModelConfig is defined elsewhere to load the YAML configuration including augmentation params
from config import ModelConfig

class DataAugmentor:
    """
    Data augmentation class that builds a tf.keras.Sequential pipeline based on
    configuration parameters. The augmentation settings are read from the
    'augmentation' section of the config loaded via ModelConfig.
    """
    def __init__(self) -> None:
        """
        Initialize the DataAugmentor by loading configuration and building
        the augmentation pipeline dynamically.
        """
        self.config = ModelConfig()
        self.augmentation_config = getattr(self.config, 'augmentation', None)
        # Build the augmentation pipeline on initialization
        self._pipeline = self._build_pipeline()

    def _build_pipeline(self) -> tf.keras.Sequential:
        """
        Build a tf.keras.Sequential model consisting of augmentation layers
        conditionally added based on configuration values.

        Returns:
            tf.keras.Sequential: Sequential model with augmentation layers.
        """
        layers = []

        if self.augmentation_config is None:
            return tf.keras.Sequential(layers)

        # Random Flip
        flip_cfg = getattr(self.augmentation_config, 'random_flip', None)
        if flip_cfg:
            # Determine mode for RandomFlip based on config (horizontal, vertical, or both)
            horizontal = getattr(flip_cfg, 'horizontal', False) or False
            vertical = getattr(flip_cfg, 'vertical', False) or False
            if horizontal and vertical:
                mode = "horizontal_and_vertical"
            elif horizontal:
                mode = "horizontal"
            elif vertical:
                mode = "vertical"
            else:
                mode = None
            if mode:
                layers.append(tf.keras.layers.RandomFlip(mode=mode))

        # Random Rotation
        rot_cfg = getattr(self.augmentation_config, 'random_rotation', None)
        if rot_cfg:
            # Expect factor or list/tuple for rotation range
            factor = getattr(rot_cfg, 'factor', None)
            if factor:
                layers.append(tf.keras.layers.RandomRotation(factor=factor))

        # Random Zoom
        zoom_cfg = getattr(self.augmentation_config, 'random_zoom', None)
        if zoom_cfg:
            # Height and width zoom factors
            height_factor = getattr(zoom_cfg, 'height_factor', 0.0) or 0.0
            width_factor = getattr(zoom_cfg, 'width_factor', 0.0) or 0.0
            if height_factor or width_factor:
                layers.append(tf.keras.layers.RandomZoom(
                    height_factor=height_factor,
                    width_factor=width_factor
                ))

        # Random Translation (Shift)
        trans_cfg = getattr(self.augmentation_config, 'random_translation', None)
        if trans_cfg:
            # Height and width shift factors
            height_factor = getattr(trans_cfg, 'height_factor', 0.0) or 0.0
            width_factor = getattr(trans_cfg, 'width_factor', 0.0) or 0.0
            if height_factor or width_factor:
                layers.append(tf.keras.layers.RandomTranslation(
                    height_factor=height_factor,
                    width_factor=width_factor
                ))

        # Random Brightness
        bright_cfg = getattr(self.augmentation_config, 'random_brightness', None)
        if bright_cfg:
            factor = getattr(bright_cfg, 'factor', None)
            if factor:
                layers.append(tf.keras.layers.RandomBrightness(factor=factor))

        # Random Contrast
        contrast_cfg = getattr(self.augmentation_config, 'random_contrast', None)
        if contrast_cfg:
            factor = getattr(contrast_cfg, 'factor', None)
            if factor:
                layers.append(tf.keras.layers.RandomContrast(factor=factor))

        return tf.keras.Sequential(layers)

    def get_augmentation_pipeline(self) -> tf.keras.Sequential:
        """
        Get the TensorFlow augmentation pipeline built from configuration.

        Returns:
            tf.keras.Sequential: A model/layer sequence applying all configured augmentations.
        """
        return self._pipeline

    def apply(self, image: Tensor) -> Tensor:
        """
        Apply the augmentation pipeline to a given image tensor.

        Args:
            image (tf.Tensor): Input image tensor to be augmented.

        Returns:
            tf.Tensor: Augmented image tensor.
        """
        # Ensure the pipeline is in training mode to apply random augmentations
        return self._pipeline(image, training=True)