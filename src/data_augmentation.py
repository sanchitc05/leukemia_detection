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
        self.augmentation_config = getattr(self.config, 'augmentation_config', None)
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
        flip_cfg = self.augmentation_config.get('random_flip')
        if flip_cfg:
            horizontal = flip_cfg.get('horizontal', False)
            vertical = flip_cfg.get('vertical', False)
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
        rot_cfg = self.augmentation_config.get('random_rotation')
        if rot_cfg:
            factor = rot_cfg.get('factor', None)
            if factor:
                layers.append(tf.keras.layers.RandomRotation(factor=factor))

        # Random Zoom
        zoom_cfg = self.augmentation_config.get('random_zoom')
        if zoom_cfg:
            height_factor = zoom_cfg.get('height_factor', 0.0)
            width_factor = zoom_cfg.get('width_factor', 0.0)
            if height_factor or width_factor:
                layers.append(tf.keras.layers.RandomZoom(
                    height_factor=height_factor,
                    width_factor=width_factor
                ))

        # Random Translation
        trans_cfg = self.augmentation_config.get('random_translation')
        if trans_cfg:
            height_factor = trans_cfg.get('height_factor', 0.0)
            width_factor = trans_cfg.get('width_factor', 0.0)
            if height_factor or width_factor:
                layers.append(tf.keras.layers.RandomTranslation(
                    height_factor=height_factor,
                    width_factor=width_factor
                ))

        # Random Brightness (custom implementation via Lambda if needed)
        bright_cfg = self.augmentation_config.get('random_brightness')
        if bright_cfg:
            factor = bright_cfg.get('factor', 0.0)
            if factor:
                layers.append(tf.keras.layers.Lambda(
                    lambda x: tf.image.random_brightness(x, max_delta=factor)
                ))

        # Random Contrast
        contrast_cfg = self.augmentation_config.get('random_contrast')
        if contrast_cfg:
            factor = contrast_cfg.get('factor', None)
            if factor:
                layers.append(tf.keras.layers.RandomContrast(factor=factor))

        return tf.keras.Sequential(layers)
    # ---------------------------------------------
# Exportable function for convenient importing
# ---------------------------------------------
    def get_augmentation_pipeline() -> tf.keras.Sequential:
        """
        Convenience function to get the augmentation pipeline.

         Returns:
        tf.keras.Sequential: Augmentation pipeline from config.
        """
        return DataAugmentor().get_augmentation_pipeline()

    def apply(self, image: Tensor) -> Tensor:
        """
        Apply the augmentation pipeline to a given image tensor.

        Args:
            image (tf.Tensor): Input image tensor to be augmented.

        Returns:
            tf.Tensor: Augmented image tensor.
        """
        return self._pipeline(image, training=True)



