"""
Unit tests for the data augmentation module in the Leukemia Detection project.
"""

import unittest
import numpy as np
import tensorflow as tf
from src.data_augmentation import get_augmentation_pipeline


class TestDataAugmentation(unittest.TestCase):
    def setUp(self):
        self.image_shape = (224, 224, 3)
        self.batch_size = 8

        # Dummy image batch: (batch_size, height, width, channels)
        self.sample_images = tf.random.uniform(
            shape=(self.batch_size, *self.image_shape),
            minval=0, maxval=255, dtype=tf.float32
        ) / 255.0

    def test_augmentation_pipeline_output_shape(self):
        """Check if output shape matches input shape after augmentation."""
        pipeline = get_augmentation_pipeline()
        augmented_images = pipeline(self.sample_images)

        self.assertEqual(augmented_images.shape, self.sample_images.shape)
        self.assertEqual(augmented_images.dtype, tf.float32)

    def test_pipeline_is_deterministic_in_graph_mode(self):
        """Ensure that augmentation runs in graph mode and does not crash."""
        @tf.function
        def apply_pipeline(images):
            pipeline = get_augmentation_pipeline()
            return pipeline(images)

        output = apply_pipeline(self.sample_images)
        self.assertEqual(output.shape, self.sample_images.shape)

    def test_pixel_value_range(self):
        """Check if augmented pixel values remain in valid [0, 1] range."""
        pipeline = get_augmentation_pipeline()
        augmented = pipeline(self.sample_images)
        self.assertTrue(tf.reduce_max(augmented) <= 1.0)
        self.assertTrue(tf.reduce_min(augmented) >= 0.0)


if __name__ == '__main__':
    unittest.main()