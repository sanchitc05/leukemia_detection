"""
Unit tests for data preprocessing functions in the Leukemia Detection project.
"""

import unittest
import tensorflow as tf
from pathlib import Path
from src.model_config import ModelConfig
from src.data_preprocessing import create_tf_datasets


class TestDataPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load config
        cls.config = ModelConfig()
        cls.data_config = cls.config.get_data_params()

    def test_processed_dirs_exist(self):
        """Test if processed data directories exist and are correctly structured."""
        base_path = Path(self.data_config['processed_path'])

        required_dirs = [
            "train/healthy", "train/leukemia",
            "validation/healthy", "validation/leukemia",
            "test/healthy", "test/leukemia"
        ]

        for subdir in required_dirs:
            full_path = base_path / subdir
            self.assertTrue(full_path.exists(), f"Missing directory: {full_path}")

    def test_tf_datasets_shapes_and_types(self):
        """Test if datasets are returned and have correct shapes and types."""
        train_ds, val_ds, test_ds = create_tf_datasets(self.config)

        sample_image, sample_label = next(iter(train_ds))

        expected_shape = tuple(self.data_config["image_size"]) + (3,)
        self.assertEqual(sample_image.shape[1:], expected_shape)
        self.assertTrue(isinstance(sample_label.numpy()[0], (int, float)))

        # Check batched shapes
        self.assertEqual(sample_image.shape[0], self.data_config["batch_size"])

    def test_tf_datasets_not_empty(self):
        """Check that the datasets are not empty."""
        train_ds, val_ds, test_ds = create_tf_datasets(self.config)

        train_count = sum(1 for _ in train_ds)
        val_count = sum(1 for _ in val_ds)
        test_count = sum(1 for _ in test_ds)

        self.assertGreater(train_count, 0, "Training dataset is empty.")
        self.assertGreater(val_count, 0, "Validation dataset is empty.")
        self.assertGreater(test_count, 0, "Test dataset is empty.")


if __name__ == "__main__":
    unittest.main()