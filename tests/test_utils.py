"""
Unit tests for utility functions in the Leukemia Detection project.
"""

import unittest
import os
import json
import numpy as np
from pathlib import Path
from src import utils
import tensorflow as tf


class TestUtils(unittest.TestCase):

    def setUp(self):
        # Create temporary directories and files
        self.test_dir = Path("tests/temp_utils")
        self.test_dir.mkdir(parents=True, exist_ok=True)

        # Sample training history
        self.sample_history = {
            'epoch': [1, 2, 3],
            'loss': [0.5, 0.4, 0.3],
            'accuracy': [0.7, 0.8, 0.9],
            'val_loss': [0.6, 0.5, 0.4],
            'val_accuracy': [0.65, 0.75, 0.85],
        }

        self.history_path = self.test_dir / "history.json"
        self.metrics_path = self.test_dir / "metrics.json"

    def tearDown(self):
        # Clean up files after each test
        for file in self.test_dir.glob("*"):
            file.unlink()
        self.test_dir.rmdir()

    def test_save_and_load_model_history(self):
        """Test saving and loading of training history."""
        class DummyHistory:
            def __init__(self, history):
                self.history = history

        dummy_history = DummyHistory(self.sample_history)
        utils.save_model_history(dummy_history, str(self.history_path))
        self.assertTrue(self.history_path.exists())

        loaded_history = utils.load_model_history(str(self.history_path))
        self.assertEqual(loaded_history['epoch'], [1, 2, 3])
        self.assertIn('loss', loaded_history)

    def test_save_and_load_metrics(self):
        """Test saving and loading evaluation metrics."""
        sample_metrics = {
            'accuracy': 0.92,
            'precision': 0.93,
            'recall': 0.91,
            'f1_score': 0.92
        }

        utils.save_metrics(sample_metrics, str(self.metrics_path))
        self.assertTrue(self.metrics_path.exists())

        loaded_metrics = utils.load_metrics(str(self.metrics_path))
        self.assertEqual(loaded_metrics['accuracy'], 0.92)

    def test_get_class_weights(self):
        """Test class weight calculation."""
        y_train = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1])
        weights = utils.get_class_weights(y_train)
        self.assertIsInstance(weights, dict)
        self.assertIn(0, weights)
        self.assertIn(1, weights)

    def test_get_device_info(self):
        """Test device info retrieval."""
        info = utils.get_device_info()
        self.assertIn('gpu_available', info)
        self.assertIn('cpu_count', info)

    def test_create_experiment_folder(self):
        """Test experiment folder creation."""
        folder_path = utils.create_experiment_folder("tests/temp_experiments")
        self.assertTrue(Path(folder_path).exists())

        # Clean up
        Path(folder_path).rmdir()
        Path("tests/temp_experiments").rmdir()


if __name__ == "__main__":
    unittest.main()