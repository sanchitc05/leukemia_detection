"""
Unit tests for model inference in the Leukemia Detection project.
"""

import unittest
import numpy as np
import tensorflow as tf
from src.model_architecture import LeukemiaDetectionModel
from src.model_config import ModelConfig


class TestModelInference(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = ModelConfig()
        cls.model_params = cls.config.get_model_params()

        # Build and compile model (not training here)
        cls.model_wrapper = LeukemiaDetectionModel(cls.config)
        cls.model = cls.model_wrapper.build_model()

        # Generate dummy image batch (batch_size, height, width, channels)
        cls.input_shape = cls.model_params["input_shape"]
        cls.dummy_batch = np.random.rand(4, *cls.input_shape).astype(np.float32)

    def test_model_prediction_shape(self):
        """Test if model returns predictions with correct shape."""
        predictions = self.model.predict(self.dummy_batch)

        # Binary classification: output should be (batch_size, 1)
        if self.model_params["num_classes"] == 2:
            self.assertEqual(predictions.shape, (4, 1))
            self.assertTrue(np.all((predictions >= 0) & (predictions <= 1)))

        # Multiclass classification: output should be (batch_size, num_classes)
        else:
            self.assertEqual(predictions.shape, (4, self.model_params["num_classes"]))
            row_sums = np.sum(predictions, axis=1)
            self.assertTrue(np.allclose(row_sums, 1.0, atol=1e-5))

    def test_model_loading(self):
        """Test saving and loading model for inference."""
        save_path = "tests/temp_saved_model"
        self.model_wrapper.save_model(save_path, format='tf')

        # Load model
        self.model_wrapper.load_model(save_path)
        loaded_model = self.model_wrapper.model

        # Ensure prediction still works
        predictions = loaded_model.predict(self.dummy_batch)
        self.assertIsInstance(predictions, np.ndarray)


if __name__ == "__main__":
<<<<<<< HEAD
    unittest.main()
# This will run the unit tests when the script is executed directly
# It is not necessary to include this in the module, as it is intended for testing purposes
=======
    unittest.main()
>>>>>>> e211ebe (commit)
