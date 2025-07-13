"""
Unit tests for model architecture and configuration in the Leukemia Detection project.
"""

import unittest
import tensorflow as tf
from src.model_architecture import LeukemiaDetectionModel, CustomCNN
from src.model_config import ModelConfig


class TestModelArchitecture(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.config = ModelConfig()
        cls.model_params = cls.config.get_model_params()

    def test_transfer_learning_model_builds(self):
        """Test if transfer learning model builds and compiles successfully."""
        model_wrapper = LeukemiaDetectionModel(self.config)
        model = model_wrapper.build_model()

        self.assertIsInstance(model, tf.keras.Model)
        self.assertGreater(len(model.layers), 0)
        self.assertEqual(model.input_shape[1:], self.model_params["input_shape"])

        # Check output layer
        if self.model_params["num_classes"] == 2:
            self.assertEqual(model.output_shape[-1], 1)
        else:
            self.assertEqual(model.output_shape[-1], self.model_params["num_classes"])

    def test_custom_cnn_model_builds(self):
        """Test if custom CNN builds with correct input/output dimensions."""
        cnn = CustomCNN(
            input_shape=self.model_params["input_shape"],
            num_classes=self.model_params["num_classes"]
        )
        model = cnn.build_model()

        self.assertIsInstance(model, tf.keras.Model)
        self.assertEqual(model.input_shape[1:], self.model_params["input_shape"])

        if self.model_params["num_classes"] == 2:
            self.assertEqual(model.output_shape[-1], 1)
        else:
            self.assertEqual(model.output_shape[-1], self.model_params["num_classes"])

    def test_parameter_count(self):
        """Test if parameter count function returns valid counts."""
        model_wrapper = LeukemiaDetectionModel(self.config)
        model_wrapper.build_model()
        params = model_wrapper.count_parameters()

        self.assertIn("trainable_parameters", params)
        self.assertIn("non_trainable_parameters", params)
        self.assertIn("total_parameters", params)

        self.assertGreater(params["total_parameters"], 0)


if __name__ == "__main__":
    unittest.main()