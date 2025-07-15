"""
DataLoader module for Leukemia Detection project.

Handles:
- Splitting raw dataset into train/val/test folders
- Loading TensorFlow datasets via DataPreprocessor
"""

from pathlib import Path
import shutil
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np

from src.model_config import ModelConfig
from src.data_preprocessing import DataPreprocessor


class DataLoader:
    """Class to manage dataset preparation and loading."""

    def __init__(self, dataset_path, processed_path, image_size, seed=42):
        self.dataset_path = Path(dataset_path)
        self.processed_path = Path(processed_path)
        self.image_size = image_size
        self.seed = seed

    def prepare_and_split_data(self, val_split=0.2, test_split=0.1):
        """
        Splits raw dataset into train, validation, and test sets.
        Args:
            val_split (float): Fraction of training data to be used as validation.
            test_split (float): Fraction of total data to be used as test.
        """
        class_dirs = [d for d in self.dataset_path.iterdir() if d.is_dir()]
        
        for class_dir in class_dirs:
            images = list(class_dir.glob("*"))
            train_val, test = train_test_split(images, test_size=test_split, random_state=self.seed)
            train, val = train_test_split(
                train_val,
                test_size=val_split / (1 - test_split),
                random_state=self.seed
            )

            self._copy_files(train, class_dir.name, "train")
            self._copy_files(val, class_dir.name, "validation")
            self._copy_files(test, class_dir.name, "test")

    def _copy_files(self, files, class_name, split_type):
        """Copies image files to appropriate subdirectories under processed_path."""
        target_dir = self.processed_path / split_type / class_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for f in files:
            shutil.copy(f, target_dir)

    def load_datasets(self):
        """
        Loads train/val/test datasets using DataPreprocessor and returns them.
        Returns:
            Tuple: (train_ds, val_ds, test_ds, class_names, y_test)
        """
        config = ModelConfig()
        preprocessor = DataPreprocessor(config)

        # Organize data structure
        preprocessor.load_and_organize_data(str(self.dataset_path), str(self.processed_path))

        # Create TensorFlow datasets
        train_ds = preprocessor.create_tf_dataset(str(self.processed_path), 'train', augment=True)
        val_ds = preprocessor.create_tf_dataset(str(self.processed_path), 'validation', augment=False)
        test_ds = preprocessor.create_tf_dataset(str(self.processed_path), 'test', augment=False)

        class_names = ['healthy', 'leukemia']

        # Extract test labels
        y_test = [int(label.numpy()[0]) for _, label in test_ds.unbatch()]

        return train_ds, val_ds, test_ds, class_names, np.array(y_test)
