"""
Data preprocessing module for Leukemia Detection project.
Handles data loading, splitting, and preprocessing operations.
"""

import os
import shutil
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
from PIL import Image
import json
from config.model_config import ModelConfig
from src.utils import setup_logging, set_random_seeds


class DataPreprocessor:
    """Data preprocessing class for leukemia detection."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize data preprocessor.
        
        Args:
            config: Model configuration object
        """
        self.config = config
        self.data_config = config.get_data_params()
        self.logger = logging.getLogger(__name__)
        
        # Set random seeds for reproducibility
        set_random_seeds(self.data_config['random_seed'])
    
    def load_and_organize_data(self, source_path: str, target_path: str) -> Dict:
        """
        Load and organize data from source to target directory structure.
        
        Args:
            source_path: Path to source dataset
            target_path: Path to target processed directory
            
        Returns:
            Dictionary containing data organization statistics
        """
        self.logger.info(f"Loading data from {source_path}")
        
        # Create target directory structure
        self._create_directory_structure(target_path)
        
        # Load image paths and labels
        image_paths, labels = self._load_image_paths_and_labels(source_path)
        
        # Create train/validation/test splits
        train_paths, test_paths, train_labels, test_labels = train_test_split(
            image_paths, labels, 
            test_size=self.data_config['test_split'],
            random_state=self.data_config['random_seed'],
            stratify=labels
        )
        
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            train_paths, train_labels,
            test_size=self.data_config['validation_split'],
            random_state=self.data_config['random_seed'],
            stratify=train_labels
        )
        
        # Copy files to appropriate directories
        self._copy_files_to_directories(train_paths, train_labels, target_path, 'train')
        self._copy_files_to_directories(val_paths, val_labels, target_path, 'validation')
        self._copy_files_to_directories(test_paths, test_labels, target_path, 'test')
        
        # Create statistics
        stats = {
            'total_images': len(image_paths),
            'train_images': len(train_paths),
            'validation_images': len(val_paths),
            'test_images': len(test_paths),
            'healthy_images': sum(1 for label in labels if label == 0),
            'leukemia_images': sum(1 for label in labels if label == 1),
            'train_healthy': sum(1 for label in train_labels if label == 0),
            'train_leukemia': sum(1 for label in train_labels if label == 1),
            'val_healthy': sum(1 for label in val_labels if label == 0),
            'val_leukemia': sum(1 for label in val_labels if label == 1),
            'test_healthy': sum(1 for label in test_labels if label == 0),
            'test_leukemia': sum(1 for label in test_labels if label == 1)
        }
        
        # Save statistics
        stats_path = Path(target_path) / 'data_stats.json'
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info(f"Data organization complete. Statistics saved to {stats_path}")
        return stats
    
    def _create_directory_structure(self, target_path: str):
        """Create directory structure for organized data."""
        directories = [
            'train/healthy', 'train/leukemia',
            'validation/healthy', 'validation/leukemia',
            'test/healthy', 'test/leukemia'
        ]
        
        for directory in directories:
            Path(target_path, directory).mkdir(parents=True, exist_ok=True)
    
    def _load_image_paths_and_labels(self, source_path: str) -> Tuple[List[str], List[int]]:
        """
        Load image paths and labels from source directory.
        
        Args:
            source_path: Path to source dataset
            
        Returns:
            Tuple of (image_paths, labels)
        """
        image_paths = []
        labels = []
        
        source_dir = Path(source_path)
        
        # Look for image files in the source directory
        for img_file in source_dir.glob('*.jpg'):
            image_paths.append(str(img_file))
            
            # Determine label based on filename convention
            # Assuming healthy images contain 'all' and leukemia images contain other patterns
            filename = img_file.name.lower()
            if 'all' in filename:
                labels.append(1)  # Leukemia (ALL - Acute Lymphoblastic Leukemia)
            else:
                labels.append(0)  # Healthy
        
        # Also check for PNG files
        for img_file in source_dir.glob('*.png'):
            image_paths.append(str(img_file))
            filename = img_file.name.lower()
            if 'all' in filename:
                labels.append(1)  # Leukemia
            else:
                labels.append(0)  # Healthy
        
        self.logger.info(f"Found {len(image_paths)} images")
        return image_paths, labels
    
    def _copy_files_to_directories(self, paths: List[str], labels: List[int], 
                                  target_path: str, split_name: str):
        """Copy files to appropriate directories based on labels."""
        for path, label in zip(paths, labels):
            filename = Path(path).name
            if label == 0:  # Healthy
                target_dir = Path(target_path, split_name, 'healthy')
            else:  # Leukemia
                target_dir = Path(target_path, split_name, 'leukemia')
            
            target_file = target_dir / filename
            shutil.copy2(path, target_file)
    
    def preprocess_image(self, image_path: str) -> np.ndarray:
        """
        Preprocess a single image.
        
        Args:
            image_path: Path to the image
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize image
        target_size = self.data_config['image_size']
        image = cv2.resize(image, target_size)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        return image
    
    def create_tf_dataset(self, data_path: str, split: str, 
                         augment: bool = False) -> tf.data.Dataset:
        """
        Create TensorFlow dataset from processed data.
        
        Args:
            data_path: Path to processed data directory
            split: Data split ('train', 'validation', 'test')
            augment: Whether to apply data augmentation
            
        Returns:
            TensorFlow dataset
        """
        split_path = Path(data_path) / split
        
        # Create dataset using tf.keras.utils.image_dataset_from_directory
        dataset = tf.keras.utils.image_dataset_from_directory(
            str(split_path),
            labels='inferred',
            label_mode='binary',
            class_names=['healthy', 'leukemia'],
            batch_size=self.data_config['batch_size'],
            image_size=self.data_config['image_size'],
            shuffle=True if split == 'train' else False,
            seed=self.data_config['random_seed']
        )
        
        # Configure dataset for performance
        dataset = dataset.cache()
        
        if split == 'train':
            dataset = dataset.shuffle(buffer_size=1000)
        
        # Apply preprocessing
        dataset = dataset.map(
            lambda x, y: (tf.cast(x, tf.float32) / 255.0, y),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation if specified
        if augment and split == 'train':
            dataset = self._apply_augmentation(dataset)
        
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def _apply_augmentation(self, dataset: tf.data.Dataset) -> tf.data.Dataset:
        """Apply data augmentation to dataset."""
        augmentation_config = self.config.augmentation_config
        
        # Create augmentation layers
        augmentation_layers = [
            tf.keras.layers.RandomRotation(
                factor=augmentation_config['rotation_range'] / 360.0
            ),
            tf.keras.layers.RandomWidth(
                factor=augmentation_config['width_shift_range']
            ),
            tf.keras.layers.RandomHeight(
                factor=augmentation_config['height_shift_range']
            ),
            tf.keras.layers.RandomZoom(
                height_factor=augmentation_config['zoom_range'],
                width_factor=augmentation_config['zoom_range']
            ),
        ]
        
        if augmentation_config['horizontal_flip']:
            augmentation_layers.append(tf.keras.layers.RandomFlip('horizontal'))
        
        if augmentation_config['vertical_flip']:
            augmentation_layers.append(tf.keras.layers.RandomFlip('vertical'))
        
        # Apply augmentation
        def augment_data(image, label):
            for layer in augmentation_layers:
                image = layer(image)
            return image, label
        
        dataset = dataset.map(augment_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        return dataset
    
    def get_class_distribution(self, data_path: str) -> Dict:
        """
        Get class distribution statistics.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Dictionary containing class distribution
        """
        distribution = {}
        
        for split in ['train', 'validation', 'test']:
            split_path = Path(data_path) / split
            
            healthy_count = len(list((split_path / 'healthy').glob('*')))
            leukemia_count = len(list((split_path / 'leukemia').glob('*')))
            
            distribution[split] = {
                'healthy': healthy_count,
                'leukemia': leukemia_count,
                'total': healthy_count + leukemia_count,
                'healthy_ratio': healthy_count / (healthy_count + leukemia_count) if (healthy_count + leukemia_count) > 0 else 0,
                'leukemia_ratio': leukemia_count / (healthy_count + leukemia_count) if (healthy_count + leukemia_count) > 0 else 0
            }
        
        return distribution
    
    def validate_data_quality(self, data_path: str) -> Dict:
        """
        Validate data quality and report any issues.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Dictionary containing validation results
        """
        validation_results = {
            'valid_images': 0,
            'invalid_images': 0,
            'corrupted_files': [],
            'size_issues': [],
            'format_issues': []
        }
        
        for split in ['train', 'validation', 'test']:
            for class_name in ['healthy', 'leukemia']:
                class_path = Path(data_path) / split / class_name
                
                for img_file in class_path.glob('*'):
                    try:
                        # Try to load image
                        image = Image.open(img_file)
                        image.verify()  # Verify image integrity
                        
                        # Check image format
                        if image.format not in ['JPEG', 'PNG', 'JPG']:
                            validation_results['format_issues'].append(str(img_file))
                        
                        # Check image size
                        if image.size[0] < 50 or image.size[1] < 50:
                            validation_results['size_issues'].append(str(img_file))
                        
                        validation_results['valid_images'] += 1
                        
                    except Exception as e:
                        validation_results['invalid_images'] += 1
                        validation_results['corrupted_files'].append(str(img_file))
                        self.logger.warning(f"Corrupted image found: {img_file} - {str(e)}")
        
        return validation_results
    
    def create_data_summary(self, data_path: str) -> Dict:
        """
        Create comprehensive data summary.
        
        Args:
            data_path: Path to processed data directory
            
        Returns:
            Dictionary containing data summary
        """
        summary = {
            'data_path': data_path,
            'image_size': self.data_config['image_size'],
            'batch_size': self.data_config['batch_size'],
            'class_distribution': self.get_class_distribution(data_path),
            'validation_results': self.validate_data_quality(data_path),
            'preprocessing_config': self.data_config
        }
        
        return summary
    
    def save_preprocessed_sample(self, dataset: tf.data.Dataset, 
                                save_path: str, num_samples: int = 16):
        """
        Save sample preprocessed images for visual inspection.
        
        Args:
            dataset: TensorFlow dataset
            save_path: Path to save samples
            num_samples: Number of samples to save
        """
        import matplotlib.pyplot as plt
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(16, 16))
        
        class_names = ['Healthy', 'Leukemia']
        
        for i, (image, label) in enumerate(dataset.take(num_samples)):
            if i >= num_samples:
                break
                
            plt.subplot(4, 4, i + 1)
            plt.imshow(image[0].numpy())  # First image in batch
            plt.title(f"Class: {class_names[int(label[0].numpy())]}")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Sample images saved to {save_path}")


def main():
    """Main function for data preprocessing."""
    # Setup logging
    setup_logging()
    
    # Load configuration
    config = ModelConfig()
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Create directories
    config.create_directories()
    
    # Load and organize data
    data_params = config.get_data_params()
    stats = preprocessor.load_and_organize_data(
        data_params['dataset_path'],
        data_params['processed_path']
    )
    
    print("Data preprocessing statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Create data summary
    summary = preprocessor.create_data_summary(data_params['processed_path'])
    
    # Save summary
    summary_path = Path(data_params['processed_path']) / 'data_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"\nData summary saved to: {summary_path}")
    
    # Create sample datasets for testing
    train_dataset = preprocessor.create_tf_dataset(
        data_params['processed_path'], 'train', augment=True
    )
    
    val_dataset = preprocessor.create_tf_dataset(
        data_params['processed_path'], 'validation', augment=False
    )
    
    # Save sample images
    sample_path = Path(config.paths_config['plots_path']) / 'preprocessed_samples.png'
    preprocessor.save_preprocessed_sample(train_dataset, str(sample_path))
    
    print(f"Sample preprocessed images saved to: {sample_path}")


if __name__ == "__main__":
    main()

# This code is part of the Leukemia Detection project and is designed to preprocess data for training machine learning models.
# It includes functionality for loading, organizing, and preprocessing images, as well as creating TensorFlow datasets.
