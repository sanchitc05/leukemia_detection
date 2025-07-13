"""
Model configuration module for Leukemia Detection project.
Contains model architecture definitions and hyperparameters.
"""

import yaml
import os
from pathlib import Path

class ModelConfig:
    """Configuration class for model parameters."""
    
    def __init__(self, config_path=None):
        if config_path is None:
            config_path = Path(__file__).parent / "config.yaml"
        
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
    
    @property
    def data_config(self):
        """Return data configuration."""
        return self.config['data']
    
    @property
    def model_config(self):
        """Return model configuration."""
        return self.config['model']
    
    @property
    def training_config(self):
        """Return training configuration."""
        return self.config['training']
    
    @property
    def augmentation_config(self):
        """Return augmentation configuration."""
        return self.config['augmentation']
    
    @property
    def evaluation_config(self):
        """Return evaluation configuration."""
        return self.config['evaluation']
    
    @property
    def paths_config(self):
        """Return paths configuration."""
        return self.config['paths']
    
    @property
    def inference_config(self):
        """Return inference configuration."""
        return self.config['inference']
    
    def get_model_params(self):
        """Get model parameters as dictionary."""
        model_params = {
            'architecture': self.model_config['architecture'],
            'input_shape': tuple(self.model_config['input_shape']),
            'num_classes': self.model_config['num_classes'],
            'dropout_rate': self.model_config['dropout_rate'],
            'pretrained_weights': self.model_config['pretrained_weights'],
            'fine_tune_layers': self.model_config['fine_tune_layers']
        }
        return model_params
    
    def get_training_params(self):
        """Get training parameters as dictionary."""
        training_params = {
            'epochs': self.training_config['epochs'],
            'initial_learning_rate': self.training_config['initial_learning_rate'],
            'fine_tune_learning_rate': self.training_config['fine_tune_learning_rate'],
            'early_stopping_patience': self.training_config['early_stopping_patience'],
            'reduce_lr_patience': self.training_config['reduce_lr_patience'],
            'reduce_lr_factor': self.training_config['reduce_lr_factor'],
            'optimizer': self.training_config['optimizer'],
            'loss_function': self.training_config['loss_function'],
            'metrics': self.training_config['metrics']
        }
        return training_params
    
    def get_data_params(self):
        """Get data parameters as dictionary."""
        data_params = {
            'dataset_path': self.data_config['dataset_path'],
            'processed_path': self.data_config['processed_path'],
            'image_size': tuple(self.data_config['image_size']),
            'batch_size': self.data_config['batch_size'],
            'validation_split': self.data_config['validation_split'],
            'test_split': self.data_config['test_split'],
            'random_seed': self.data_config['random_seed']
        }
        return data_params
    
    def create_directories(self):
        """Create necessary directories based on configuration."""
        paths = self.paths_config
        directories = [
            paths['model_save_path'],
            paths['checkpoint_path'],
            paths['export_path'],
            paths['results_path'],
            paths['logs_path'],
            paths['plots_path'],
            self.data_config['processed_path'] + 'train/healthy/',
            self.data_config['processed_path'] + 'train/leukemia/',
            self.data_config['processed_path'] + 'validation/healthy/',
            self.data_config['processed_path'] + 'validation/leukemia/',
            self.data_config['processed_path'] + 'test/healthy/',
            self.data_config['processed_path'] + 'test/leukemia/',
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def update_config(self, section, key, value):
        """Update configuration value."""
        if section in self.config and key in self.config[section]:
            self.config[section][key] = value
        else:
            raise KeyError(f"Section '{section}' or key '{key}' not found in configuration")
    
    def save_config(self, path=None):
        """Save configuration to file."""
        if path is None:
            path = Path(__file__).parent / "config.yaml"
        
        with open(path, 'w') as file:
            yaml.dump(self.config, file, default_flow_style=False)


# Global configuration instance
config = ModelConfig()

# Convenience functions
def get_model_config():
    """Get model configuration."""
    return config.model_config

def get_training_config():
    """Get training configuration."""
    return config.training_config

def get_data_config():
    """Get data configuration."""
    return config.data_config