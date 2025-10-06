"""
Customer Churn Analysis - Source Package

This package contains modules for:
- Data preprocessing
- Feature engineering
- Model training and evaluation
- Visualization
"""

__version__ = '1.0.0'
__author__ = 'Vaibhav Hirpara'

from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer
from .model_training import ModelTrainer
from .visualization import ChurnVisualizer

__all__ = [
    'DataPreprocessor',
    'FeatureEngineer',
    'ModelTrainer',
    'ChurnVisualizer'
]
