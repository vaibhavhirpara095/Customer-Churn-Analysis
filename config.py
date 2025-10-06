"""
Configuration settings for the Customer Churn Analysis project.
"""

import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
PLOTS_DIR = os.path.join(OUTPUTS_DIR, 'plots')
REPORTS_DIR = os.path.join(OUTPUTS_DIR, 'reports')

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering
NUMERICAL_FEATURES = []
CATEGORICAL_FEATURES = []
TARGET_COLUMN = 'Churn'

# Model selection
MODELS_TO_TRAIN = [
    'logistic_regression',
    'random_forest',
    'xgboost',
    'lightgbm'
]

# Hyperparameter tuning
USE_GRID_SEARCH = False
N_ITER_RANDOM_SEARCH = 50

# Evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']

# Visualization
PLOT_STYLE = 'seaborn'
FIGURE_SIZE = (10, 6)
DPI = 100

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
