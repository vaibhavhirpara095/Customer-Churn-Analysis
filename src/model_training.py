"""
Model training and evaluation module for Customer Churn Analysis.

This module handles:
- Training multiple ML models
- Model evaluation and comparison
- Hyperparameter tuning
- Model persistence
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve
)
from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
import xgboost as xgb
import lightgbm as lgb
import joblib
import os
import config


class ModelTrainer:
    """Class for training and evaluating churn prediction models."""
    
    def __init__(self):
        """Initialize the model trainer."""
        self.models = {}
        self.results = {}
        self.best_model = None
        
    def initialize_models(self):
        """Initialize all machine learning models."""
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=config.RANDOM_STATE,
                max_iter=1000
            ),
            'decision_tree': DecisionTreeClassifier(
                random_state=config.RANDOM_STATE
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=config.RANDOM_STATE,
                n_jobs=-1
            ),
            'gradient_boosting': GradientBoostingClassifier(
                random_state=config.RANDOM_STATE
            ),
            'xgboost': xgb.XGBClassifier(
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                eval_metric='logloss'
            ),
            'lightgbm': lgb.LGBMClassifier(
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1
            ),
            'svm': SVC(
                random_state=config.RANDOM_STATE,
                probability=True
            )
        }
        print(f"Initialized {len(self.models)} models")
        
    def train_model(self, model_name, X_train, y_train):
        """
        Train a specific model.
        
        Args:
            model_name (str): Name of the model to train
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found")
            return
        
        print(f"\nTraining {model_name}...")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        print(f"{model_name} training completed")
        
    def evaluate_model(self, model_name, X_test, y_test):
        """
        Evaluate a trained model.
        
        Args:
            model_name (str): Name of the model to evaluate
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
            
        Returns:
            dict: Evaluation metrics
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found")
            return None
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        
        self.results[model_name] = metrics
        
        print(f"\n{model_name} Results:")
        for metric, value in metrics.items():
            print(f"{metric.capitalize()}: {value:.4f}")
        
        return metrics
    
    def cross_validate_model(self, model_name, X, y, cv=None):
        """
        Perform cross-validation on a model.
        
        Args:
            model_name (str): Name of the model
            X (pd.DataFrame): Features
            y (pd.Series): Labels
            cv (int): Number of cross-validation folds
            
        Returns:
            dict: Cross-validation scores
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found")
            return None
        
        cv = cv or config.CV_FOLDS
        model = self.models[model_name]
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
        
        print(f"\n{model_name} Cross-Validation (CV={cv}):")
        print(f"Mean Accuracy: {cv_results['mean_score']:.4f} (+/- {cv_results['std_score']:.4f})")
        
        return cv_results
    
    def train_all_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate all models.
        
        Args:
            X_train (pd.DataFrame): Training features
            y_train (pd.Series): Training labels
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
        """
        print("\n" + "="*50)
        print("Training All Models")
        print("="*50)
        
        for model_name in self.models.keys():
            self.train_model(model_name, X_train, y_train)
            self.evaluate_model(model_name, X_test, y_test)
        
        self.display_results_summary()
        
    def display_results_summary(self):
        """Display summary of all model results."""
        if not self.results:
            print("No results to display")
            return
        
        results_df = pd.DataFrame(self.results).T
        results_df = results_df.sort_values('accuracy', ascending=False)
        
        print("\n" + "="*50)
        print("Model Comparison Summary")
        print("="*50)
        print(results_df)
        
        # Identify best model
        best_model_name = results_df.index[0]
        self.best_model = self.models[best_model_name]
        print(f"\nBest Model: {best_model_name}")
        print(f"Accuracy: {results_df.loc[best_model_name, 'accuracy']:.4f}")
        
        return results_df
    
    def save_model(self, model_name, filename=None):
        """
        Save a trained model to disk.
        
        Args:
            model_name (str): Name of the model to save
            filename (str): Output filename
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found")
            return
        
        filename = filename or f"{model_name}_model.pkl"
        filepath = os.path.join(config.MODELS_DIR, filename)
        
        joblib.dump(self.models[model_name], filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath):
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the model file
            
        Returns:
            object: Loaded model
        """
        try:
            model = joblib.load(filepath)
            print(f"Model loaded from {filepath}")
            return model
        except FileNotFoundError:
            print(f"Error: Model file not found at {filepath}")
            return None
    
    def get_classification_report(self, model_name, X_test, y_test):
        """
        Generate detailed classification report.
        
        Args:
            model_name (str): Name of the model
            X_test (pd.DataFrame): Test features
            y_test (pd.Series): Test labels
        """
        if model_name not in self.models:
            print(f"Error: Model '{model_name}' not found")
            return
        
        model = self.models[model_name]
        y_pred = model.predict(X_test)
        
        print(f"\n{model_name} - Classification Report:")
        print(classification_report(y_test, y_pred))
        
        print(f"\n{model_name} - Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))


def main():
    """Main function for model training."""
    print("="*50)
    print("Customer Churn Model Training")
    print("="*50)
    
    trainer = ModelTrainer()
    trainer.initialize_models()
    
    print("\nModel training pipeline ready!")
    print("Available models:")
    for model_name in trainer.models.keys():
        print(f"  - {model_name}")
    
    print("\nPlease prepare your training data and call:")
    print("trainer.train_all_models(X_train, y_train, X_test, y_test)")


if __name__ == "__main__":
    main()
