"""
Feature engineering module for Customer Churn Analysis.

This module handles:
- Creating new features from existing data
- Feature selection
- Feature importance analysis
"""

import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
import config


class FeatureEngineer:
    """Class for feature engineering on customer churn data."""
    
    def __init__(self, data=None):
        """
        Initialize the feature engineer.
        
        Args:
            data (pd.DataFrame): Input data
        """
        self.data = data
        self.selected_features = None
        
    def create_interaction_features(self, feature_pairs):
        """
        Create interaction features between pairs of features.
        
        Args:
            feature_pairs (list): List of tuples containing feature pairs
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        for feat1, feat2 in feature_pairs:
            if feat1 in self.data.columns and feat2 in self.data.columns:
                new_feature = f"{feat1}_{feat2}_interaction"
                self.data[new_feature] = self.data[feat1] * self.data[feat2]
                print(f"Created interaction feature: {new_feature}")
                
    def create_ratio_features(self, numerator, denominator, feature_name):
        """
        Create ratio features.
        
        Args:
            numerator (str): Column name for numerator
            denominator (str): Column name for denominator
            feature_name (str): Name for the new feature
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        if numerator in self.data.columns and denominator in self.data.columns:
            # Avoid division by zero
            self.data[feature_name] = self.data[numerator] / (self.data[denominator] + 1e-10)
            print(f"Created ratio feature: {feature_name}")
            
    def create_binned_features(self, column, bins, labels=None):
        """
        Create binned features from continuous variables.
        
        Args:
            column (str): Column name to bin
            bins (int or list): Number of bins or bin edges
            labels (list): Labels for bins
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        if column in self.data.columns:
            new_column = f"{column}_binned"
            self.data[new_column] = pd.cut(self.data[column], bins=bins, labels=labels)
            print(f"Created binned feature: {new_column}")
            
    def select_k_best_features(self, X, y, k=10, score_func=f_classif):
        """
        Select top k features using statistical tests.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            k (int): Number of top features to select
            score_func: Scoring function (f_classif or mutual_info_classif)
            
        Returns:
            list: Selected feature names
        """
        selector = SelectKBest(score_func=score_func, k=k)
        selector.fit(X, y)
        
        # Get feature scores
        scores = pd.DataFrame({
            'Feature': X.columns,
            'Score': selector.scores_
        })
        scores = scores.sort_values('Score', ascending=False)
        
        self.selected_features = scores.head(k)['Feature'].tolist()
        print(f"\nTop {k} features selected:")
        print(scores.head(k))
        
        return self.selected_features
    
    def get_feature_importance(self, X, y, n_estimators=100):
        """
        Get feature importance using Random Forest.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            n_estimators (int): Number of trees in the forest
            
        Returns:
            pd.DataFrame: Feature importance scores
        """
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=config.RANDOM_STATE,
            n_jobs=-1
        )
        rf.fit(X, y)
        
        importance_df = pd.DataFrame({
            'Feature': X.columns,
            'Importance': rf.feature_importances_
        })
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        print("\nFeature Importance (Random Forest):")
        print(importance_df.head(15))
        
        return importance_df
    
    def remove_low_variance_features(self, threshold=0.01):
        """
        Remove features with low variance.
        
        Args:
            threshold (float): Variance threshold
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        numerical_cols = self.data.select_dtypes(include=[np.number]).columns
        variances = self.data[numerical_cols].var()
        
        low_variance_cols = variances[variances < threshold].index.tolist()
        
        if low_variance_cols:
            print(f"Removing {len(low_variance_cols)} low variance features:")
            print(low_variance_cols)
            self.data = self.data.drop(columns=low_variance_cols)
        else:
            print("No low variance features found")
            
    def handle_outliers(self, columns, method='iqr', threshold=1.5):
        """
        Handle outliers in specified columns.
        
        Args:
            columns (list): List of column names
            method (str): Method for handling outliers ('iqr', 'zscore')
            threshold (float): Threshold for outlier detection
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        for col in columns:
            if col not in self.data.columns:
                continue
                
            if method == 'iqr':
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                outliers = (self.data[col] < lower_bound) | (self.data[col] > upper_bound)
                outlier_count = outliers.sum()
                
                # Cap outliers
                self.data.loc[self.data[col] < lower_bound, col] = lower_bound
                self.data.loc[self.data[col] > upper_bound, col] = upper_bound
                
                print(f"Handled {outlier_count} outliers in {col}")


def main():
    """Main function for feature engineering."""
    print("="*50)
    print("Customer Churn Feature Engineering")
    print("="*50)
    
    print("\nFeature engineering utilities ready!")
    print("This module provides:")
    print("- Interaction features")
    print("- Ratio features")
    print("- Binned features")
    print("- Feature selection")
    print("- Feature importance analysis")
    print("- Outlier handling")


if __name__ == "__main__":
    main()
