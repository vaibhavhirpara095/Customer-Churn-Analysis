"""
Data preprocessing module for Customer Churn Analysis.

This module handles:
- Loading raw data
- Handling missing values
- Encoding categorical variables
- Feature scaling
- Data splitting
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import config
import os


class DataPreprocessor:
    """Class for preprocessing customer churn data."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.data = None
        
    def load_data(self, filepath):
        """
        Load data from CSV file.
        
        Args:
            filepath (str): Path to the CSV file
            
        Returns:
            pd.DataFrame: Loaded data
        """
        try:
            self.data = pd.read_csv(filepath)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"Error: File not found at {filepath}")
            return None
        
    def handle_missing_values(self, strategy='mean'):
        """
        Handle missing values in the dataset.
        
        Args:
            strategy (str): Strategy for handling missing values ('mean', 'median', 'mode', 'drop')
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        missing_count = self.data.isnull().sum()
        if missing_count.sum() > 0:
            print(f"Missing values found:\n{missing_count[missing_count > 0]}")
            
            if strategy == 'drop':
                self.data = self.data.dropna()
            elif strategy == 'mean':
                for col in self.data.select_dtypes(include=[np.number]).columns:
                    self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif strategy == 'median':
                for col in self.data.select_dtypes(include=[np.number]).columns:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
            elif strategy == 'mode':
                for col in self.data.columns:
                    self.data[col].fillna(self.data[col].mode()[0], inplace=True)
                    
            print(f"Missing values handled using {strategy} strategy")
        else:
            print("No missing values found")
            
    def encode_categorical_features(self, columns=None):
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            columns (list): List of columns to encode. If None, encode all object columns.
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        if columns is None:
            columns = self.data.select_dtypes(include=['object']).columns.tolist()
            
        for col in columns:
            if col in self.data.columns:
                le = LabelEncoder()
                self.data[col] = le.fit_transform(self.data[col].astype(str))
                self.label_encoders[col] = le
                print(f"Encoded column: {col}")
                
    def scale_features(self, columns=None, exclude_target=True):
        """
        Scale numerical features using StandardScaler.
        
        Args:
            columns (list): List of columns to scale. If None, scale all numerical columns.
            exclude_target (bool): Whether to exclude target column from scaling
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        if columns is None:
            columns = self.data.select_dtypes(include=[np.number]).columns.tolist()
            
        if exclude_target and config.TARGET_COLUMN in columns:
            columns.remove(config.TARGET_COLUMN)
            
        self.data[columns] = self.scaler.fit_transform(self.data[columns])
        print(f"Scaled {len(columns)} features")
        
    def split_data(self, test_size=None, random_state=None):
        """
        Split data into training and testing sets.
        
        Args:
            test_size (float): Proportion of data for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test)
        """
        if self.data is None:
            print("Error: No data loaded")
            return None
        
        if config.TARGET_COLUMN not in self.data.columns:
            print(f"Error: Target column '{config.TARGET_COLUMN}' not found")
            return None
        
        test_size = test_size or config.TEST_SIZE
        random_state = random_state or config.RANDOM_STATE
        
        X = self.data.drop(columns=[config.TARGET_COLUMN])
        y = self.data[config.TARGET_COLUMN]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Data split - Train: {X_train.shape}, Test: {X_test.shape}")
        return X_train, X_test, y_train, y_test
    
    def save_processed_data(self, filename='processed_data.csv'):
        """
        Save processed data to CSV file.
        
        Args:
            filename (str): Name of the output file
        """
        if self.data is None:
            print("Error: No data loaded")
            return
        
        output_path = os.path.join(config.PROCESSED_DATA_DIR, filename)
        self.data.to_csv(output_path, index=False)
        print(f"Processed data saved to {output_path}")


def main():
    """Main function for data preprocessing."""
    print("="*50)
    print("Customer Churn Data Preprocessing")
    print("="*50)
    
    preprocessor = DataPreprocessor()
    
    # Example usage (uncomment when data is available)
    # data_path = os.path.join(config.RAW_DATA_DIR, 'customer_data.csv')
    # preprocessor.load_data(data_path)
    # preprocessor.handle_missing_values()
    # preprocessor.encode_categorical_features()
    # preprocessor.scale_features()
    # preprocessor.save_processed_data()
    
    print("\nPreprocessing pipeline ready!")
    print("Please add your data file to the data/raw/ directory")


if __name__ == "__main__":
    main()
