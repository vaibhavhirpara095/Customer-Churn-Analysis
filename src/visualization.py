"""
Visualization module for Customer Churn Analysis.

This module handles:
- Data visualization
- Model performance visualization
- Feature importance plots
- Distribution plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix
import config
import os


class ChurnVisualizer:
    """Class for creating visualizations for churn analysis."""
    
    def __init__(self):
        """Initialize the visualizer."""
        plt.style.use(config.PLOT_STYLE)
        self.figure_size = config.FIGURE_SIZE
        self.dpi = config.DPI
        
    def plot_churn_distribution(self, y, save_path=None):
        """
        Plot the distribution of churn labels.
        
        Args:
            y (pd.Series): Target variable
            save_path (str): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        churn_counts = y.value_counts()
        ax.bar(churn_counts.index, churn_counts.values, color=['green', 'red'], alpha=0.7)
        ax.set_xlabel('Churn Status')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Customer Churn')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['No Churn', 'Churn'])
        
        # Add percentage labels
        total = len(y)
        for i, v in enumerate(churn_counts.values):
            ax.text(i, v, f'{v} ({v/total*100:.1f}%)', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_feature_distributions(self, data, features, target_col, save_path=None):
        """
        Plot distributions of features split by target variable.
        
        Args:
            data (pd.DataFrame): Data
            features (list): List of features to plot
            target_col (str): Target column name
            save_path (str): Path to save the plot
        """
        n_features = len(features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, n_rows * 4), dpi=self.dpi)
        axes = axes.flatten() if n_features > 1 else [axes]
        
        for idx, feature in enumerate(features):
            if feature not in data.columns:
                continue
                
            for churn_val in data[target_col].unique():
                subset = data[data[target_col] == churn_val][feature]
                axes[idx].hist(subset, alpha=0.5, label=f'Churn={churn_val}', bins=30)
            
            axes[idx].set_xlabel(feature)
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'Distribution of {feature}')
            axes[idx].legend()
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_correlation_matrix(self, data, save_path=None):
        """
        Plot correlation matrix heatmap.
        
        Args:
            data (pd.DataFrame): Data
            save_path (str): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        
        correlation = data.corr()
        sns.heatmap(correlation, annot=False, cmap='coolwarm', center=0,
                    square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
        
        ax.set_title('Feature Correlation Matrix')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_feature_importance(self, importance_df, top_n=15, save_path=None):
        """
        Plot feature importance.
        
        Args:
            importance_df (pd.DataFrame): DataFrame with 'Feature' and 'Importance' columns
            top_n (int): Number of top features to plot
            save_path (str): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        top_features = importance_df.head(top_n)
        ax.barh(range(len(top_features)), top_features['Importance'], color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        """
        Plot confusion matrix.
        
        Args:
            y_true (array-like): True labels
            y_pred (array-like): Predicted labels
            labels (list): Label names
            save_path (str): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)
        
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True, ax=ax)
        
        if labels:
            ax.set_xticklabels(labels)
            ax.set_yticklabels(labels)
        
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """
        Plot ROC curve.
        
        Args:
            y_true (array-like): True labels
            y_pred_proba (array-like): Predicted probabilities
            save_path (str): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_model_comparison(self, results_df, metric='accuracy', save_path=None):
        """
        Plot comparison of multiple models.
        
        Args:
            results_df (pd.DataFrame): DataFrame with model results
            metric (str): Metric to compare
            save_path (str): Path to save the plot
        """
        fig, ax = plt.subplots(figsize=self.figure_size, dpi=self.dpi)
        
        sorted_df = results_df.sort_values(metric, ascending=True)
        ax.barh(range(len(sorted_df)), sorted_df[metric], color='steelblue')
        ax.set_yticks(range(len(sorted_df)))
        ax.set_yticklabels(sorted_df.index)
        ax.set_xlabel(metric.capitalize())
        ax.set_title(f'Model Comparison - {metric.capitalize()}')
        ax.set_xlim([0, 1])
        
        # Add value labels
        for i, v in enumerate(sorted_df[metric]):
            ax.text(v, i, f'{v:.4f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to {save_path}")
        
        plt.show()


def main():
    """Main function for visualization."""
    print("="*50)
    print("Customer Churn Visualization Utilities")
    print("="*50)
    
    visualizer = ChurnVisualizer()
    
    print("\nVisualization utilities ready!")
    print("Available visualization functions:")
    print("  - plot_churn_distribution()")
    print("  - plot_feature_distributions()")
    print("  - plot_correlation_matrix()")
    print("  - plot_feature_importance()")
    print("  - plot_confusion_matrix()")
    print("  - plot_roc_curve()")
    print("  - plot_model_comparison()")


if __name__ == "__main__":
    main()
