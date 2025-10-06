"""
Main script for Customer Churn Analysis and Prediction.

This script orchestrates the entire ML pipeline:
1. Data loading and preprocessing
2. Feature engineering
3. Model training and evaluation
4. Results visualization and reporting
"""

import os
import sys
import argparse


def run_pipeline(data_path, output_dir=None):
    """
    Run the complete churn analysis pipeline.
    
    Args:
        data_path (str): Path to the input data CSV file
        output_dir (str): Directory to save outputs
    """
    # Import dependencies here to allow demo mode without installing them
    import pandas as pd
    import config
    from src.data_preprocessing import DataPreprocessor
    from src.feature_engineering import FeatureEngineer
    from src.model_training import ModelTrainer
    from src.visualization import ChurnVisualizer
    
    print("\n" + "="*60)
    print("Customer Churn Analysis Pipeline")
    print("="*60 + "\n")
    
    # Set output directory
    import config
    output_dir = output_dir or config.OUTPUTS_DIR
    
    # Step 1: Data Preprocessing
    print("Step 1: Data Preprocessing")
    print("-" * 60)
    preprocessor = DataPreprocessor()
    preprocessor.load_data(data_path)
    
    if preprocessor.data is None:
        print("Error: Failed to load data. Exiting.")
        return
    
    preprocessor.handle_missing_values(strategy='mean')
    preprocessor.encode_categorical_features()
    preprocessor.scale_features()
    
    # Save processed data
    preprocessor.save_processed_data()
    
    # Step 2: Split data
    print("\nStep 2: Data Splitting")
    print("-" * 60)
    X_train, X_test, y_train, y_test = preprocessor.split_data()
    
    if X_train is None:
        print("Error: Failed to split data. Exiting.")
        return
    
    # Step 3: Feature Engineering
    print("\nStep 3: Feature Engineering")
    print("-" * 60)
    engineer = FeatureEngineer(preprocessor.data)
    
    # Get feature importance
    importance_df = engineer.get_feature_importance(X_train, y_train)
    
    # Step 4: Model Training
    print("\nStep 4: Model Training and Evaluation")
    print("-" * 60)
    trainer = ModelTrainer()
    trainer.initialize_models()
    trainer.train_all_models(X_train, y_train, X_test, y_test)
    
    # Step 5: Visualization
    print("\nStep 5: Generating Visualizations")
    print("-" * 60)
    visualizer = ChurnVisualizer()
    
    # Plot churn distribution
    plot_path = os.path.join(config.PLOTS_DIR, 'churn_distribution.png')
    visualizer.plot_churn_distribution(y_train, save_path=plot_path)
    
    # Plot feature importance
    plot_path = os.path.join(config.PLOTS_DIR, 'feature_importance.png')
    visualizer.plot_feature_importance(importance_df, save_path=plot_path)
    
    # Plot model comparison
    results_df = trainer.display_results_summary()
    if results_df is not None:
        plot_path = os.path.join(config.PLOTS_DIR, 'model_comparison.png')
        visualizer.plot_model_comparison(results_df, save_path=plot_path)
    
    # Step 6: Save best model
    print("\nStep 6: Saving Best Model")
    print("-" * 60)
    if trainer.best_model is not None:
        best_model_name = list(trainer.results.keys())[0]
        trainer.save_model(best_model_name, 'best_model.pkl')
    
    print("\n" + "="*60)
    print("Pipeline Completed Successfully!")
    print("="*60)
    print(f"\nOutputs saved to: {config.OUTPUTS_DIR}")
    print(f"Models saved to: {config.MODELS_DIR}")
    print(f"Processed data saved to: {config.PROCESSED_DATA_DIR}")


def main():
    """Main function with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description='Customer Churn Analysis and Prediction Pipeline'
    )
    
    parser.add_argument(
        '--data',
        type=str,
        help='Path to the input data CSV file',
        default=None
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Directory to save outputs',
        default=None
    )
    
    parser.add_argument(
        '--demo',
        action='store_true',
        help='Run in demo mode (shows pipeline structure without data)'
    )
    
    args = parser.parse_args()
    
    if args.demo or args.data is None:
        print("\n" + "="*60)
        print("Customer Churn Analysis - Demo Mode")
        print("="*60)
        print("\nProject Structure:")
        print("  data/raw/        - Place your raw data here")
        print("  data/processed/  - Processed data will be saved here")
        print("  notebooks/       - Jupyter notebooks for analysis")
        print("  src/             - Source code modules")
        print("  models/          - Trained models will be saved here")
        print("  outputs/         - Analysis outputs and visualizations")
        print("\nTo run the pipeline:")
        print("  python main.py --data data/raw/your_data.csv")
        print("\nModules available:")
        print("  - data_preprocessing.py: Data cleaning and preprocessing")
        print("  - feature_engineering.py: Feature creation and selection")
        print("  - model_training.py: Model training and evaluation")
        print("  - visualization.py: Data and model visualization")
        print("\nFor Jupyter notebook analysis:")
        print("  jupyter notebook notebooks/exploratory_analysis.ipynb")
        return
    
    # Run the pipeline
    run_pipeline(args.data, args.output)


if __name__ == "__main__":
    main()
