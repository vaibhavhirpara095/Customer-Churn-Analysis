# Processed Data Directory

This directory stores cleaned and preprocessed data ready for model training.

## Files

Processed data files will be saved here after running the preprocessing pipeline:

- `processed_data.csv`: Main cleaned dataset
- `train_data.csv`: Training dataset
- `test_data.csv`: Testing dataset
- `feature_engineered_data.csv`: Dataset with engineered features

## Processing Steps

Data in this directory has typically undergone:

1. **Missing Value Handling**: Imputation or removal
2. **Encoding**: Categorical variables converted to numerical
3. **Scaling**: Numerical features normalized/standardized
4. **Feature Engineering**: New features created from existing ones
5. **Data Splitting**: Split into train/test sets

## Note

- Files in this directory are gitignored by default
- Regenerate processed data when raw data is updated
- Ensure preprocessing pipeline is documented
