# Models Directory

This directory stores trained machine learning models.

## Model Files

Trained models will be saved here:

- `logistic_regression_model.pkl`: Logistic Regression model
- `random_forest_model.pkl`: Random Forest model
- `xgboost_model.pkl`: XGBoost model
- `lightgbm_model.pkl`: LightGBM model
- `best_model.pkl`: Best performing model

## Model Metadata

Along with model files, consider storing:
- Model hyperparameters
- Training metrics
- Feature importance scores
- Model version information

## Usage

Load a saved model:
```python
import joblib
model = joblib.load('models/random_forest_model.pkl')
predictions = model.predict(X_test)
```

## Note

- Model files are gitignored by default (can be large)
- Document model versions and performance metrics
- Consider using model versioning tools for production
