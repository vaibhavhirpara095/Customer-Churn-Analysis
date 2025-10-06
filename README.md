# Customer Churn Analysis and Prediction

This project aims to predict customer churn for a telecommunications company using data analytics and machine learning. The goal is to identify at-risk customers and recommend strategies to improve retention.

## Project Overview

Customer churn is a critical metric for telecommunications companies, as retaining existing customers is often more cost-effective than acquiring new ones. This project uses machine learning to predict which customers are likely to churn and provides actionable insights to improve customer retention.

## Project Structure

```
Customer-Churn-Analysis/
│
├── data/
│   ├── raw/                 # Original, immutable data
│   └── processed/           # Cleaned and preprocessed data
│
├── notebooks/
│   └── exploratory_analysis.ipynb    # Jupyter notebooks for exploration
│
├── src/
│   ├── data_preprocessing.py         # Data cleaning and preprocessing
│   ├── feature_engineering.py        # Feature creation and selection
│   ├── model_training.py             # Model training and evaluation
│   └── visualization.py              # Visualization utilities
│
├── models/                  # Trained model files (gitignored)
│
├── outputs/
│   ├── plots/              # Generated visualizations
│   └── reports/            # Analysis reports
│
├── requirements.txt         # Project dependencies
├── config.py               # Configuration settings
└── README.md               # Project documentation
```

## Features

- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of customer data to identify patterns and trends
- **Data Preprocessing**: Handling missing values, encoding categorical variables, and feature scaling
- **Feature Engineering**: Creating meaningful features from raw data
- **Model Training**: Training multiple machine learning models (Logistic Regression, Random Forest, XGBoost, etc.)
- **Model Evaluation**: Comparing models using accuracy, precision, recall, F1-score, and ROC-AUC
- **Feature Importance**: Identifying key factors that contribute to customer churn
- **Predictions**: Generating churn predictions for new customers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/vaibhavhirpara095/Customer-Churn-Analysis.git
cd Customer-Churn-Analysis
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. **Data Preprocessing**: 
```bash
python src/data_preprocessing.py
```

2. **Feature Engineering**:
```bash
python src/feature_engineering.py
```

3. **Model Training**:
```bash
python src/model_training.py
```

4. **Exploratory Analysis**:
```bash
jupyter notebook notebooks/exploratory_analysis.ipynb
```

## Machine Learning Models

The project implements and compares several machine learning algorithms:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines (SVM)

## Key Metrics

- **Accuracy**: Overall correctness of the model
- **Precision**: Proportion of predicted churners who actually churned
- **Recall**: Proportion of actual churners correctly identified
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Results

Detailed results and model performance metrics will be documented in the `outputs/reports/` directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Contact

For questions or feedback, please contact Vaibhav Hirpara.
