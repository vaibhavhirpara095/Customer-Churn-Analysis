# import all require libraries  

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix, roc_curve
)

# load dataset

df = pd.read_csv("Telco_Customer_Churn_Dataset.csv")

# check dataset informations

print("check first five rows",df.head())

print("chech last five rows",df.tail())

print("dataset info",df.info())

print("dataset size",df.shape)

# check missing values

print("Missing values:\n",df.isnull().sum())

# drop customer id column 

df = df.drop('customerID',axis=1)
print("dataset info",df.info())

# convert total charge to numeric

df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
print("check first five rows",df.head())

# fill missing total charges with median

df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
print("check first five rows",df.head())

# labelencoder

cat_cols = df.select_dtypes(include='object').columns
le = LabelEncoder()
for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# seprate features and target

X = df.drop('Churn', axis=1)
y = df['Churn']

# split dataset into train and test sets (80-20 split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training set shape:", X_train.shape)
print("Testing set shape:", X_test.shape)

churn_rate = df['Churn'].mean()
print(f"Overall churn rate: {churn_rate*100:.2f}%")

# Visulization graphs

fig, axes = plt.subplots(1, 3, figsize=(18,5))
sns.countplot(x='gender', hue='Churn', data=df, ax=axes[0])
sns.countplot(x='Partner', hue='Churn', data=df, ax=axes[1])
sns.countplot(x='Dependents', hue='Churn', data=df, ax=axes[2])
axes[0].set_title('Churn by Gender')
axes[1].set_title('Churn by Partner Status')
axes[2].set_title('Churn by Dependent Status')
plt.show()

plt.figure(figsize=(10,5))
sns.histplot(data=df, x='tenure', hue='Churn', bins=30, kde=True)
plt.title('Tenure Distribution vs Churn')
plt.show()

fig, axes = plt.subplots(1,2, figsize=(18,5))
sns.countplot(x='Contract', hue='Churn', data=df, ax=axes[0])
sns.countplot(x='PaymentMethod', hue='Churn', data=df, ax=axes[1])
axes[0].set_title('Churn by Contract Type')
axes[1].set_title('Churn by Payment Method')
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title('Monthly Charges vs Churn')
plt.show()

plt.figure(figsize=(8,5))
sns.boxplot(x='Churn', y='TotalCharges', data=df)
plt.title('Total Charges vs Churn')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
corr = df.corr(numeric_only=True)  # only numeric columns
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Feature Correlation Heatmap')
plt.show()

plt.figure(figsize=(8,5))
sns.countplot(x='InternetService', hue='Churn', data=df)
plt.title('Churn by Internet Service')
plt.show()

# Segment customers based on tenure, monthly charges, contract type
df['TenureGroup'] = pd.cut(df['tenure'], bins=[0,12,24,48,60,72], labels=['0-12','12-24','24-48','48-60','60-72'])
df['MonthlyChargeGroup'] = pd.qcut(df['MonthlyCharges'], q=4, labels=['Low','Medium','High','Very High'])

# Churn rate by segment
tenure_churn = df.groupby('TenureGroup')['Churn'].mean()
monthly_churn = df.groupby('MonthlyChargeGroup')['Churn'].mean()
contract_churn = df.groupby('Contract')['Churn'].mean()

print("Churn by Tenure Group:\n", tenure_churn)
print("Churn by Monthly Charge Group:\n", monthly_churn)
print("Churn by Contract Type:\n", contract_churn)

# Visualize churn in segments
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
tenure_churn.plot(kind='bar', color='skyblue', title='Churn by Tenure Group')
plt.subplot(1,3,2)
monthly_churn.plot(kind='bar', color='salmon', title='Churn by Monthly Charges Group')
plt.subplot(1,3,3)
contract_churn.plot(kind='bar', color='lightgreen', title='Churn by Contract Type')
plt.show()

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# Evaluation function
def evaluate_model(y_true, y_pred):
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall:", recall_score(y_true, y_pred))
    print("F1 Score:", f1_score(y_true, y_pred))
    
print("Logistic Regression Performance:")
evaluate_model(y_test, y_pred_lr)
print("\nRandom Forest Performance:")
evaluate_model(y_test, y_pred_rf)

# Feature Importance for Random Forest
importances = rf.feature_importances_
feature_names = X.columns
feat_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Feature Importance:\n", feat_importance.head(10))

# ROC curve
y_prob_rf = rf.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob_rf)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0,1],[0,1], color='red', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Random Forest')
plt.legend()
plt.show()


high_risk_customers = df[(df['Churn']==1) & (df['MonthlyCharges'] > 70)]
print("High risk, high-value customers count:", high_risk_customers.shape[0])

print("""
Business Recommendations:
1. Offer special retention plans or discounts to high-risk, high-value customers.
2. Provide loyalty programs or contract benefits for long-tenure customers.
3. Focus marketing campaigns on customers with month-to-month contracts.
4. Monitor customers with high monthly charges for potential churn.
5. Improve engagement and communication with customers who have multiple dependents.
""")
