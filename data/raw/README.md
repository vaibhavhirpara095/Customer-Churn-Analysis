# Raw Data Directory

Place your raw, unprocessed customer data files here.

## Expected Data Format

The customer churn dataset should include:

- **Customer Information**: Customer ID, demographics (age, gender, location)
- **Account Information**: Tenure, contract type, payment method
- **Service Usage**: Phone service, internet service, streaming services, etc.
- **Billing Information**: Monthly charges, total charges
- **Target Variable**: Churn (Yes/No or 1/0)

## Sample Data Structure

```csv
CustomerID,Gender,SeniorCitizen,Partner,Dependents,Tenure,PhoneService,InternetService,MonthlyCharges,TotalCharges,Churn
001,Male,0,Yes,No,12,Yes,Fiber optic,75.5,905.0,No
002,Female,1,No,Yes,24,Yes,DSL,55.2,1324.8,Yes
...
```

## Data Sources

Common sources for customer churn datasets:
- Kaggle: Telco Customer Churn dataset
- Company internal databases
- Publicly available telecommunications datasets

## Note

- Files in this directory are gitignored by default
- Do not commit sensitive customer data
- Ensure data is anonymized before use
