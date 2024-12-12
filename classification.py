from preprocesing_knn import smote_balance, scale_data
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd


def classification(data: pd.DataFrame ,save: bool=False):
    """
    Function to perform data classification using Random Forest and XGBoost methods.

    Parameters:
    - data: pd.DataFrame
        The input dataset containing numerical features.
    - save: bool
        Flag to save classification reports to CSV files.

    Returns:
        - None

    Steps:
    1. Balance the data using the SMOTE method.
    2. Split the data into training and testing sets.
    3. Scale the data.
    4. Train Random Forest and XGBoost models on the training set.
    5. Evaluate the models on the test set and generate classification reports.
    6. Save the reports to CSV files (if the save flag is set to True).
    """
    X_resampled, y_resampled = smote_balance(data=data, columns_to_use=[
    "Age_of_Vehicle",
    "Driver_IMD_Decile",
    "Engine_Capacity_.CC.",
    "Speed_limit",
    "Road_Surface_Conditions",
    "Weather_Conditions",
    "Road_Type",
    "Urban_or_Rural_Area",
    ], key_column=["Accident_Severity"])

    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    X_train, X_test = scale_data(X_train, X_test)

    models = {
        "Random Forest": RandomForestClassifier(random_state=42, max_depth= 20, n_estimators= 300),
        "XGBoost": XGBClassifier(random_state=42, colsample_bytree= 1, learning_rate= 0.1, max_depth= 15, n_estimators= 300, subsample= 0.8)
    }

    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = classification_report(y_test, y_pred, output_dict=True)
    
    for model_name, report in results.items():
        print(f"Results for {model_name}:")
        df_report = pd.DataFrame(report).transpose()
        if save:
            df_report.to_csv(f"results/{model_name}_report.csv")
        print(df_report)
        print("\n")