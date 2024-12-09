from preprocesing_knn import smote_balance, scale_data
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import pandas as pd


def classification(data: pd.DataFrame ,save: bool=False):
    # Balancing
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

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

    # Standardizing the features
    X_train, X_test = scale_data(X_train, X_test)

    # Using two classification models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, max_depth= 20, n_estimators= 300),
        "XGBoost": XGBClassifier(random_state=42, colsample_bytree= 1, learning_rate= 0.1, max_depth= 15, n_estimators= 300, subsample= 0.8)
    }

    # Training and evaluating the models
    results = {}
    for model_name, model in models.items():
        print(f"Training {model_name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert results dictionary to a DataFrame
    for model_name, report in results.items():
        print(f"Results for {model_name}:")
        df_report = pd.DataFrame(report).transpose()
        if save:
            df_report.to_csv(f"results/{model_name}_report.csv")
        print(df_report)
        print("\n")