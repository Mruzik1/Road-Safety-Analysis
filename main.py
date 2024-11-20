from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pandas as pd
from exploratory_analysis import download_dataset, read_data, drop_choosen_attributes
from preprocesing_knn import preprocesing_knn, smote_balance

if __name__ == "__main__":
    # Loading the data
    dst_folder = "./data"
    run_download = False

    if run_download:
        download_dataset(dst_folder)

    df_accident, df_vehicle = read_data(dst_folder, accident_fp="accident_info.csv", vehicle_fp="vehicle_info.csv")
    df_merged = pd.merge(df_vehicle, df_accident, on="Accident_Index", how="inner")
    print(f"Merged size: {len(df_merged)}")

    df_merged = drop_choosen_attributes(df_merged)

    # Preprocessing the data
    df_merged = preprocesing_knn(data=df_merged)

    # Balancing
    X_resampled, y_resampled = smote_balance(data=df_merged, columns_to_use=[
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
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Using two classification models
    models = {
        "Random Forest": RandomForestClassifier(random_state=42, max_depth=5),
        "SVM": SVC(random_state=42, max_iter=1000),
    }

    # Training and evaluating the models
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        results[model_name] = classification_report(y_test, y_pred, output_dict=True)
    
    # Convert results dictionary to a DataFrame
    for model_name, report in results.items():
        print(f"Results for {model_name}:")
        df_report = pd.DataFrame(report).transpose()
        print(df_report)
        print("\n")