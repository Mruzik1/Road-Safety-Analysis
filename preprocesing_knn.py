from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from typing import List, Any
from sklearn.preprocessing import LabelEncoder

def scale_data(
        train_data: pd.DataFrame,
        test_data: pd.DataFrame = None,
        inverse: bool = False
    ) -> pd.DataFrame:
    scaler = StandardScaler()
    if test_data is None:
        train_data = scaler.fit_transform(train_data)
        if inverse:
            return train_data, scaler
        return train_data
    else:
        train_data = scaler.fit_transform(train_data)
        test_data = scaler.transform(test_data)
        if inverse:
            return train_data, test_data, scaler
        return train_data, test_data


def preprocesing_knn(
        path_to_merged_data: str = None,
        data: pd.DataFrame = None,
        output_path: str = None
    ):
    # Load the data
    if data is None and path_to_merged_data is not None:
        data = pd.read_csv(path_to_merged_data)
    elif data is None and path_to_merged_data is None:
        return None

    print("Before imputation:")
    print(data.isnull().sum())

    frequent_imputer = SimpleImputer(strategy="most_frequent")
    data["Propulsion_Code"] = frequent_imputer.fit_transform(data[["Propulsion_Code"]]).ravel()
    data = data.dropna(subset=["Time"])

    numerical_columns = [
        'Age_of_Vehicle', 
        'Driver_IMD_Decile', 
        'Engine_Capacity_.CC.', 
        'Vehicle_Location.Restricted_Lane', 
        'Did_Police_Officer_Attend_Scene_of_Accident', 
        'Pedestrian_Crossing-Human_Control', 
        'Pedestrian_Crossing-Physical_Facilities'
    ]

    numerical_data = data[numerical_columns]

    scaled_data, scaler = scale_data(numerical_data, inverse=True)

    knn_imputer = KNNImputer(n_neighbors=5)
    imputed_data = knn_imputer.fit_transform(scaled_data)

    imputed_data = scaler.inverse_transform(imputed_data)
    data[numerical_columns] = imputed_data

    print(data.isnull().sum())
    print(len(data))

    if output_path:
        data.to_csv(output_path, index=False)
    return data


def smote_balance(
        path_to_preprocesed_data: str=None,
        data: pd.DataFrame=None,
        columns_to_use: List[str]=[
            "Age_of_Vehicle",
            "Driver_IMD_Decile",
            "Engine_Capacity_.CC.",
            "Speed_limit",
            "Road_Surface_Conditions",
            "Weather_Conditions",
            "Road_Type",
            "Urban_or_Rural_Area",
        ],
        key_column: List[str] = ["Accident_Severity"]
    ):
    if data is None and path_to_preprocesed_data is  None:
        return None, None
    elif data is None and path_to_preprocesed_data is not None:
        data = pd.read_csv(path_to_preprocesed_data)
    data_to_use = data[columns_to_use + key_column].copy()
    print(data_to_use["Accident_Severity"].value_counts())


    for col in data_to_use.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data_to_use[col] = le.fit_transform(data_to_use[col])
    
    smote = SMOTE()
    X = data_to_use.drop("Accident_Severity", axis=1).copy()
    y = data_to_use["Accident_Severity"].copy()
    X_smote, y_smote = smote.fit_resample(X, y)

    print(f'{y_smote.value_counts()}')

    return X_smote,y_smote