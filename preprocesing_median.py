import pandas as pd
from sklearn.impute import SimpleImputer

# Load the data
data = pd.read_csv("data/merged_data.csv")
print("Before imputation:")
print(data.isnull().sum())

frequent_imputer = SimpleImputer(strategy="most_frequent")
data["Propulsion_Code"] = frequent_imputer.fit_transform(data[["Propulsion_Code"]]).ravel()
data = data.dropna(subset=["Time"])

# Define columns with missing numerical values
numerical_columns = [
    'Age_of_Vehicle', 
    'Driver_IMD_Decile', 
    'Engine_Capacity_.CC.', 
    'Vehicle_Location.Restricted_Lane', 
    'Did_Police_Officer_Attend_Scene_of_Accident', 
    'Pedestrian_Crossing-Human_Control', 
    'Pedestrian_Crossing-Physical_Facilities'
]

# Impute using median for robustness to outliers
for column in numerical_columns:
    if column in data.columns:
        median_value = data[column].median()
        data[column] = data[column].fillna(median_value)

data.to_csv("data/merged_data_num_imputed_median.csv", index=False)
# Verify that there are no more missing values in the selected columns
missing_after_imputation = data.isnull().sum()

print("\nAfter imputation:\n")
print(missing_after_imputation)
print(len(data))