import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any

from acquire_data import read_data, download_dataset

def find_nan_attributes(
    df: pd.DataFrame, 
    nan_threshold: float = 0.4
) -> List[Any]:
    """
    Find columns with high percentage of nan values.
    
    Parameters:
    - df: pd.DataFrame
        The input dataframe.
    - nan_threshold: float
        The threshold percentage of nan values to consider a column as having high nan values.
    
    Returns:
    - List[Any]
        List of column names with high percentage of nan values.
    """
    nan_percentage = df.isna().mean()
    return nan_percentage[nan_percentage > nan_threshold].index.tolist()

def drop_choosen_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop chosen attributes from the dataframe.
    
    Parameters:
    - df: pd.DataFrame
        The input dataframe.
    
    Returns:
    - pd.DataFrame
        The dataframe with chosen attributes dropped.
    """
    attrs2drop = find_nan_attributes(df)
    attrs2drop += ["Accident_Index", "Year_x", "Year_y"]
    attrs2drop += [
        "1st_Road_Number", 
        "2nd_Road_Number", 
        "Latitude", 
        "Longitude", 
        "Location_Easting_OSGR",
        "Location_Northing_OSGR",
        "LSOA_of_Accident_Location",
        "Local_Authority_(District)",
        "Local_Authority_(Highway)"
    ]
    attrs2drop += [
        "Was_Vehicle_Left_Hand_Drive",
        "model"
    ]
    return df.drop(attrs2drop, axis=1)

def save_plot(plot, folder: str, filename: str):
    """
    Save a plot to a specific folder with a specific name.
    
    Parameters:
    - plot: matplotlib.pyplot
        The plot to save.
    - folder: str
        The folder to save the plot in.
    - filename: str
        The name of the file to save the plot as.
    """
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)
    plot_path = os.path.join(folder, filename)
    plot.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

def draw_correlation_matrix(
    df: pd.DataFrame, 
    save: bool = False, 
    folder: str = "./visualizations", 
    filename: str = "correlation_matrix.png"
):
    """
    Draw a correlation matrix for the dataframe.
    
    Parameters:
    - df: pd.DataFrame
        The input dataframe.
    - save: bool
        Whether to save the plot.
    - folder: str
        The folder to save the plot in.
    - filename: str
        The name of the file to save the plot as.
    """
    df_numeric = df.apply(pd.to_numeric, errors='coerce')
    df_numeric = df_numeric.dropna(axis=1, how='all')
    corr = df_numeric.corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap of Numerical Attributes")
    
    if save:
        save_plot(plt, folder, filename)
    else:
        plt.show()

def plot_distribution(
    df: pd.DataFrame, 
    attribute: str, 
    save: bool = False, 
    folder: str = "./visualizations", 
    filename: str = None
):
    """
    Plot a values distribution (histogram) for a specific attribute.
    
    Parameters:
    - df: pd.DataFrame
        The input dataframe.
    - attribute: str
        The attribute to plot the distribution for.
    - save: bool
        Whether to save the plot.
    - folder: str
        The folder to save the plot in.
    - filename: str
        The name of the file to save the plot as.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(df[attribute].dropna())
    plt.title(f'Distribution of {attribute}')
    plt.xlabel(attribute)
    plt.ylabel('Frequency')
    
    if save:
        if filename is None:
            filename = f"{attribute}_distribution.png"
        save_plot(plt, folder, filename)
    else:
        plt.show()

def box_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    name: str,
    xlabel: str,
    ylabel: str,
    save: bool = False,
    folder: str = "./visualizations"
):
    """
    Plot box plots for a list of attributes.
    
    Parameters:
    - df: pd.DataFrame
        The input dataframe.
    - x: str
        The attribute for the x-axis.
    - y: str
        The attribute for the y-axis.
    - name: str
        The name of the plot.
    - xlabel: str
        The label for the x-axis.
    - ylabel: str
        The label for the y-axis.
    - save: bool
        Whether to save the plot.
    - folder: str
        The folder to save the plot in.
    """
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df,x=x,y=y,showfliers=False)
    plt.title(f'{name} boxplot')
    plt.xlabel(f'{xlabel}')
    plt.ylabel(f'{ylabel}')
        
    if save:
        filename = f"{name}_boxplot.png"
        save_plot(plt, folder, filename)
    else:
        plt.show()

def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    xlabel: str,
    ylabel: str,
    dop: List[List[Any]] = None,
    save: bool = False,
    folder: str = "./visualizations"
):
    """
    Plot a scatter plot for two attributes.
    
    Parameters:
    - df: pd.DataFrame
        The input dataframe.
    - x: str
        The attribute for the x-axis.
    - y: str
        The attribute for the y-axis.
    - xlabel: str
        The label for the x-axis.
    - ylabel: str
        The label for the y-axis.
    - dop: List[List[Any]]
        Optional parameter for y-axis ticks.
    - save: bool
        Whether to save the plot.
    - folder: str
        The folder to save the plot in.
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(df[x], df[y])
    plt.title(f'Scatter plot of {xlabel} and {ylabel}')
    plt.xlabel(xlabel)
    if dop:
        plt.yticks(dop[0], dop[1])
    plt.ylabel(ylabel)
    
    if save:
        filename = f"{x}_and_{y}_scatterplot.png"
        save_plot(plt, folder, filename)
    else:
        plt.show()

if __name__ == "__main__":
    dst_folder = "./data"
    run_download = False

    if run_download:
        download_dataset(dst_folder)

    df_accident, df_vehicle = read_data(dst_folder, accident_fp="Accident_Information.csv", vehicle_fp="Vehicle_Information.csv")
    df_merged = pd.merge(df_vehicle, df_accident, on="Accident_Index", how="inner")
    print(f"Merged size: {len(df_merged)}")

    df_merged = drop_choosen_attributes(df_merged)
    print(df_merged.head())

    df_merged['Accident_Severity'] = df_merged['Accident_Severity'].astype('category')
    df_merged['Speed_limit'] = df_merged['Speed_limit'].astype(float)
    df_merged['Engine_Capacity_.CC.'] = df_merged['Engine_Capacity_.CC.'].astype(float)
    df_merged['Age_of_Vehicle'] = df_merged['Age_of_Vehicle'].astype(float)
    df_merged.to_csv("./data/merged_data.csv", index=False)
    severity_mapping = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
    df_merged['Severity_Num'] = df_merged['Accident_Severity'].map(severity_mapping)

    draw_correlation_matrix(df_merged, save=True)
    plot_distribution(df_merged, attribute="Age_Band_of_Driver", save=True)
    plot_distribution(df_merged, attribute="Accident_Severity", save=True)
    box_plot(df_merged, x='Accident_Severity', y='Speed_limit', name='Speed Limit by Accident Severity', xlabel='Accident Severity', ylabel='Speed Limit (mph)', save=True)
    box_plot(df_merged, x='Age_of_Vehicle', y='Engine_Capacity_.CC.', name='Engine Capacity by Age of Vehicle', xlabel='Age of Vehicle (years)', ylabel='Engine Capacity (CC)', save=True)
    # scatter_plot(df_merged, x='Engine_Capacity_.CC.', y='Severity_Num', xlabel='Engine Capacity (CC)', ylabel='Accident Severity (Numerical)', dop=[[1, 2, 3], ['Slight', 'Serious', 'Fatal']], save=True)
    # scatter_plot(df_merged, x='Age_of_Vehicle', y='Severity_Num',xlabel='Age of Vehicle (years)', ylabel='Accident Severity (Numerical)', dop=[[1, 2, 3], ['Slight', 'Serious', 'Fatal']], save=True)