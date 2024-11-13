import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Any

from acquire_data import read_data, download_dataset


def find_nan_attributes(
    df: pd.DataFrame, 
    nan_threshold: float = 0.5
) -> List[Any]:
    """ Find columns with high percentage of nan values """
    nan_percentage = df.isna().mean()
    return nan_percentage[nan_percentage > nan_threshold].index.tolist()


def drop_choosen_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """ Drop choosen attributes """
    # drop columns with high nan %
    attrs2drop = find_nan_attributes(df)
    
    # drop redundant attributes and columns with unique identifiers
    # attrs2drop += ["Unnamed: 0_x", "Unnamed: 0_y", "Accident_Index", "Year_x", "Year_y"]
    attrs2drop += ["Accident_Index", "Year_x", "Year_y"]

    # drop location-specific info
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

    # drop other unnecessary attributes 
    attrs2drop += [
        "Was_Vehicle_Left_Hand_Drive",       # highly low percentage of values other than "No" - ~0.05%
        "model"                              # too many unique values / limited analytical impact
    ]

    return df.drop(attrs2drop, axis=1)


def save_plot(plot, folder: str, filename: str):
    """ Save a plot to a specific folder with a specific name """
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
    """ Draw a correlation matrix for the dataframe """
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
    """ Plot a values distribution (histogram) for a specific attribute """
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
    attributes: List[str],
    save: bool = False,
    folder: str = "./visualizations"
):
    """ Plot box plots for a list of attributes """
    for attribute in attributes:
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df[attribute],showfliers=False)
        plt.title(f'Boxplot of {attribute}')
        plt.xlabel('Attributes')
        plt.ylabel('Value')
            
        if save:
            # attributes_str = "_".join(attributes)
            filename = f"{attribute}_boxplot.png"
            save_plot(plt, folder, filename)
        else:
            plt.show()

def scatter_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    save: bool = False,
    folder: str = "./visualizations"
):
    """ Plot a scatter plot for two attributes """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x, y=y)
    plt.title(f'Scatter plot of {x} and {y}')
    plt.xlabel(x)
    plt.ylabel(y)
    
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

    df_accident, df_vehicle = read_data(dst_folder, accident_fp="accident_info.csv", vehicle_fp="vehicle_info.csv")
    df_merged = pd.merge(df_vehicle, df_accident, on="Accident_Index", how="inner")
    print(f"Merged size: {len(df_merged)}")

    df_merged = drop_choosen_attributes(df_merged)
    print(df_merged.head())

    draw_correlation_matrix(df_merged, save=True)
    plot_distribution(df_merged, attribute="Age_Band_of_Driver", save=True)
    plot_distribution(df_merged, attribute="Accident_Severity", save=True)
    box_plot(df_merged, attributes=['Age_of_Vehicle', 'Engine_Capacity_.CC.', 'Driver_IMD_Decile'], save=True)
    scatter_plot(df_merged, x='Vehicle_Reference', y='Number_of_Vehicles', save=True)
    scatter_plot(df_merged, x='Number_of_Vehicles', y='Number_of_Casualties', save=True)