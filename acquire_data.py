import kagglehub
import shutil
import os

import pandas as pd


def download_dataset(dst_folder):
    """ Download dataset """
    path = kagglehub.dataset_download("tsiaras/uk-road-safety-accidents-and-vehicles")
    for f in os.listdir(path):
        try:
            shutil.move(os.path.join(path, f), dst_folder)
        except shutil.Error as e:
            print(f"WARNING: {e}")
    shutil.rmtree(path)
    print(f"Path to dataset files: {dst_folder}")
    return dst_folder


def read_data(data_path, accident_fp="Accident_Information.csv", vehicle_fp="Vehicle_Information.csv"):
    """ Read data from accidents and vehicles """
    with open(os.path.join(data_path, accident_fp), "r") as f:
        df_accident = pd.read_csv(f, dtype='unicode')
    with open(os.path.join(data_path, vehicle_fp), "r") as f:
        df_vehicle = pd.read_csv(f, dtype='unicode')

    return df_accident, df_vehicle


if __name__ == "__main__":
    dst_folder = "./data"
    run_download = False

    if run_download:
        download_dataset(dst_folder)
    df_accident, df_vehicle = read_data(dst_folder)

    with open(f"{dst_folder}/accident_info.csv", "w") as f:
        df_accident.iloc[:int(len(df_accident) // 10)].to_csv(f)
    with open(f"{dst_folder}/vehicle_info.csv", "w") as f:
        df_vehicle.iloc[:int(len(df_vehicle) // 10)].to_csv(f)