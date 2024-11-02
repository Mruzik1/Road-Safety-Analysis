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


def read_data(data_path):
    """ Read data from accidents and vehicles """
    with open(os.path.join(data_path, "Accident_Information.csv"), "r") as f:
        df_accident = pd.read_csv(f, dtype='unicode')
    with open(os.path.join(data_path, "Vehicle_Information.csv"), "r") as f:
        df_vehicle = pd.read_csv(f, dtype='unicode')

    return df_accident, df_vehicle

if __name__ == "__main__":
    dst_folder = "./data"
    run_download = False

    if run_download:
        download_dataset(dst_folder)
    read_data(dst_folder)