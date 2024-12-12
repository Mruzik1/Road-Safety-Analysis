import pandas as pd
from exploratory_analysis import download_dataset, read_data, drop_choosen_attributes
from preprocesing_knn import preprocesing_knn
from classification import classification
from anomaly_detection import anomaly_detection
import time


if __name__ == "__main__":
    start_time = time.time()

    # Loading the data
    dst_folder = "./data"
    run_download = False

    if run_download:
        download_dataset(dst_folder)

    df_accident, df_vehicle = read_data(dst_folder, accident_fp="Accident_Information.csv", vehicle_fp="Vehicle_Information.csv")    
    df_merged = pd.merge(df_vehicle, df_accident, on="Accident_Index", how="inner")
    print(df_merged.head())
    print(f"Merged size: {len(df_merged)}")
    
    df_merged = drop_choosen_attributes(df_merged)

    # Preprocessing the data
    df_merged = preprocesing_knn(data=df_merged)
    df_merged.to_csv(f"{dst_folder}/info_procesed.csv", index=False)
    # df_merged = pd.read_csv('data/merged_data_num_imputed_knn.csv')

    classification(df_merged, save=True)

    anomaly_detection(df_merged, save=True, show=False)

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Finished in {int(hours)}:{int(minutes)}:{int(seconds)}")