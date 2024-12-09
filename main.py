import pandas as pd
from exploratory_analysis import download_dataset, read_data, drop_choosen_attributes
from preprocesing_knn import preprocesing_knn
from classification import classification

if __name__ == "__main__":
    # # Loading the data
    # dst_folder = "./data"
    # run_download = False

    # if run_download:
    #     download_dataset(dst_folder)

    # df_accident, df_vehicle = read_data(dst_folder, accident_fp="accident_info.csv", vehicle_fp="vehicle_info.csv")
    # df_merged = pd.merge(df_vehicle, df_accident, on="Accident_Index", how="inner")
    # print(f"Merged size: {len(df_merged)}")

    # df_merged = drop_choosen_attributes(df_merged)

    # # Preprocessing the data
    # df_merged = preprocesing_knn(data=df_merged)
    df_merged = pd.read_csv('data/merged_data_num_imputed_knn.csv')

    classification(df_merged, save=True)

    