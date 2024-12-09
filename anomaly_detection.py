# Reload the dataset and necessary libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocesing_knn import scale_data
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis

os.environ['OMP_NUM_THREADS'] = '1'


def anomaly_detection(data: pd.DataFrame, n_clusters: int=100, threshold_percentile: int=99, save: bool=False):

    # Step 1: Preprocess the data - Select numerical columns and scale them
    numerical_data = data.select_dtypes(include=[np.number]).dropna()
    # print(numerical_data.head())
    scaled_data, scaler = scale_data(numerical_data, inverse=True)
    # print(scaled_data)
    # Calculate variance for each numerical column
    variances = numerical_data.var()

    # Sort columns by variance in descending order
    sorted_variances = variances.sort_values(ascending=False)
    
    # Step 2: Perform clustering using k-means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    # print("GOOD")
    # Step 3: Calculate Mahalanobis distances for each cluster
    def calculate_mahalanobis(cluster_points, point, cov_inv):
        mean_cluster = np.mean(cluster_points, axis=0)
        return mahalanobis(point, mean_cluster, cov_inv)

    mahalanobis_distances = []
    for cluster in range(n_clusters):
        cluster_points = scaled_data[labels == cluster]
        cov_matrix = np.cov(cluster_points, rowvar=False)
        cov_inv = np.linalg.pinv(cov_matrix)
        for point in cluster_points:
            dist = calculate_mahalanobis(cluster_points, point, cov_inv)
            mahalanobis_distances.append(dist)

    # Step 4: Identify anomalies using a threshold (e.g., top 1% of Mahalanobis distances)
    threshold = np.percentile(mahalanobis_distances, threshold_percentile)
    anomalies = [i for i, dist in enumerate(mahalanobis_distances) if dist > threshold]

    # Step 5: Visualize anomalies relative to a single feature

    single_feature_index = numerical_data.columns.get_loc(sorted_variances.index[0])  # Index of the feature to plot (e.g., first feature in scaled_data)
    single_feature = scaler.inverse_transform(scaled_data)[:, single_feature_index]
    single_feature_name = numerical_data.columns[single_feature_index]

    # Create a density plot of the feature (Normal Points)

    plt.figure(figsize=(12, 6))
    plt.scatter(
        single_feature[range(len(single_feature))],
        np.zeros(len(single_feature)),
        c=["red" if i in anomalies else "blue" for i in range(len(single_feature))],
        alpha=0.6,
        label="Data Points"
    )

    # Add density plot for normal points
    sns.kdeplot(
        x=single_feature[[i for i in range(len(single_feature)) if i not in anomalies]],
        color="blue",
        fill=True,
        alpha=0.3,
        label="Density (Normal Points)"
    )

    # Add title and labels
    plt.title(f"Anomaly Detection Relative to {single_feature_name}", fontsize=16)
    plt.xlabel(single_feature_name, fontsize=12)
    plt.yticks([])  # Remove y-axis ticks as they are not meaningful in this context
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig('results/anomaly_detection_normal_density.png')
    plt.show()

    # Create a density plot of the feature (anomalous points)

    plt.figure(figsize=(12, 6))
    plt.scatter(
        single_feature[range(len(single_feature))],
        np.zeros(len(single_feature)),
        c=["red" if i in anomalies else "blue" for i in range(len(single_feature))],
        alpha=0.6,
        label="Data Points"
    )

    # Add density plot for normal points
    sns.kdeplot(
        x=single_feature[[i for i in range(len(single_feature)) if i in anomalies]],
        color="red",
        fill=True,
        alpha=0.3,
        label="Density (Anomal Points)"
    )

    # Add title and labels
    plt.title(f"Anomaly Detection Relative to {single_feature_name}", fontsize=16)
    plt.xlabel(single_feature_name, fontsize=12)
    plt.yticks([])  # Remove y-axis ticks as they are not meaningful in this context
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig('results/anomaly_detection_anomalous_density.png')
    plt.show()

    # Visualize anomalies with Mahalanobis distance on the Y-axis for better clarity

    plt.figure(figsize=(12, 6))
    plt.scatter(
        single_feature[range(len(single_feature))],
        mahalanobis_distances,
        c=["red" if i in anomalies else "blue" for i in range(len(single_feature))],
        alpha=0.6,
        label="Data Points"
    )

    # Highlight anomalies with larger markers for emphasis
    plt.scatter(
        single_feature[anomalies],
        [mahalanobis_distances[i] for i in anomalies],
        c="red",
        s=50,
        alpha=0.9,
        label="Anomalies"
    )

    # Add title and labels
    plt.title(f"Anomaly Detection with Mahalanobis Distance\nRelative to {single_feature_name}", fontsize=16)
    plt.xlabel(single_feature_name, fontsize=12)
    plt.ylabel("Mahalanobis Distance", fontsize=12)
    plt.legend()
    plt.grid(True)
    if save:
        plt.savefig('results/anomaly_detection_mahalanobis_distance.png')
    plt.show()


if __name__ == "__main__":
    df_merged = pd.read_csv('data/merged_data_num_imputed_knn.csv')

    anomaly_detection(df_merged, save=True)