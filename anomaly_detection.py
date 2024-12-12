import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from preprocesing_knn import scale_data
from sklearn.cluster import KMeans
from scipy.spatial.distance import mahalanobis

os.environ['OMP_NUM_THREADS'] = '1'

def anomaly_detection(data: pd.DataFrame, n_clusters: int=100, threshold_percentile: int=99, save: bool=False, show: bool=True):
    """
    Perform anomaly detection on the given dataset using K-Means clustering and Mahalanobis distance.

    Parameters:
    - data: pd.DataFrame
        The input dataset containing numerical features.
    - n_clusters: int
        The number of clusters to form using K-Means.
    - threshold_percentile: int
        The percentile to determine the threshold for anomalies.
    - save: bool
        Whether to save the generated plots.
    - show: bool
        Whether to display the generated plots.

    Returns:
    - None
    
    Steps:
    1. Select numerical features from the dataset.
    2. Scale the data using the provided scaling function.
    3. Calculate variances of the numerical features.
    4. Perform K-Means clustering on the scaled data.
    5. Calculate Mahalanobis distances for each point in each cluster.
    6. Determine the threshold for anomalies based on the specified percentile.
    7. Identify anomalies based on the calculated threshold.
    8. Generate and optionally save/display plots for anomaly detection relative to specified features.
    """

    numerical_data = data.select_dtypes(include=[np.number])
    scaled_data, scaler = scale_data(numerical_data, inverse=True)

    variances = numerical_data.var()
    sorted_variances = variances.sort_values(ascending=False)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(scaled_data)

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

    threshold = np.percentile(mahalanobis_distances, threshold_percentile)
    anomalies = [i for i, dist in enumerate(mahalanobis_distances) if dist > threshold]

    tables_to_analyze = [
        'Age_of_Vehicle',
        'Engine_Capacity_.CC.',
    ]

    for feature_name in tables_to_analyze:
        
        feature_index = numerical_data.columns.get_loc(feature_name)
        single_feature = scaler.inverse_transform(scaled_data)[:, feature_index]

        plt.figure(figsize=(12, 6))
        plt.scatter(
            single_feature,
            np.zeros(len(single_feature)),
            c=["red" if i in anomalies else "blue" for i in range(len(single_feature))],
            alpha=0.6,
            label="Data Points"
        )

        sns.kdeplot(
            x=single_feature[[i for i in range(len(single_feature)) if i not in anomalies]],
            color="blue",
            fill=True,
            alpha=0.3,
            label="Density (Normal Points)"
        )

        plt.title(f"Anomaly Detection Relative to {feature_name}", fontsize=16)
        plt.xlabel(feature_name, fontsize=12)
        plt.yticks([]) 
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(f'results/anomaly_detection_normal_density_{feature_name}.png')
        if show:
            plt.show()

        plt.figure(figsize=(12, 6))
        plt.scatter(
            single_feature,
            np.zeros(len(single_feature)),
            c=["red" if i in anomalies else "blue" for i in range(len(single_feature))],
            alpha=0.6,
            label="Data Points"
        )

        sns.kdeplot(
            x=single_feature[[i for i in range(len(single_feature)) if i in anomalies]],
            color="red",
            fill=True,
            alpha=0.3,
            label="Density (Anomalous Points)"
        )

        plt.title(f"Anomaly Detection Relative to {feature_name}", fontsize=16)
        plt.xlabel(feature_name, fontsize=12)
        plt.yticks([])
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(f'results/anomaly_detection_anomalous_density_{feature_name}.png')
        if show:
            plt.show()

        plt.figure(figsize=(12, 6))
        plt.scatter(
            single_feature,
            mahalanobis_distances,
            c=["red" if i in anomalies else "blue" for i in range(len(single_feature))],
            alpha=0.6,
            label="Data Points"
        )

        plt.scatter(
            single_feature[anomalies],
            [mahalanobis_distances[i] for i in anomalies],
            c="red",
            s=50,
            alpha=0.9,
            label="Anomalies"
        )

        plt.title(f"Anomaly Detection with Mahalanobis Distance\nRelative to {feature_name}", fontsize=16)
        plt.xlabel(feature_name, fontsize=12)
        plt.ylabel("Mahalanobis Distance", fontsize=12)
        plt.legend()
        plt.grid(True)
        if save:
            plt.savefig(f'results/anomaly_detection_mahalanobis_distance_{feature_name}.png')
        if show:
            plt.show()
