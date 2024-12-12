# Road-Safety-Analysis
Analyzing UK Road Safety dataset using CRISP DM - https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles

# UK Road Safety: Traffic Accidents and Vehicles

This repository contains a comprehensive analysis of the UK Road Safety dataset, utilizing the CRISP-DM methodology to understand the factors influencing traffic accident severity and develop predictive and descriptive models.

<img src="https://miro.medium.com/v2/resize:fit:1200/1*JYbymHifAk7aQ1pHm_IdMQ.png" alt="CRISP DM Diagram" width="600" />

## Table of Contents

1. [Introduction](#introduction)
2. [Business and KDD Goals](#business-and-kdd-goals)
3. [Data Understanding and Preparation](#data-understanding-and-preparation)
4. [Modeling](#modeling)
5. [Evaluation](#evaluation)
6. [Conclusion](#conclusion)

## Introduction

This project applies the CRISP-DM framework to the "UK Road Safety: Traffic Accidents and Vehicles" dataset obtained from Kaggle [Dataset](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles). The primary objective is to identify key factors influencing accident severity and to build models for prediction and analysis.

## Business and KDD Goals

### Business Goal
Improve the understanding of factors leading to varying levels of traffic accident severity, with the aim of developing models to predict such scenarios and contribute to traffic safety improvements.

### Knowledge Discovery in Databases (KDD) Goal
Identify critical attributes affecting accident severity and develop predictive/classification and descriptive models to analyze these factors.

### Data Sources
The dataset consists of:
- **Accident Information**: Details about traffic accidents.
- **Vehicle Information**: Information about vehicles involved in accidents.

## Data Understanding and Preparation

### Dataset Creation
- Merged data using the key `Accident Index`.
- Removed columns with >40% missing values.

### Handling Missing Data
- Used `SimpleImputer` for categorical attributes (most frequent strategy) and numerical attributes (mean strategy).
- Applied `KNNImputer` for advanced processing of numerical attributes.

### Oversampling
Addressed data imbalance using the **SMOTE** technique, improving the representation of severe and fatal accidents.

## Modeling

### Classification Models
The following models were tested:
- **Random Forest**
- **SVM**
- **Gradient Boosting**
- **XGBoost**
- **MLPClassifier**

Optimized hyperparameters using `GridSearchCV`:
- **Random Forest**: Max depth = 20, Trees = 300.
- **XGBoost**: Learning rate = 0.1, Max depth = 15, Trees = 300.

### Descriptive Models
Performed anomaly detection using:
- **K-Means Clustering**
- **Mahalanobis Distance**

## Evaluation

### Predictive Models
Evaluation metrics included precision, recall, accuracy, and F1-score. The best-performing model for this task was **XGBoost**, which demonstrated robust classification capabilities compared to other models.

### Descriptive Models
Key findings:
- Age of vehicles and engine capacity were significant in identifying anomalies.
- Visualized distributions and anomaly clusters.

## Conclusion
The project successfully applied CRISP-DM to analyze traffic accidents. The findings provide insights into traffic safety and offer a basis for future implementations related to accident prediction and prevention.

## References
1. Andreas Tsiaras. UK Road Safety - Accidents and Vehicles. Kaggle. [Dataset](https://www.kaggle.com/datasets/tsiaras/uk-road-safety-accidents-and-vehicles).
2. GeeksforGeeks. SVM Hyperparameter Tuning using GridSearchCV. [Article](https://www.geeksforgeeks.org/svm-hyperparameter-tuning-using-gridsearchcv-ml/).