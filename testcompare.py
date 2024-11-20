import pandas as pd
import numpy as np
from scipy.stats import ttest_ind

# Загрузка двух датасетов
df1 = pd.read_csv('data\merged_data_num_imputed_median.csv')
df2 = pd.read_csv('data\merged_data_num_imputed_median.csv')

# Определите столбцы, которые хотите сравнить
columns_to_compare = [
    # 'Propulsion_Code',
    'Age_of_Vehicle', 
    'Driver_IMD_Decile', 
    'Engine_Capacity_.CC.', 
    'Vehicle_Location.Restricted_Lane', 
    'Did_Police_Officer_Attend_Scene_of_Accident', 
    'Pedestrian_Crossing-Human_Control', 
    'Pedestrian_Crossing-Physical_Facilities'
]

# Функция для сравнения столбцов
def compare_columns(data1, data2, columns):
    comparison_results = []
    for column in columns:
        if column in data1.columns and column in data2.columns:
            # Проверяем на наличие пропусков
            col1 = data1[column].dropna()
            col2 = data2[column].dropna()
            
            # Статистические тесты
            t_stat, p_val = ttest_ind(col1, col2, equal_var=True)  # t-тест для независимых выборок
            
            # Метрики
            mean_diff = abs(col1.mean() - col2.mean())  # Среднее абсолютное различие
            std_diff = abs(col1.std() - col2.std())    # Разница стандартных отклонений
            
            comparison_results.append({
                'Column': column,
                'Mean_Dataset1': col1.mean(),
                'Mean_Dataset2': col2.mean(),
                'Mean_Difference': mean_diff,
                'Std_Difference': std_diff,
                'T-Statistic': t_stat,
                'P-Value': p_val
            })
    return pd.DataFrame(comparison_results)

# Сравнение столбцов
results = compare_columns(df1, df2, columns_to_compare)

# Сохранение или отображение результатов
results.to_csv('comparison_results.csv', index=False)
print(results)
