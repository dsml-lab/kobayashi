import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def calculate_eigenvalues(variance_values, correlation=0.5):
    """
    Calculate eigenvalues of covariance matrices for given variances and a fixed correlation.

    Parameters:
    variance_values (list): List of variances to use.
    correlation (float): Correlation coefficient between -1 and 1.

    Returns:
    list: Eigenvalues for each covariance matrix, sorted in descending order.
    """
    eigenvalues_list = []
    for variance in variance_values:
        if variance < 0:
            raise ValueError("Variance must be non-negative.")
        # Calculate standard deviation
        std_dev = np.sqrt(variance)

        # Create correlation matrix
        corr_matrix = np.array([[1, correlation, correlation],
                                [correlation, 1, correlation],
                                [correlation, correlation, 1]])

        # Compute covariance matrix
        cov = np.outer(std_dev, std_dev) * corr_matrix

        # Compute eigenvalues
        eigenvalues = np.linalg.eigvals(cov)

        # Sort eigenvalues in descending order
        sorted_eigenvalues = np.sort(eigenvalues)[::-1]
        eigenvalues_list.append(sorted_eigenvalues)

    return eigenvalues_list

# Variance values to test
variance_values = [0, 100, 3162, 10000]

# 固有値を計算
eigenvalues_list = calculate_eigenvalues(variance_values)

# CSVに保存するためのデータ準備
data = []
for variance, eigenvalues in zip(variance_values, eigenvalues_list):
    data.append([variance] + eigenvalues.tolist())

# データフレームの作成
df = pd.DataFrame(data, columns=['Variance', 'Eigenvalue1', 'Eigenvalue2', 'Eigenvalue3'])

# CSVファイルとして保存
csv_filename = 'eigenvalues.csv'
df.to_csv(csv_filename, index=False, encoding='utf-8-sig')

print(f"固有値が {csv_filename} に保存されました。")
