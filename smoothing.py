# smoothing.py
import numpy as np
import pandas as pd

def moving_average(data, window_size=5):
    """
    移動平均を計算し、エッジ部分をパディングで補完する関数。

    Parameters:
    - data (list or np.ndarray): スムージングを適用するデータ。
    - window_size (int): 移動平均のウィンドウサイズ。

    Returns:
    - np.ndarray: スムージングされたデータ。
    """
    if window_size < 1:
        raise ValueError("window_size must be at least 1.")
    
    data = np.array(data)
    pad_size = window_size // 2
    # パディングを適用（エッジを反転させてパディング）
    padded_data = np.pad(data, (pad_size, pad_size), mode='edge')
    # 移動平均を計算
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded_data, kernel, mode='valid')
    return smoothed
