import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams["figure.dpi"] = 400
plt.rcParams["font.size"] = 18
# CSVファイルのパス
file1 = '/workspace/miru_vizualize/distances_root_dir.csv'
file2 = '/workspace/miru_vizualize/distances_root_dir2.csv'

# データの読み込み
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)

# Distance Betweenの抽出
dist1 = df1['Distance Between']
dist2 = df2['Distance Between']

# ヒストグラムの描画
plt.figure(figsize=(8, 5))
plt.hist(dist1, bins=20, alpha=0.6, label='clean - clean',color="blue")
plt.hist(dist2, bins=20, alpha=0.6, label='clean - noisy', color="red")

# グラフの装飾
plt.xlabel('Distance Between $x_0$ and $x_1$')
plt.ylabel('frequency')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存 or 表示
plt.savefig('hist_distance_between.pdf')  # 保存する場合
print('hist_distance_between.pdf')
# plt.show()  # 表示したい場合はこちらを使う
