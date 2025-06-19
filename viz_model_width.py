import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# ルートディレクトリ
root_dir = "/workspace/save_model/cifar10/noise_0.2"

# 検索対象のエポック
target_epochs = [4000, 2000]

# 結果保存用
results = []

# ディレクトリを走査
for dir_name in os.listdir(root_dir):
    dir_path = os.path.join(root_dir, dir_name)
    csv_path = os.path.join(dir_path, "csv", "training_metrics.csv")
    
    # widthの抽出
    match = re.search(r"width(\d+)", dir_name)
    if not match or not os.path.isfile(csv_path):
        continue
    width = int(match.group(1))
    
    # CSV読み込み
    df = pd.read_csv(csv_path)
    
    # epochの該当行を取得
    row = None
    for epoch in target_epochs:
        row = df[df["epoch"] == epoch]
        if not row.empty:
            break
    
    if row is not None and not row.empty:
        error = row.iloc[0]["test_error"]
        results.append((width, error))

# 幅でソート
results.sort(key=lambda x: x[0])
widths, errors = zip(*results)

# プロット
plt.figure(figsize=(8,5))
plt.plot(widths, errors, marker='o')
plt.xlabel("Model Width")
plt.ylabel("Train Error")
plt.title("Train Error at Epoch 4000 (or 2000 if 4000 not found)")
plt.grid(True)
plt.tight_layout()
plt.savefig("train_error_vs_width.png")
print("train_error_vs_width.png")