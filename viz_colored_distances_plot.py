import os
import pandas as pd
import re

def extract_distances_for_csv(root_dir):
    data = []
    for x in range(100):
        for y in range(100):
            distances_file = os.path.join(root_dir, str(x), str(y), "fig_and_log", "distances.txt")
            if os.path.exists(distances_file):
                with open(distances_file, "r") as f:
                    lines = f.readlines()
                
                distance_to_centroid = None
                distance_between = None
                
                for line in lines:
                    match1 = re.search(r"Distance to centroid.*?:\s*([\d.]+)", line)
                    match2 = re.search(r"Distance between .*?:\s*([\d.]+)", line)

                    if match1:
                        distance_to_centroid = float(match1.group(1))
                    if match2:
                        distance_between = float(match2.group(1))

                if distance_to_centroid is not None and distance_between is not None:
                    # distance.txt のディレクトリから 2 つ上と 1 つ上のディレクトリ名を取得
                    parent_dir = os.path.dirname(os.path.dirname(distances_file))
                    parts = parent_dir.split('/')
                    dir_name = f"{parts[-2]}/{parts[-1]}"
                    data.append([dir_name, distance_to_centroid, distance_between])
    
    return pd.DataFrame(data, columns=["Directory", "Distance to Centroid", "Distance Between"])

# ルートディレクトリの設定
root_dir = "/workspace/alpha_test/seed/width_4/lr_0.0_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.0_Optimsgd_Momentum0.0/no_noise_no_noise"
root_dir_2 = "/workspace/alpha_test/seed/width_4/lr_0.2_sigma_0/ori_closet_seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise"

# データの抽出
df1 = extract_distances_for_csv(root_dir)
df2 = extract_distances_for_csv(root_dir_2)

# CSVの保存
df1.to_csv("/workspace/miru_vizualize/distances_root_dir.csv", index=False)
df2.to_csv("/workspace/miru_vizualize/distances_root_dir2.csv", index=False)

# 出力の確認
print("Saved CSV files:")
print("/workspace/miru_vizualize/distances_root_dir.csv")
print("/workspace/miru_vizualize/distances_root_dir2.csv")