import os
import csv
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from datasets import load_datasets, BalancedBatchSampler, NoisyDataset, load_or_create_noisy_dataset
import numpy as np
import argparse

# デバッグ用スイッチ
DEBUG = True

def debug_print(*args):
    if DEBUG:
        print("[DEBUG]", *args)

gpu_number = 2  # 例えば2番目のGPU (0-based index)

# GPUが利用可能かつ指定した番号のGPUが使える場合
if torch.cuda.is_available() and gpu_number < torch.cuda.device_count():
    device = torch.device(f"cuda:{gpu_number}")
else:
    device = torch.device("cpu")

print(f'Using device: {device}')

# バッチ書き込み用のグローバル辞書
# key: ファイルパス, value: (mode, content)
# mode "csv" の場合は (header, rows) のタプル、"text" の場合は文字列として内容を保持する
batch_data = {}

def batch_write_file(file_path, mode, content):
    """グローバル辞書に内容を蓄積する"""
    batch_data[file_path] = (mode, content)

def flush_batch_data():
    """蓄積された内容をすべてディスクへ書き出す"""
    for file_path, (mode, content) in batch_data.items():
        # 出力先のディレクトリが存在しなければ作成
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        if mode == "csv":
            header, rows = content
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            debug_print("Batch wrote CSV:", file_path)
        elif mode == "text":
            with open(file_path, 'w') as f:
                f.write(content)
            debug_print("Batch wrote text:", file_path)
    # 書き込み後は辞書をクリア
    batch_data.clear()

def compute_class_centroid_and_variance(dataset, labels, class_label):
    """
    指定したクラスラベル (class_label) に属するすべてのデータ点から、
    重心 (centroid) と分散 (variance) を計算して返す。
    GPUとベクトル化を使用。
    """
    debug_print("Computing centroid and variance for class", class_label)
    indices = torch.where(labels == class_label)[0]
    debug_print("Found indices:", indices)
    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label}")

    if hasattr(dataset, 'tensors'):
        images = dataset.tensors[0][indices].to(device)
    else:
        images = torch.stack([dataset[i][0] for i in indices]).to(device)
    debug_print("Images shape:", images.shape)

    centroid = images.mean(dim=0)
    variance = images.var(dim=0)
    debug_print("Centroid shape:", centroid.shape, "Variance shape:", variance.shape)
    return centroid.cpu(), variance.cpu()

def compute_distances_to_centroid(dataset, labels, class_label, centroid, mode='no_noise', noise_info=None):
    """
    指定したクラスラベルに属するデータ点を、mode と noise_info に応じて
    選別し、重心とのユークリッド距離を計算して返す。
    GPU とベクトル化による処理を使用。
    """
    debug_print("Computing distances for class", class_label, "mode:", mode)
    indices = torch.where(labels == class_label)[0]
    debug_print("Initial indices:", indices)
    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label}")

    if noise_info is not None:
        if mode == 'no_noise':
            indices = indices[noise_info[indices] == 0]
        elif mode == 'noise':
            indices = indices[noise_info[indices] == 1]
        debug_print("Indices after noise filtering:", indices)
        if len(indices) == 0:
            raise ValueError(f"No data points found for class label {class_label} under mode='{mode}'.")

    if hasattr(dataset, 'tensors'):
        images = dataset.tensors[0][indices].to(device)
    else:
        images = torch.stack([dataset[i][0] for i in indices]).to(device)
    debug_print("Images shape after filtering:", images.shape)

    centroid_flat = centroid.view(-1).to(device)
    images_flat = images.view(images.size(0), -1)
    debug_print("Centroid_flat shape:", centroid_flat.shape, "Images_flat shape:", images_flat.shape)

    distances_tensor = torch.norm(images_flat - centroid_flat.unsqueeze(0), dim=1, p=2)
    debug_print("Distances tensor:", distances_tensor)

    distances = distances_tensor.cpu().tolist()
    return distances, indices.cpu().tolist() if isinstance(indices, torch.Tensor) else indices

def save_distance_results(class_label, distances, indices, base_dir='cen_distance', mode='no_noise', noise_info=None, args=None):
    """
    距離情報と統計量をCSV、TXT、ヒストグラム画像で保存する。
    CSVとTXTはバッチ書き込み用に内容をメモリに蓄積する。
    """
    sub_sub_dir = "noisy_info=none" if noise_info is None else mode
    out_dir = os.path.join(base_dir, "distribution_colored_emnist", str(class_label), sub_sub_dir)
    os.makedirs(out_dir, exist_ok=True)
    debug_print("Saving distance results in directory:", out_dir)

    dist_tensor = torch.tensor(distances)
    distances_np = dist_tensor.cpu().numpy()
    
    mean_val, std_val, var_val = distances_np.mean(), distances_np.std(), distances_np.var()
    debug_print("Distance stats - Mean:", mean_val, "Std:", std_val, "Var:", var_val)

    # CSV用内容を蓄積
    csv_path = os.path.join(out_dir, 'distance.csv')
    header = ['index', 'distance']
    rows = list(zip(indices, distances_np))
    batch_write_file(csv_path, "csv", (header, rows))

    # 統計情報をテキストとして蓄積
    stats_path = os.path.join(out_dir, 'stats.txt')
    stats_content = (f"Mean: {mean_val}\n"
                     f"Standard Deviation: {std_val}\n"
                     f"Variance: {var_val}\n"
                     f"Min Distance: {distances_np.min()}\n")
    batch_write_file(stats_path, "text", stats_content)

    # ヒストグラム画像はそのまま保存
    plt.figure()
    plt.hist(distances_np, bins=30, alpha=0.7, color='blue')
    plt.title(f"Distance Histogram for label {class_label} [{sub_sub_dir}]")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    hist_path = os.path.join(out_dir, 'distance_hist.png')
    plt.savefig(hist_path)
    plt.close()
    debug_print("Saved histogram image:", hist_path)

def save_secondary_distance_results(class_label, secondary_distances, secondary_indices, base_dir='cen_distance', mode='no_noise', noise_info=None, suffix='_from_closest', args=None):
    """
    セカンダリ距離（重心に最も近いデータ点を基準）の結果を
    CSV、TXT、ヒストグラム画像で保存する。
    CSVとTXTはバッチ書き込み用に内容をメモリに蓄積する。
    """
    sub_sub_dir = "noisy_info=none" if noise_info is None else mode
    out_dir = os.path.join(base_dir, "distribution_colored_emnist", str(class_label), sub_sub_dir)
    os.makedirs(out_dir, exist_ok=True)
    debug_print("Saving secondary distance results in directory:", out_dir)

    secondary_distances_np = np.array(secondary_distances)
    mean_val = secondary_distances_np.mean()
    std_val = secondary_distances_np.std()
    var_val = secondary_distances_np.var()
    debug_print("Secondary distance stats - Mean:", mean_val, "Std:", std_val, "Var:", var_val)
    
    # CSVの内容を蓄積
    csv_filename = os.path.join(out_dir, f'distance{suffix}.csv')
    header = ['index', 'distance']
    rows = list(zip(secondary_indices, secondary_distances_np))
    batch_write_file(csv_filename, "csv", (header, rows))
    
    # 統計情報テキストの蓄積
    stats_filename = os.path.join(out_dir, f'stats{suffix}.txt')
    stats_content = (f"Mean: {mean_val}\n"
                     f"Standard Deviation: {std_val}\n"
                     f"Variance: {var_val}\n"
                     f"Min Distance: {secondary_distances_np.min()}\n")
    batch_write_file(stats_filename, "text", stats_content)
    
    # ヒストグラム画像はそのまま保存
    plt.figure()
    plt.hist(secondary_distances_np, bins=30, alpha=0.7, color='blue')
    plt.title(f"Distance Histogram for label {class_label} {suffix} [{sub_sub_dir}]")
    plt.xlabel("Distance")
    plt.ylabel("Frequency")
    hist_filename = os.path.join(out_dir, f'distance_hist{suffix}.png')
    plt.savefig(hist_filename)
    plt.close()
    debug_print("Saved secondary histogram image:", hist_filename)

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize distance to centroid with or without noise")
    parser.add_argument('--data_dir', type=str, default='/workspace/data', help='Root data directory')
    parser.add_argument("-correlation", "--correlation", type=float, default=0.5, help="Correlation parameter for distribution datasets")
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
    parser.add_argument("-gray_scale", "--gray_scale", action='store_true', help="Convert images to grayscale")
    parser.add_argument('--target', type=str, required=True, choices=['combined', 'digits', 'colors'], help='Target type')
    parser.add_argument('--fix_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--variance', type=int, required=True, help='Variance value')
    parser.add_argument('--label_noise_rate', type=float, required=True, help='Label noise rate')    
    return parser.parse_args()

def main():
    args = parse_args()
    debug_print("Arguments parsed:", args)
    
    try:
        print('Loading datasets...')
        try:
            train_dataset_original, test_dataset, imagesize, num_classes, in_channels = load_datasets(
                args.dataset, args.target, args.gray_scale, args
            )
            debug_print("Loaded original dataset. Imagesize:", imagesize, "Num classes:", num_classes)
        except FileNotFoundError as e:
            print(f"Error loading dataset: {e}")
            return
        except Exception as e:
            print(f"Unexpected error loading dataset: {e}")
            return
        
        if hasattr(train_dataset_original, 'tensors'):
            x_original_train, y_original_train = train_dataset_original.tensors
        elif hasattr(train_dataset_original, 'targets'):
            y_original_train = torch.tensor(train_dataset_original.targets)
            x_original_train = None
        else:
            raise ValueError("Cannot extract clean labels from train_dataset_original")
        original_targets = y_original_train  # Clean labels
        debug_print("Original targets shape:", original_targets.shape)
        
        print('Creating noisy datasets...')
        train_dataset, test_dataset, meta = load_or_create_noisy_dataset(
            args.dataset, args.target, args.gray_scale, args, return_type="torch"
        )
        if isinstance(train_dataset, NoisyDataset):
            x_train_noisy, y_train_noisy = train_dataset.dataset.tensors
            noise_info = train_dataset.noise_info
        else:
            x_train_noisy, y_train_noisy = train_dataset.tensors
            noise_info = None
        dataset = train_dataset
        targets = y_train_noisy
        debug_print("Noisy dataset loaded. Targets shape:", targets.shape)

        unique_labels = torch.unique(original_targets)
        debug_print("Unique labels:", unique_labels)
        for c_label in unique_labels:
            debug_print("Processing label:", c_label)
            centroid, _ = compute_class_centroid_and_variance(dataset, original_targets, c_label)

            for mode in ['no_noise', 'noise']:
                try:
                    distances, indices = compute_distances_to_centroid(dataset, original_targets, c_label, centroid, mode, noise_info)
                    debug_print(f"Label {c_label} mode {mode} distances computed. Count:", len(distances))
                    save_distance_results(c_label, distances, indices, mode=mode, noise_info=noise_info)
                    
                    # --- 追加機能：重心に最も近いデータ点を基準としたセカンダリ距離計算 ---
                    if len(distances) > 1:
                        distances_np = np.array(distances)
                        min_pos = int(np.argmin(distances_np))
                        closest_dataset_index = indices[min_pos]
                        debug_print("Closest dataset index:", closest_dataset_index)
                        if hasattr(dataset, 'tensors'):
                            closest_img = dataset.tensors[0][closest_dataset_index].to(device)
                        else:
                            closest_img = dataset[closest_dataset_index][0].to(device)
                        indices_tensor = torch.tensor(indices)
                        mask = indices_tensor != closest_dataset_index
                        secondary_indices_tensor = indices_tensor[mask]
                        debug_print("Secondary indices:", secondary_indices_tensor)
                        if hasattr(dataset, 'tensors'):
                            imgs = dataset.tensors[0][secondary_indices_tensor].to(device)
                        else:
                            imgs = torch.stack([dataset[i][0] for i in secondary_indices_tensor]).to(device)
                        imgs_flat = imgs.view(imgs.size(0), -1)
                        closest_img_flat = closest_img.view(-1)
                        secondary_distances_tensor = torch.norm(imgs_flat - closest_img_flat.unsqueeze(0), dim=1, p=2)
                        secondary_distances = secondary_distances_tensor.cpu().tolist()
                        secondary_indices = secondary_indices_tensor.cpu().tolist()
                        debug_print("Secondary distances computed. Count:", len(secondary_distances))
                        if len(secondary_distances) > 0:
                            save_secondary_distance_results(c_label, secondary_distances, secondary_indices, mode=mode, noise_info=noise_info, suffix='_from_closest')
                    # ---------------------------------------------------------------
                    
                except ValueError as e:
                    print(f"[WARN] {e}")

            distances, indices = compute_distances_to_centroid(dataset, targets, c_label, centroid, mode='no_noise', noise_info=None)
            debug_print("Processing label", c_label, "for noise_info=None. Count:", len(distances))
            save_distance_results(c_label, distances, indices, mode='no_noise', noise_info=None)
            if len(distances) > 1:
                distances_np = np.array(distances)
                min_pos = int(np.argmin(distances_np))
                closest_dataset_index = indices[min_pos]
                if hasattr(dataset, 'tensors'):
                    closest_img = dataset.tensors[0][closest_dataset_index].to(device)
                else:
                    closest_img = dataset[closest_dataset_index][0].to(device)
                indices_tensor = torch.tensor(indices)
                mask = indices_tensor != closest_dataset_index
                secondary_indices_tensor = indices_tensor[mask]
                if hasattr(dataset, 'tensors'):
                    imgs = dataset.tensors[0][secondary_indices_tensor].to(device)
                else:
                    imgs = torch.stack([dataset[i][0] for i in secondary_indices_tensor]).to(device)
                imgs_flat = imgs.view(imgs.size(0), -1)
                closest_img_flat = closest_img.view(-1)
                secondary_distances_tensor = torch.norm(imgs_flat - closest_img_flat.unsqueeze(0), dim=1, p=2)
                secondary_distances = secondary_distances_tensor.cpu().tolist()
                secondary_indices = secondary_indices_tensor.cpu().tolist()
                debug_print("Secondary distances (noise_info=None) computed. Count:", len(secondary_distances))
                if len(secondary_distances) > 0:
                    save_secondary_distance_results(c_label, secondary_distances, secondary_indices, mode='no_noise', noise_info=None, suffix='_from_closest')

        # バッチ書き込み内容を一括でディスクへ出力
        flush_batch_data()

        print("[INFO] 処理完了。'cen_distance/' 以下に出力が保存されました。")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
