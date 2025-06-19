import os 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import torch.nn.functional as F
import argparse
from config import parse_args, parse_args_save
from utils import set_seed, set_device, clear_memory
import torch.nn as nn
from models import load_models

def create_experiment_directories(args, experiment_name):
    """
    experiment_name を外部から受け取り、ディレクトリを作成する。
    """
    root_experiment_dir = '/workspace/alpha_test/EMNIST'
    experiment_dir = os.path.join(root_experiment_dir, experiment_name)
    
    fig_and_log_dir = os.path.join(experiment_dir, 'fig_and_log')
    csv_dir = os.path.join(experiment_dir, 'csv')
    
    os.makedirs(fig_and_log_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print(f'Experiment directory created at: {experiment_dir}')
    print(f'fig_and_log directory created at: {fig_and_log_dir}')
    print(f'csv directory created at: {csv_dir}')
    
    return experiment_dir, fig_and_log_dir, csv_dir

def save_alpha_interpolation_results_to_csv(results, csv_dir, experiment_name, epoch):
    df = pd.DataFrame({
        'alpha': results['alpha_values'],
        'predicted_digit': results['predicted_digits'],
        'digit_label_match': results['digit_label_matches'],
    })
    digit_probs_df = pd.DataFrame(
        results['digit_probabilities'], 
        columns=[f'digit_probability_{i}' for i in range(len(results['digit_probabilities'][0]))]
    )
    full_df = pd.concat([df, digit_probs_df], axis=1)

    csv_path = os.path.join(csv_dir, f'alpha_log_epoch_{epoch}.csv')
    full_df.to_csv(csv_path, index=False)
    print(f'Alpha interpolation results saved to {csv_path}')

def save_selected_data_points(fig_and_log_dir, experiment_name, 
                              x1_np, x2_np, idx1, idx2, 
                              orig_label_x1, noisy_label_x1, distance1, 
                              orig_label_x2, noisy_label_x2, distance2, 
                              distance_x1_x2, epoch="0", cmap='gray'):
    """
    (x1, x2) とその情報を画像・ログに保存。
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # x2
    axes[0].imshow(x2_np.squeeze(), cmap=cmap)
    axes[0].set_title(
        f"Data idx={idx2}\n"
        f"Ori Label: {orig_label_x2}, Noisy Label: {noisy_label_x2}, C_Dis={distance2:.2f}"
    )
    axes[0].axis('off')

    # x1
    axes[1].imshow(x1_np.squeeze(), cmap=cmap)
    axes[1].set_title(
        f"Data idx={idx1}\n"
        f"Ori Label: {orig_label_x1}, Noisy Label: {noisy_label_x1}, "
        f"C_Dis={distance1:.2f}, Dis_x1_x2={distance_x1_x2:.2f}"
    )
    axes[1].axis('off')

    fig_path = os.path.join(fig_and_log_dir, f"selected_data_points_{idx1}_{idx2}_epoch_{epoch}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Selected data points image saved to {fig_path}')

    x1_npy_path = os.path.join(fig_and_log_dir, f"x1_idx{idx1}_epoch_{epoch}.npy")
    x2_npy_path = os.path.join(fig_and_log_dir, f"x2_idx{idx2}_epoch_{epoch}.npy")
    np.save(x1_npy_path, x1_np)
    np.save(x2_npy_path, x2_np)
    print(f'Selected data points saved as .npy files to {fig_and_log_dir}')

    distance_log_path = os.path.join(fig_and_log_dir, "distances.txt")
    with open(distance_log_path, 'a') as f:
        f.write(f"Experiment: {experiment_name}, Epoch: {epoch}\n")
        f.write(f"Data Point 1 - Index: {idx1}, Original Label: {orig_label_x1}, Noisy Label: {noisy_label_x1}, Distance to Centroid: {distance1:.2f}\n")
        f.write(f"Data Point 2 - Index: {idx2}, Original Label: {orig_label_x2}, Noisy Label: {noisy_label_x2}, Distance to Centroid: {distance2:.2f}\n")
        f.write(f"Distance between Data Points: {distance_x1_x2:.2f}\n")
        f.write("-" * 50 + "\n")
    print(f'Distance information logged to {distance_log_path}')

def compute_class_centroid_and_variance(dataset, labels, class_label):
    """
    クラス class_label 内の重心と分散を計算する。
    ※ class_label を本当に廃止したい場合はここも修正が必要
    """
    indices = np.where(labels == class_label)[0]
    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label}")

    images = torch.tensor(dataset[indices]).float()
    centroid = images.mean(dim=0)
    variance = images.var(dim=0)
    return centroid, variance

def find_closest_data_point_to_centroid(
    centroid, 
    dataset, 
    mode='no_noise', 
    noise_info=None
):
    """
    (x の取得) 重心にもっとも近いサンプルを検索。
    mode='no_noise' ならクリーンのみ、mode='noise' ならノイズのみから探す。
    """
    if mode not in ['no_noise', 'noise']:
        raise ValueError("mode must be 'no_noise' or 'noise'")

    if noise_info is None:
        raise ValueError("noise_info is None, but mode was specified.")

    if mode == 'no_noise':
        indices = np.where(noise_info == 0)[0]
    else:
        indices = np.where(noise_info == 1)[0]

    if len(indices) == 0:
        raise ValueError(f"No data points found under mode='{mode}'")

    images = torch.tensor(dataset[indices]).float()
    distances = torch.norm(images.view(len(images), -1) - centroid.view(-1), dim=1)

    min_distance, min_index = torch.min(distances, dim=0)
    closest_data_point = images[min_index].numpy()
    data_point_index = indices[min_index].item()

    return closest_data_point, data_point_index, min_distance.item()

def find_closest_data_point(
    reference_data,
    dataset,
    mode='no_noise',
    noise_info=None
):
    """
    (y の取得) 任意の点 reference_data に最も近いサンプルを検索。
    mode='no_noise' ならクリーンのみ、mode='noise' ならノイズのみ。
    """
    if mode not in ['no_noise', 'noise']:
        raise ValueError("mode must be 'no_noise' or 'noise'")

    if noise_info is None:
        raise ValueError("noise_info is None, but mode was specified.")

    if mode == 'no_noise':
        indices = np.where(noise_info == 0)[0]
    else:
        indices = np.where(noise_info == 1)[0]

    if len(indices) == 0:
        raise ValueError(f"No data points found under mode='{mode}'")

    images = torch.tensor(dataset[indices]).float()
    ref_t = torch.tensor(reference_data).float().view(-1)

    distances = torch.norm(images.view(len(images), -1) - ref_t, dim=1)
    min_distance, min_index = torch.min(distances, dim=0)

    closest_data_point = images[min_index].numpy()
    data_point_index = indices[min_index].item()

    return closest_data_point, data_point_index, min_distance.item()

def alpha_interpolation_test_save_emnist(
    model, x_clean, x_noisy,
    digit_label_x, digit_label_y,
    num_digits, device
):
    """
    (x, y) の α 補間をして、その予測結果を返す。
    digit_label_x, digit_label_y は "ノイズ後ラベル" などで指定してOK。
    """
    alpha_values = np.arange(-0.5, 1.51, 0.01)
    model.eval()

    x_clean = x_clean.to(device)
    x_noisy = x_noisy.to(device)

    predicted_digits = []
    digit_probabilities = []
    digit_label_matches = []

    with torch.no_grad():
        for alpha in alpha_values:
            z = alpha * x_clean + (1 - alpha) * x_noisy
            z = z.unsqueeze(0)

            outputs = model(z)
            output_probs = F.softmax(outputs, dim=1)
            digit_probs = output_probs.squeeze(0).cpu().numpy()

            predicted_digit = np.argmax(digit_probs)
            predicted_digits.append(predicted_digit)
            digit_probabilities.append(digit_probs)

            # 簡易的なラベル一致判定例
            if predicted_digit == digit_label_x:
                digit_label_matches.append(1)
            elif predicted_digit == digit_label_y:
                digit_label_matches.append(-1)
            else:
                digit_label_matches.append(0)

    return {
        'alpha_values': alpha_values.tolist(),
        'predicted_digits': predicted_digits,
        'digit_probabilities': digit_probabilities,
        'digit_label_matches': digit_label_matches,
    }

def main():
    args = parse_args_save()
    set_seed(args.fix_seed)
        
    # デバイス設定
    device = set_device(args.gpu)
    print(f'Using device: {device}')

    # -----------------
    # モデル関連セットアップ
    # -----------------
    model_root_dir = '/workspace/save_model/EMNIST'
    model_sub_dir = (f'{args.model}_width{args.model_width}_{args.dataset}_lr{args.lr}_batch_size{args.batch_size}'
                     f'_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_momentum{args.momentum}')
    model_dir = os.path.join(model_root_dir, model_sub_dir)
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    model_files = [f for f in os.listdir(model_dir) if f.startswith('epoch_') and f.endswith('.pth')]
    model_files.sort(key=lambda x: int(x.split('_')[1].split('.pth')[0]))
    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}.")
    print(f'Found {len(model_files)} model files in {model_dir}.')

    # -----------------
    # データセットの読み込み
    # -----------------
    no_noise_dir = '/workspace/data/EMNIST/EMNIST_0.0'
    train_images = np.load(os.path.join(no_noise_dir, 'train_images.npy'))
    train_labels = np.load(os.path.join(no_noise_dir, 'train_labels.npy'))
    train_noise_info = np.load(os.path.join(no_noise_dir, 'train_noise_info.npy'))

    noisy_dir = os.path.join('/workspace/data/EMNIST', f'EMNIST_{args.label_noise_rate}')
    noisy_train_images = np.load(os.path.join(noisy_dir, 'train_images.npy'))
    noisy_train_labels = np.load(os.path.join(noisy_dir, 'train_labels.npy'))
    noisy_train_noise_info = np.load(os.path.join(noisy_dir, 'train_noise_info.npy'))
    noisy_test_images = np.load(os.path.join(noisy_dir, 'test_images.npy'))
    noisy_test_labels = np.load(os.path.join(noisy_dir, 'test_labels.npy'))

    # -----------------
    # モデル準備
    # -----------------
    model = load_models(in_channels=1, args=args, img_size=(32,32), num_classes=10).to(device)

    # -----------------
    # (A) 重心計算
    # -----------------
    n1 = args.n1
    centroid_n1, var_n1 = compute_class_centroid_and_variance(train_images, train_labels, class_label=n1)

    # -----------------
    # (B) x を取得 (例: 重心に最も近い clean データ)
    # -----------------
    x, idx_x, dist_x = find_closest_data_point_to_centroid(
        centroid=centroid_n1,
        dataset=train_images,
        mode='noise',       # または 'noise'
        noise_info=noisy_train_noise_info
    )
    # ノイズ後ラベル
    noisy_label_x = noisy_train_labels[idx_x]

    # -----------------
    # (C) y を取得 (例: x から最も近い noise データ)
    #     ※ ここはお好みで 'no_noise' or 'noise'
    # -----------------
    y, idx_y, dist_y = find_closest_data_point(
        reference_data=x,
        dataset=train_images,
        mode='no_noise',          # 例: ここではノイズサンプルから探す
        noise_info=noisy_train_noise_info
    )
    # ノイズ後ラベル
    noisy_label_y = noisy_train_labels[idx_y]

    # 2点間の距離a
    dist_xy = np.linalg.norm(x - y)

    # -----------------
    # experiment_name の組み立て
    #    以前は .../{args.n1}/{args.n2} の部分を
    #    .../{noisy_label_x}/{noisy_label_y} に変更
    # -----------------
    experiment_name = (
        f'{args.model}_{args.dataset}_variance{args.variance}_{args.target}'
        f'_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}'
        f'_Optim{args.optimizer}_Momentum{args.momentum}'
        f'/{args.mode1}_{args.mode2}'
        f'/{noisy_label_y}/{noisy_label_x}'
    )

    # -----------------
    # 実験用ディレクトリ作成
    # -----------------
    experiment_dir, fig_and_log_dir, csv_dir = create_experiment_directories(args, experiment_name)

    # -----------------
    # (x, y) の画像や情報を保存
    # -----------------
    save_selected_data_points(
        fig_and_log_dir=fig_and_log_dir,
        experiment_name=experiment_name,
        x1_np=x,
        x2_np=y,
        idx1=idx_x,
        idx2=idx_y,
        orig_label_x1=train_labels[idx_x],
        noisy_label_x1=noisy_train_labels[idx_x],
        distance1=dist_x,
        orig_label_x2=train_labels[idx_y],
        noisy_label_x2=noisy_train_labels[idx_y],
        distance2=dist_y,
        distance_x1_x2=dist_xy,
        epoch="0"
    )

    # -----------------
    # モデルごとの α補間テスト
    # -----------------
    for model_file in model_files:
        epoch_num = int(model_file.split('_')[1].split('.pth')[0])
        model_path = os.path.join(model_dir, model_file)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        x_clean = torch.tensor(x).float()
        x_noisy = torch.tensor(y).float()

        # ここで digit_label_x, digit_label_y をどう設定するかは用途による
        # 例：x のノイズ後ラベル, y のノイズ後ラベルを渡す
        results = alpha_interpolation_test_save_emnist(
            model=model,
            x_clean=x_clean,
            x_noisy=x_noisy,
            digit_label_x=noisy_label_x,   # ノイズ後のラベル
            digit_label_y=noisy_label_y,   # ノイズ後のラベル
            num_digits=10,
            device=device
        )

        save_alpha_interpolation_results_to_csv(results, csv_dir, experiment_name, epoch_num)
        clear_memory()

    print("Process completed with new (x, y) logic and experiment_name including their noisy labels.")

if __name__ == "__main__":
    main()
