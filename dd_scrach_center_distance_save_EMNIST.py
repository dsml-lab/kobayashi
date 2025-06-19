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


def create_experiment_directories(args):
    """
    指定された引数に基づいて実験用ディレクトリを作成します。
    
    Args:
        args: 実験の設定を含む引数オブジェクト。
    
    Returns:
        tuple: experiment_dir, fig_and_log_dir, csv_dir, experiment_name
    """
    # experiment_nameの生成
    experiment_name = f'{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}/{args.mode1}_{args.mode2}/{args.n1}/{args.n2}'
    
    # ルートディレクトリ
    root_experiment_dir = '/workspace/alpha_test/EMNIST'
    
    # experiment_nameに基づいたディレクトリパス
    experiment_dir = os.path.join(root_experiment_dir, experiment_name)
    
    # サブディレクトリのパス
    fig_and_log_dir = os.path.join(experiment_dir, 'fig_and_log')
    csv_dir = os.path.join(experiment_dir, 'csv')
    
    # ディレクトリの作成
    os.makedirs(fig_and_log_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    
    print(f'Experiment directory created at: {experiment_dir}')
    print(f'fig_and_log directory created at: {fig_and_log_dir}')
    print(f'csv directory created at: {csv_dir}')
    
    return experiment_dir, fig_and_log_dir, csv_dir, experiment_name

def save_alpha_interpolation_results_to_csv(results, csv_dir, experiment_name, epoch):
    """
    alpha_interpolation_test_save_emnist の結果をCSVファイルとして保存します。

    Args:
        results (dict): alpha_interpolation_test_save_emnist の出力結果。
        csv_dir (str): CSVファイルを保存するディレクトリのパス。
        experiment_name (str): 実験名（ファイル名の一部として使用）。
        epoch (int): エポック番号（ファイル名に含める）。
    """
    # データフレームの作成
    df = pd.DataFrame({
        'alpha': results['alpha_values'],
        'predicted_digit': results['predicted_digits'],
        'digit_label_match': results['digit_label_matches'],
    })

    # 必要に応じて、probabilities を分割して保存
    digit_probs_df = pd.DataFrame(results['digit_probabilities'], columns=[f'digit_probability_{i}' for i in range(len(results['digit_probabilities'][0]))])
    
    # データフレームを結合
    full_df = pd.concat([df, digit_probs_df], axis=1)

    # CSVファイルのパスを定義
    csv_path = os.path.join(csv_dir, f'alpha_log_epoch_{epoch}.csv')

    # CSVとして保存
    full_df.to_csv(csv_path, index=False)
    #print(f'Alpha interpolation results saved to {csv_path}')

def save_selected_data_points(fig_and_log_dir, experiment_name, 
                              x1_np, x2_np, idx1, idx2, 
                              orig_label_x1, noisy_label_x1, distance1, 
                              orig_label_x2, noisy_label_x2, distance2, 
                              distance_x1_x2, epoch="0", cmap='gray'):
    """
    選択した2点のデータを画像および.npy形式で保存し、距離情報をログファイルに保存します。

    Args:
        fig_and_log_dir (str): fig_and_log ディレクトリのパス。
        experiment_name (str): 実験名（ファイル名の一部として使用）。
        x1_np (np.ndarray): 1つ目のデータポイントの画像。
        x2_np (np.ndarray): 2つ目のデータポイントの画像。
        idx1 (int): 1つ目のデータポイントのインデックス。
        idx2 (int): 2つ目のデータポイントのインデックス。
        orig_label_x1 (int): 1つ目のデータポイントの元のラベル。
        noisy_label_x1 (int): 1つ目のデータポイントのノイズ後のラベル。
        distance1 (float): 1つ目のデータポイントの重心からの距離。
        orig_label_x2 (int): 2つ目のデータポイントの元のラベル。
        noisy_label_x2 (int): 2つ目のデータポイントのノイズ後のラベル。
        distance2 (float): 2つ目のデータポイントの重心からの距離。
        distance_x1_x2 (float): 2点間の距離。
        epoch (int): エポック番号（ログに含める）。
        cmap (str): 画像表示のカラーマップ。
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # 2つ目のデータポイント
    axes[0].imshow(x2_np.squeeze(), cmap=cmap)
    axes[0].set_title(
        f"Class {orig_label_x2} Data (idx={idx2})\nOri Label: {orig_label_x2}, Noisy Label: {noisy_label_x2}, C_Dis={distance2:.2f}"
    )
    axes[0].axis('off')

    # 1つ目のデータポイント
    axes[1].imshow(x1_np.squeeze(), cmap=cmap)
    axes[1].set_title(
        f"Class {orig_label_x1} Data (idx={idx1})\nOri Label: {orig_label_x1}, Noisy Label: {noisy_label_x1}, C_Dis={distance1:.2f}, Dis_x1_x2={distance_x1_x2:.2f}"
    )
    axes[1].axis('off')

    # 画像の保存
    fig_path = os.path.join(fig_and_log_dir, f"selected_data_points_{idx1}_{idx2}_epoch_{epoch}.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'Selected data points image saved to {fig_path}')

    # データ点の保存
    x1_npy_path = os.path.join(fig_and_log_dir, f"x1_idx{idx1}_epoch_{epoch}.npy")
    x2_npy_path = os.path.join(fig_and_log_dir, f"x2_idx{idx2}_epoch_{epoch}.npy")
    np.save(x1_npy_path, x1_np)
    np.save(x2_npy_path, x2_np)
    print(f'Selected data points saved as .npy files to {fig_and_log_dir}')

    # 距離情報の保存
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
    指定したクラスラベルに属するデータポイントの重心（平均）と分散を計算します。

    Args:
        dataset (np.ndarray): 画像データセット。
        labels (np.ndarray): データセットに対応するラベル。
        class_label (int): 計算対象のクラスラベル。

    Returns:
        tuple: 重心テンソルと分散テンソル。
    """
    indices = np.where(labels == class_label)[0]

    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label}")

    images = torch.tensor(dataset[indices]).float()

    centroid = images.mean(dim=0)
    variance = images.var(dim=0)

    return centroid, variance

def find_closest_data_point_to_centroid(centroid, dataset, labels, class_label, mode='no_noise', noise_info=None):
    """
    指定したクラスラベルに属するデータポイントの中で、重心に最も近いデータポイントを見つけます。

    Args:
        centroid (Tensor): 重心テンソル。
        dataset (np.ndarray): 画像データセット。
        labels (np.ndarray): データセットに対応するラベル。
        class_label (int): 検索対象のクラスラベル。
        mode (str): 'no_noise'（ノイズなし）または 'noise'（ノイズあり）。
        noise_info (np.ndarray): ノイズ情報配列（1がノイズあり、0がノイズなし）。

    Returns:
        tuple: 最も近いデータポイントの画像（numpy.ndarray）、そのインデックス、重心からの距離。
    """
    if mode not in ['no_noise', 'noise']:
        raise ValueError("Mode must be 'no_noise' or 'noise'")

    indices = np.where(labels == class_label)[0]

    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label}")

    if noise_info is not None:
        if mode == 'no_noise':
            indices = indices[noise_info[indices] == 0]
        elif mode == 'noise':
            indices = indices[noise_info[indices] == 1]

    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label} under mode '{mode}'")

    images = torch.tensor(dataset[indices]).float()

    # 重心との距離を計算
    distances = torch.norm(images.view(len(images), -1) - centroid.view(-1), dim=1)

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
    EMNIST用に調整された alpha_interpolation_test_save 関数。

    Args:
        model: 評価に使用するモデル。
        x_clean (Tensor): クリーンなデータポイントの画像テンソル。
        x_noisy (Tensor): ノイズのあるデータポイントの画像テンソル。
        digit_label_x (int): 1つ目のデータポイントのクラスラベル。
        digit_label_y (int): 2つ目のデータポイントのクラスラベル。
        num_digits (int): クラス数（EMNIST digitsは10）。
        device (str): 使用するデバイス（'cuda' または 'cpu'）。

    Returns:
        dict: 補間結果を含む辞書。
    """
    alpha_values = np.arange(-0.5, 1.51, 0.01)
    model.eval()

    x_clean = x_clean.to(device)
    x_noisy = x_noisy.to(device)

    predicted_digits = []
    digit_probabilities = []
    digit_label_matches = []

    with torch.no_grad():  # 推論時に勾配計算を無効化
        for alpha in alpha_values:
            # 補間されたデータを生成
            z = alpha * x_clean + (1 - alpha) * x_noisy
            z = z.unsqueeze(0)  # バッチ次元を追加

            # モデルの出力（ロジット）
            outputs = model(z)  # 形状: [1, num_digits]

            # softmaxを計算
            output_probs = F.softmax(outputs, dim=1)  # 形状: [1, num_digits]

            # 数字の確率を取得
            digit_probs = output_probs.squeeze(0).cpu().numpy()  # [num_digits]
            digit_probabilities.append(digit_probs)

            # 予測ラベルを取得
            predicted_digit = np.argmax(digit_probs)
            predicted_digits.append(predicted_digit)

            # ラベルの一致結果を判定
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
        
    # Set device
    device = set_device(args.gpu)
    print(f'Using device: {device}')

    # ディレクトリの作成
    experiment_dir, fig_and_log_dir, csv_dir, experiment_name = create_experiment_directories(args)

    # モデルの保存ディレクトリを指定
    # ここでは、モデルが保存されているディレクトリのパスを指定してください。
    # あなたのディレクトリ構造に基づき、以下のように設定します。
    # 例: /workspace/save_model/EMNIST/cnn_3layers_width1_emnist_digits_lr0.01_batch_size256_epoch2000_LabelNoiseRate0.1_Optimsgd_momentum0.0
    model_root_dir = '/workspace/save_model/EMNIST'
    model_sub_dir = f'{args.model}_width{args.model_width}_{args.dataset}_lr{args.lr}_batch_size{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_momentum{args.momentum}'
    model_dir = os.path.join(model_root_dir, model_sub_dir)

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory {model_dir} does not exist.")

    # EMNISTデータセットのロード（ノイズなしデータセット）
    no_noise_dir = '/workspace/data/EMNIST/EMNIST_0.0'
    train_images = np.load(os.path.join(no_noise_dir, 'train_images.npy'))
    train_labels = np.load(os.path.join(no_noise_dir, 'train_labels.npy'))
    train_noise_info = np.load(os.path.join(no_noise_dir, 'train_noise_info.npy'))

    # EMNISTデータセットのロード（ノイズありデータセット）
    noisy_dir = os.path.join('/workspace/data/EMNIST', f'EMNIST_{args.label_noise_rate}')
    noisy_train_images = np.load(os.path.join(noisy_dir, 'train_images.npy'))
    noisy_train_labels = np.load(os.path.join(noisy_dir, 'train_labels.npy'))
    noisy_test_images = np.load(os.path.join(noisy_dir, 'test_images.npy'))
    noisy_test_labels = np.load(os.path.join(noisy_dir, 'test_labels.npy'))
    noisy_train_noise_info = np.load(os.path.join(noisy_dir, 'train_noise_info.npy'))

    # モデルの準備（ここではダミーモデルを使用）
    class DummyModel(torch.nn.Module):
        def __init__(self, num_classes=10):
            super(DummyModel, self).__init__()
            self.fc = torch.nn.Linear(28*28, num_classes)
        
        def forward(self, x):
            x = x.view(x.size(0), -1)
            return self.fc(x)
    
    # モデルの初期化
    model = load_models(in_channels=1, args=args, img_size=(32,32), num_classes=10).to(device)
    # モデルファイルのリストを取得（epoch_0.pth, epoch_1.pth, ...）
    model_files = [f for f in os.listdir(model_dir) if f.startswith('epoch_') and f.endswith('.pth')]
    model_files.sort(key=lambda x: int(x.split('_')[1].split('.pth')[0]))  # epoch順にソート

    if not model_files:
        raise FileNotFoundError(f"No model files found in {model_dir}.")

    print(f'Found {len(model_files)} model files in {model_dir}.')

    # 選択するクラスラベル（例）
    n1 = args.n1  # First class label
    n2 = args.n2  # Second class label
    mode1 = args.mode1  # 'no_noise' or 'noise'
    mode2 = args.mode2  # 'no_noise' or 'noise'

    # 重心と分散の計算（ノイズなしデータを使用）
    centroid1, variance1 = compute_class_centroid_and_variance(
        train_images, train_labels, class_label=n1
    )
    centroid2, variance2 = compute_class_centroid_and_variance(
        train_images, train_labels, class_label=n2
    )

    # 最も重心に近いデータポイントを選択（最初のモデルを使用して選択）
    # ここでは、データ選択はモデルに依存しないため、一度だけ行います。
    x1, idx1, distance1 = find_closest_data_point_to_centroid(
        centroid1, train_images, noisy_train_labels, class_label=n1, mode=mode1, noise_info=noisy_train_noise_info
    )
    x2, idx2, distance2 = find_closest_data_point_to_centroid(
        centroid2, train_images, noisy_train_labels, class_label=n2, mode=mode2, noise_info=noisy_train_noise_info
    )

    # 2点間の距離を計算（例: Euclidean distance）
    distance_x1_x2 = np.linalg.norm(x1 - x2)

    # データポイントを保存（エポックごとに異なるため、ループ内で行うことも可能）
    # ここでは、一度だけ保存していますが、必要に応じてループ内に移動してください。
    save_selected_data_points(
         fig_and_log_dir=fig_and_log_dir,
         experiment_name=experiment_name,
         x1_np=x1,
         x2_np=x2,
         idx1=idx1,
         idx2=idx2,
         orig_label_x1=train_labels[idx1],
         noisy_label_x1=noisy_train_labels[idx1],
         distance1=distance1,
         orig_label_x2=train_labels[idx2],
         noisy_label_x2=noisy_train_labels[idx2],
         distance2=distance2,
         distance_x1_x2=distance_x1_x2
     )

    # 各エポックのモデルに対してαテストを実行
    for model_file in model_files:
        # エポック番号の抽出
        epoch_num = int(model_file.split('_')[1].split('.pth')[0])

        # モデルのパス
        model_path = os.path.join(model_dir, model_file)
        #print(f'Loading model from {model_path}')

        # モデルの重みをロード
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()  # 評価モードに設定

        # alpha_interpolation_test_save_emnist の実行
        # EMNIST用に調整された関数を使用
        x_clean = torch.tensor(x1).float()
        x_noisy = torch.tensor(x2).float()
        results = alpha_interpolation_test_save_emnist(
            model=model,
            x_clean=x_clean,
            x_noisy=x_noisy,
            digit_label_x=n1,
            digit_label_y=n2,
            num_digits=10,
            device=device
        )

        # 結果をCSVに保存
        save_alpha_interpolation_results_to_csv(results, csv_dir, experiment_name, epoch_num)

        # 必要に応じてメモリをクリア
        clear_memory()

    #print("Alpha interpolation tests for all epochs completed.")

if __name__ == "__main__":
    main()
