# train.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import os
import csv
from sklearn.metrics import classification_report, accuracy_score
from utils import clear_memory
from datasets import compute_distances_between_indices  # Assuming it's moved here
import math


#lossの計算だけ変更しよう。soft label化が必要ない
#setupへの送る結果の種類を変える必要がある
def alpha_interpolation_test_save(
    model, x_clean, x_noisy,
    digit_label_x, digit_label_y,
    color_label_x, color_label_y,
    combined_label_x, combined_label_y,
    num_digits, num_colors, device
):
    import torch
    import torch.nn.functional as F

    # モデルをGPUに移動し評価モードに
    model = model.to(device)
    model.eval()

    # alpha_valuesを作成 (例: -0.5から1.5まで0.01刻み)
    alpha_values = torch.arange(-0.5, 1.51, 0.01, device=device)  # shape: (N,)
    N = alpha_values.shape[0]

    # x_clean, x_noisyは (C, H, W) と仮定。まずバッチ次元を追加して (1, C, H, W) にする
    x_clean = x_clean.to(device).unsqueeze(0)
    x_noisy = x_noisy.to(device).unsqueeze(0)

    # バッチ処理のため、x_clean, x_noisyを各αに合わせて展開 (N, C, H, W)
    x_clean_batch = x_clean.expand(N, *x_clean.shape[1:])
    x_noisy_batch = x_noisy.expand(N, *x_noisy.shape[1:])

    # alpha_valuesを (N,1,1,1) に変換してブロードキャスト計算
    alphas = alpha_values.view(-1, 1, 1, 1)
    # 一括で補間データを作成: z = α * x_clean + (1 - α) * x_noisy
    z = alphas * x_clean_batch + (1 - alphas) * x_noisy_batch  # shape: (N, C, H, W)

    with torch.no_grad():
        # まとめてモデル推論
        outputs = model(z)  # shape: (N, num_classes)
        output_probs = F.softmax(outputs, dim=1)  # shape: (N, num_classes)

    # 予測ラベルの計算（バッチ処理）
    predicted_combined_tensor = torch.argmax(output_probs, dim=1)  # shape: (N,)
    predicted_digits_tensor = predicted_combined_tensor // num_colors
    predicted_colors_tensor = predicted_combined_tensor % num_colors

    # ラベルマッチング判定
    digit_label_matches_tensor = torch.where(
        predicted_digits_tensor == digit_label_x,
        torch.tensor(1, device=device),
        torch.tensor(0, device=device)
    )
    digit_label_matches_tensor = torch.where(
        predicted_digits_tensor == digit_label_y,
        torch.tensor(-1, device=device),
        digit_label_matches_tensor
    )
    color_label_matches_tensor = torch.where(
        predicted_colors_tensor == color_label_x,
        torch.tensor(1, device=device),
        torch.tensor(0, device=device)
    )
    color_label_matches_tensor = torch.where(
        predicted_colors_tensor == color_label_y,
        torch.tensor(-1, device=device),
        color_label_matches_tensor
    )

    # 出力確率からdigit, colorごとの確率を計算
    output_probs_reshaped = output_probs.view(N, num_digits, num_colors)
    digit_probabilities_tensor = output_probs_reshaped.sum(dim=2)  # shape: (N, num_digits)
    color_probabilities_tensor = output_probs_reshaped.sum(dim=1)  # shape: (N, num_colors)

    # 最終結果のみをCPUに転送し、リストに変換
    return {
        'alpha_values': alpha_values.cpu().numpy().tolist(),
        'predicted_digits': predicted_digits_tensor.cpu().numpy().tolist(),
        'predicted_colors': predicted_colors_tensor.cpu().numpy().tolist(),
        'predicted_combined': predicted_combined_tensor.cpu().numpy().tolist(),
        'digit_probabilities': digit_probabilities_tensor.cpu().numpy().tolist(),
        'color_probabilities': color_probabilities_tensor.cpu().numpy().tolist(),
        'raw_probabilities': output_probs.cpu().numpy().tolist(),
        'digit_label_matches': digit_label_matches_tensor.cpu().numpy().tolist(),
        'color_label_matches': color_label_matches_tensor.cpu().numpy().tolist()
    }





def alpha_interpolation_test(
    model, x_clean, x_noisy,
    digit_label_x, digit_label_y,
    color_label_x, color_label_y,
    combined_label_x, combined_label_y,
    num_digits, num_colors, device
):
    import torch.nn.functional as F
    alpha_values = np.arange(-0.5, 1.6, 0.01)
    model.eval()

    x_clean = x_clean.to(device)
    x_noisy = x_noisy.to(device)

    digit_losses = []
    color_losses = []
    combined_losses = []
    predicted_digits = []
    predicted_colors = []
    predicted_combined = []
    digit_probabilities = []
    color_probabilities = []
    digit_label_matches = []  # 追加
    color_label_matches = []  # 追加

    for alpha in alpha_values:
        # 補間されたデータを生成
        z = alpha * x_clean + (1 - alpha) * x_noisy
        z = z.unsqueeze(0)  # バッチ次元を追加

        # モデルの出力（ロジット）
        outputs = model(z)  # 形状: [1, num_digits * num_colors]

        # softmaxを計算
        output_probs = F.softmax(outputs, dim=1)  # 形状: [1, num_digits * num_colors]

        # 数字と色の確率を取得
        output_probs_reshaped = output_probs.view(1, num_digits, num_colors)
        digit_probs = output_probs_reshaped.sum(dim=2).squeeze(0)  # [num_digits]
        color_probs = output_probs_reshaped.sum(dim=1).squeeze(0)  # [num_colors]

        # ソフトラベルを作成
        soft_digit_label = torch.zeros(num_digits, device=device)
        soft_digit_label[digit_label_x] = alpha
        soft_digit_label[digit_label_y] = 1 - alpha

        soft_color_label = torch.zeros(num_colors, device=device)
        soft_color_label[color_label_x] = alpha
        soft_color_label[color_label_y] = 1 - alpha

        soft_combined_label = torch.zeros(num_digits * num_colors, device=device)
        soft_combined_label[combined_label_x] = alpha
        soft_combined_label[combined_label_y] = 1 - alpha

        # 損失を計算
        digit_loss = -torch.sum(soft_digit_label * torch.log(digit_probs + 1e-8))
        color_loss = -torch.sum(soft_color_label * torch.log(color_probs + 1e-8))
        combined_loss = -torch.sum(soft_combined_label * torch.log(output_probs.squeeze(0) + 1e-8))

        # 予測ラベルを取得
        predicted_combined_label = torch.argmax(output_probs, dim=1).item()
        predicted_combined.append(predicted_combined_label)
        predicted_digit = predicted_combined_label // num_colors
        predicted_color = predicted_combined_label % num_colors
        predicted_digits.append(predicted_digit)
        predicted_colors.append(predicted_color)

        # ラベルの一致結果を判定
        if predicted_digit == digit_label_x:
            digit_label_matches.append(1)
        elif predicted_digit == digit_label_y:
            digit_label_matches.append(-1)
        else:
            digit_label_matches.append(0)

        if predicted_color == color_label_x:
            color_label_matches.append(1)
        elif predicted_color == color_label_y:
            color_label_matches.append(-1)
        else:
            color_label_matches.append(0)

        # ログを保存
        digit_losses.append(digit_loss.item())
        color_losses.append(color_loss.item())
        combined_losses.append(combined_loss.item())
        digit_probabilities.append(digit_probs.detach().cpu().numpy())  # NumPy 配列に変換
        color_probabilities.append(color_probs.detach().cpu().numpy())  # NumPy 配列に変換

    return {
        'alpha_values': alpha_values.tolist(),
        'digit_losses': digit_losses,
        'color_losses': color_losses,
        'combined_losses': combined_losses,
        'predicted_digits': predicted_digits,
        'predicted_colors': predicted_colors,
        'predicted_combined': predicted_combined,
        'digit_probabilities': digit_probabilities,
        'color_probabilities': color_probabilities,
        'digit_label_matches': digit_label_matches,  # 追加
        'color_label_matches': color_label_matches   # 追加
    }


def select_data_points(original_targets, noisy_targets, noise_info, num_colors, train_dataset, max_points=5):
    """
    Select data points (x, y) where the original digit labels are the same,
    but after label noise, the digit labels are different. Also compute and
    display distances.

    Returns:
        tuple: Indices of data points (idx_clean, idx_noisy) and their distance.
    """
    original_digit_labels = original_targets // num_colors
    noisy_digit_labels = noisy_targets // num_colors
    n = 5  # You can choose any digit label
    mode = 0  # 0 for closest, 1 for farthest

    indices_with_label_n = torch.where(original_digit_labels == n)[0]
    clean_indices = indices_with_label_n[noise_info[indices_with_label_n] == 0]  # Clean indices with digit label n
    noisy_indices = indices_with_label_n[noise_info[indices_with_label_n] == 1]  # Noisy indices with digit label n

    distance_pair = []
    for i in range(min(len(clean_indices), max_points)):
        clean_index = [clean_indices[i].item()]
        if len(clean_index) > 0 and len(noisy_indices) > 0:
            distances_results = compute_distances_between_indices(train_dataset, [clean_index], noisy_indices.tolist(), mode)
            # Ensure that selected indices are different
            if clean_index[0] in noisy_indices:
                continue
            # Extract the result
            clean_group, result_tuple = next(iter(distances_results.items()))
            label, idx_noisy, distance = result_tuple if len(result_tuple) == 3 else (None, None, None)
            idx_clean = clean_group[0]
            print(f"Selected pair indices (clean: {idx_clean}, noisy: {idx_noisy}), Distance: {distance}")
            # Save the pair (you can add code to save images if needed)
            distance_pair.append((idx_clean, idx_noisy, distance))

    # Return the pair with the smallest distance
    if distance_pair:
        min_pair = min(distance_pair, key=lambda x: x[2])
        idx_clean, idx_noisy, distance = min_pair
        print(f"Best pair selected with distance: {distance}")
        return idx_clean, idx_noisy, distance
    else:
        print("No data points satisfying the conditions were found.")
        return (None, None, None)

def add_label_noise(targets, label_noise_rate, num_digits, num_colors):
    """
    Add label noise to the targets.

    Args:
        targets (torch.Tensor): Original labels.
        label_noise_rate (float): Fraction of labels to corrupt.
        num_digits (int): Number of digit classes.
        num_colors (int): Number of color classes.

    Returns:
        tuple: Noisy targets and noise information tensor.
    """
    noisy_targets = targets.clone()
    num_noisy = int(label_noise_rate * len(targets))
    noisy_indices = torch.randperm(len(targets))[:num_noisy]
    noise_info = torch.zeros(len(targets), dtype=torch.int)  # Initialize as clean

    if num_digits == 10 and num_colors == 1:
        for idx in noisy_indices:
            original_label = targets[idx].item()
            new_label = random.randint(0, num_digits - 1)
            while new_label == original_label:
                new_label = random.randint(0, num_digits - 1)
            noisy_targets[idx] = new_label
            noise_info[idx] = 1  # Mark as noisy

    elif num_digits == 10 and num_colors == 10:
        for idx in noisy_indices:
            original_label = targets[idx].item()
            original_digit = original_label // num_colors
            original_color = original_label % num_colors

            new_digit = random.randint(0, num_digits - 1)
            new_color = random.randint(0, num_colors - 1)
            new_label = new_digit * num_colors + new_color
            while new_label == original_label:
                new_digit = random.randint(0, num_digits - 1)
                new_color = random.randint(0, num_colors - 1)
                new_label = new_digit * num_colors + new_color

            noisy_targets[idx] = new_label
            noise_info[idx] = 1  # Mark as noisy

    return noisy_targets, noise_info

import torch
import numpy as np

# ===========================
# Train: 通常用
# ===========================
# --- 修正版: train_model_standard ---
import torch
import numpy as np

def train_model_standard(
    model, train_loader, optimizer,
    criterion_noisy, criterion_clean,
    weight_noisy, weight_clean,
    device,
    epoch_batch=None,
    experiment_name=None,
    args=None
):    
    model.train()
    running_loss = 0.0
    total_samples = 0
    correct_total = 0

    correct_noisy = 0
    total_noisy = 0
    correct_clean = 0
    total_clean = 0

    running_loss_noisy = 0.0
    running_loss_clean = 0.0

    criterion_noisy.reduction = 'none'
    criterion_clean.reduction = 'none'
    batch_idx=0
    for inputs, labels, noise_flags in train_loader:
        inputs, labels, noise_flags = inputs.to(device), labels.to(device), noise_flags.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_total += (predicted == labels).sum().item()

        idx_noisy = (noise_flags == 1)
        idx_clean = (noise_flags == 0)

        loss_per_sample = torch.zeros(labels.size(0), device=device)

        if idx_noisy.sum() > 0:
            loss_noisy = criterion_noisy(outputs[idx_noisy], labels[idx_noisy]) * weight_noisy
            loss_per_sample[idx_noisy] = loss_noisy
            running_loss_noisy += loss_noisy.mean().item()
            correct_noisy += (predicted[idx_noisy] == labels[idx_noisy]).sum().item()
            total_noisy += idx_noisy.sum().item()

        if idx_clean.sum() > 0:
            loss_clean = criterion_clean(outputs[idx_clean], labels[idx_clean]) * weight_clean
            loss_per_sample[idx_clean] = loss_clean
            running_loss_clean += loss_clean.mean().item()
            correct_clean += (predicted[idx_clean] == labels[idx_clean]).sum().item()
            total_clean += idx_clean.sum().item()
        if epoch_batch is not None and epoch_batch <= 3 and experiment_name is not None and args is not None:
            
            save_dir = os.path.join(
            "save_model", args.dataset, f"noise_{args.label_noise_rate}",
            experiment_name, "csv", "batch_log", f"epoch_{epoch_batch}")
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "inputs": inputs.detach().cpu(),
                "labels": labels.detach().cpu(),
                "noise_flags": noise_flags.detach().cpu(),
                "predicted": predicted.detach().cpu(),
                "loss_per_sample": loss_per_sample.detach().cpu()
            }, os.path.join(save_dir, f"batch_{batch_idx}.pt"))
            batch_idx+=1
        loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

    avg_loss = running_loss / total_samples
    avg_loss_noisy = running_loss_noisy / total_noisy if total_noisy > 0 else float('nan')
    avg_loss_clean = running_loss_clean / total_clean if total_clean > 0 else float('nan')
    accuracy_total = 100. * correct_total / total_samples
    accuracy_noisy = 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan')
    accuracy_clean = 100. * correct_clean / total_clean if total_clean > 0 else float('nan')

    return {
        "avg_loss": avg_loss,
        "train_accuracy": accuracy_total,
        "train_accuracy_noisy": accuracy_noisy,
        "train_accuracy_clean": accuracy_clean,
        "avg_loss_noisy": avg_loss_noisy,
        "avg_loss_clean": avg_loss_clean,
        "total_samples": total_samples,
        "total_noisy": total_noisy,
        "total_clean": total_clean,
        "correct_total": correct_total,
        "correct_noisy": correct_noisy,
        "correct_clean": correct_clean,
        "train_error_total": 100 - accuracy_total,
        "train_error_noisy": 100 - accuracy_noisy if total_noisy > 0 else float('nan'),
        "train_error_clean": 100 - accuracy_clean if total_clean > 0 else float('nan'),
    }

def train_model_distr_colored(model, train_loader, optimizer, criterion_noisy, criterion_clean,
                              weight_noisy, weight_clean, device, num_colors, num_digits):
    model.train()
    total_samples = 0
    correct_total = 0
    correct_noisy = 0
    correct_clean = 0
    correct_digit_total = 0
    correct_color_total = 0
    correct_digit_noisy = 0
    correct_color_noisy = 0
    correct_digit_clean = 0
    correct_color_clean = 0
    total_noisy = 0
    total_clean = 0
    loss_values = []
    loss_values_noisy = []
    loss_values_clean = []

    criterion_noisy.reduction = 'none'
    criterion_clean.reduction = 'none'

    for inputs, labels, noise_flags in train_loader:
        inputs, labels, noise_flags = inputs.to(device), labels.to(device), noise_flags.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)

        batch_size = labels.size(0)
        total_samples += batch_size
        correct_total += (predicted == labels).sum().item()

        digit_labels = labels // num_colors
        color_labels = labels % num_colors
        digit_preds = predicted // num_colors
        color_preds = predicted % num_colors
        correct_digit_total += (digit_preds == digit_labels).sum().item()
        correct_color_total += (color_preds == color_labels).sum().item()

        idx_noisy = (noise_flags == 1)
        idx_clean = (noise_flags == 0)

        loss_per_sample = torch.zeros(labels.size(0), device=device)

        if idx_noisy.sum() > 0:
            out_n, lbl_n = outputs[idx_noisy], labels[idx_noisy]
            l_n = criterion_noisy(out_n, lbl_n) * weight_noisy
            loss_per_sample[idx_noisy] = l_n
            loss_values_noisy.extend(l_n.detach().cpu().numpy())
            correct_noisy += (predicted[idx_noisy] == lbl_n).sum().item()
            total_noisy += idx_noisy.sum().item()

            digit_lbl_n = lbl_n // num_colors
            color_lbl_n = lbl_n % num_colors
            digit_pred_n = predicted[idx_noisy] // num_colors
            color_pred_n = predicted[idx_noisy] % num_colors
            correct_digit_noisy += (digit_pred_n == digit_lbl_n).sum().item()
            correct_color_noisy += (color_pred_n == color_lbl_n).sum().item()

        if idx_clean.sum() > 0:
            out_c, lbl_c = outputs[idx_clean], labels[idx_clean]
            l_c = criterion_clean(out_c, lbl_c) * weight_clean
            loss_per_sample[idx_clean] = l_c
            loss_values_clean.extend(l_c.detach().cpu().numpy())
            correct_clean += (predicted[idx_clean] == lbl_c).sum().item()
            total_clean += idx_clean.sum().item()

            digit_lbl_c = lbl_c // num_colors
            color_lbl_c = lbl_c % num_colors
            digit_pred_c = predicted[idx_clean] // num_colors
            color_pred_c = predicted[idx_clean] % num_colors
            correct_digit_clean += (digit_pred_c == digit_lbl_c).sum().item()
            correct_color_clean += (color_pred_c == color_lbl_c).sum().item()

        loss_values.extend(loss_per_sample.detach().cpu().numpy())
        loss = loss_per_sample.mean()
        loss.backward()
        optimizer.step()

    avg_loss = np.mean(loss_values) if loss_values else float('nan')
    avg_loss_noisy = np.mean(loss_values_noisy) if loss_values_noisy else float('nan')
    avg_loss_clean = np.mean(loss_values_clean) if loss_values_clean else float('nan')

    return {
        "avg_loss": avg_loss,
        "accuracy_total": 100. * correct_total / total_samples,
        "accuracy_noisy": 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan'),
        "accuracy_clean": 100. * correct_clean / total_clean if total_clean > 0 else float('nan'),
        "avg_loss_noisy": avg_loss_noisy,
        "avg_loss_clean": avg_loss_clean,
        "total_samples": total_samples,
        "total_noisy": total_noisy,
        "total_clean": total_clean,
        "correct_total": correct_total,
        "correct_noisy": correct_noisy,
        "correct_clean": correct_clean,
        "accuracy_digit_total": 100. * correct_digit_total / total_samples,
        "accuracy_color_total": 100. * correct_color_total / total_samples,
        "accuracy_digit_noisy": 100. * correct_digit_noisy / total_noisy if total_noisy > 0 else float('nan'),
        "accuracy_color_noisy": 100. * correct_color_noisy / total_noisy if total_noisy > 0 else float('nan'),
        "accuracy_digit_clean": 100. * correct_digit_clean / total_clean if total_clean > 0 else float('nan'),
        "accuracy_color_clean": 100. * correct_color_clean / total_clean if total_clean > 0 else float('nan'),
        "train_error_total": 100. - (100. * correct_total / total_samples),
        "train_error_noisy": 100. - (100. * correct_noisy / total_noisy) if total_noisy > 0 else float('nan'),
        "train_error_clean": 100. - (100. * correct_clean / total_clean) if total_clean > 0 else float('nan'),
    }

# 残りの 0 epoch 版・test_model_* も必要であれば続けて出力します。

# ===========================
# Train: 0エポック
# ===========================
# --- 修正後: train_model_0_epoch_standard ---
def train_model_0_epoch_standard(model, train_loader, optimizer, criterion_noisy, criterion_clean, device):
    model.eval()
    total_samples = 0
    correct_total = 0
    correct_noisy = 0
    total_noisy = 0
    correct_clean = 0
    total_clean = 0
    running_loss = 0.0
    running_loss_noisy = 0.0
    running_loss_clean = 0.0

    # 損失関数を per-sample モードに設定
    criterion_noisy.reduction = 'none'
    criterion_clean.reduction = 'none'

    with torch.no_grad():
        for inputs, labels, noise_flags in train_loader:
            inputs, labels, noise_flags = inputs.to(device), labels.to(device), noise_flags.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total_samples += labels.size(0)
            correct_total += (predicted == labels).sum().item()

            # noisy / clean サンプルのインデックス
            idx_noisy = (noise_flags == 1)
            idx_clean = (noise_flags == 0)

            # 各サンプルの損失を保持するテンソル
            per_sample_loss = torch.zeros(labels.size(0), device=device)

            # ノイズありの処理
            if idx_noisy.sum() > 0:
                outputs_noisy = outputs[idx_noisy]
                labels_noisy = labels[idx_noisy]
                loss_noisy = criterion_noisy(outputs_noisy, labels_noisy)
                per_sample_loss[idx_noisy] = loss_noisy

                running_loss_noisy += loss_noisy.mean().item()
                correct_noisy += (predicted[idx_noisy] == labels_noisy).sum().item()
                total_noisy += idx_noisy.sum().item()

            # ノイズなしの処理
            if idx_clean.sum() > 0:
                outputs_clean = outputs[idx_clean]
                labels_clean = labels[idx_clean]
                loss_clean = criterion_clean(outputs_clean, labels_clean)
                per_sample_loss[idx_clean] = loss_clean

                running_loss_clean += loss_clean.mean().item()
                correct_clean += (predicted[idx_clean] == labels_clean).sum().item()
                total_clean += idx_clean.sum().item()

            # 全体の平均損失
            loss = per_sample_loss.mean()
            running_loss += loss.item()

    # 統計指標の計算
    avg_loss = running_loss / len(train_loader)
    avg_loss_noisy = running_loss_noisy / total_noisy if total_noisy > 0 else float('nan')
    avg_loss_clean = running_loss_clean / total_clean if total_clean > 0 else float('nan')
    accuracy_total = 100. * correct_total / total_samples
    accuracy_noisy = 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan')
    accuracy_clean = 100. * correct_clean / total_clean if total_clean > 0 else float('nan')
    error_total = 100. - accuracy_total
    error_noisy = 100. - accuracy_noisy if total_noisy > 0 else float('nan')
    error_clean = 100. - accuracy_clean if total_clean > 0 else float('nan')

    return {
        "avg_loss": avg_loss,
        "train_accuracy": accuracy_total,
        "train_accuracy_noisy": accuracy_noisy,
        "train_accuracy_clean": accuracy_clean,
        "avg_loss_noisy": avg_loss_noisy,
        "avg_loss_clean": avg_loss_clean,
        "total_samples": total_samples,
        "total_noisy": total_noisy,
        "total_clean": total_clean,
        "correct_total": correct_total,
        "correct_noisy": correct_noisy,
        "correct_clean": correct_clean,
        "train_error_total": error_total,
        "train_error_noisy": error_noisy,
        "train_error_clean": error_clean
    }
# --- 修正後: train_model_0_epoch_distr_colored ---
def train_model_0_epoch_distr_colored(model, train_loader, optimizer, criterion, weight_noisy, weight_clean, device, num_colors, num_digits):
    model.eval()
    running_loss = 0.0
    total_samples = 0
    correct_total = 0
    correct_noisy = 0
    total_noisy = 0
    correct_clean = 0
    total_clean = 0
    correct_digit_total = 0
    correct_color_total = 0
    correct_digit_noisy = 0
    correct_color_noisy = 0
    correct_digit_clean = 0
    correct_color_clean = 0
    loss_values = []
    loss_values_noisy = []
    loss_values_clean = []

    criterion.reduction = 'none'

    with torch.no_grad():
        for inputs, labels, noise_labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            noise_labels = noise_labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            batch_size = labels.size(0)
            total_samples += batch_size

            idx_noisy = (noise_labels == 1)
            idx_clean = (noise_labels == 0)
            num_noisy = idx_noisy.sum().item()
            num_clean = idx_clean.sum().item()

            per_sample_loss = criterion(outputs, labels)
            total_weight = weight_clean + weight_noisy
            weights = torch.zeros_like(per_sample_loss, device=device)

            if num_noisy == 0:
                weights.fill_((weight_clean / total_weight) * 2)
            elif num_clean == 0:
                weights.fill_((weight_noisy / total_weight) * 2)
            else:
                weights[idx_noisy] = (weight_noisy / total_weight) * 2
                weights[idx_clean] = (weight_clean / total_weight) * 2

            per_sample_loss_weighted = per_sample_loss * weights

            loss = per_sample_loss_weighted.mean()
            running_loss += loss.item() * batch_size
            correct_total += (predicted == labels).sum().item()

            digit_labels = labels // num_colors
            color_labels = labels % num_colors
            digit_predictions = predicted // num_colors
            color_predictions = predicted % num_colors

            correct_digit_total += (digit_predictions == digit_labels).sum().item()
            correct_color_total += (color_predictions == color_labels).sum().item()

            if num_noisy > 0:
                labels_noisy = labels[idx_noisy]
                predicted_noisy = predicted[idx_noisy]
                correct_noisy += (predicted_noisy == labels_noisy).sum().item()
                total_noisy += num_noisy

                digit_labels_noisy = labels_noisy // num_colors
                color_labels_noisy = labels_noisy % num_colors
                digit_predictions_noisy = predicted_noisy // num_colors
                color_predictions_noisy = predicted_noisy % num_colors

                correct_digit_noisy += (digit_predictions_noisy == digit_labels_noisy).sum().item()
                correct_color_noisy += (color_predictions_noisy == color_labels_noisy).sum().item()
                loss_values_noisy.extend(per_sample_loss_weighted[idx_noisy].detach().cpu().numpy())

            if num_clean > 0:
                labels_clean = labels[idx_clean]
                predicted_clean = predicted[idx_clean]
                correct_clean += (predicted_clean == labels_clean).sum().item()
                total_clean += num_clean

                digit_labels_clean = labels_clean // num_colors
                color_labels_clean = labels_clean % num_colors
                digit_predictions_clean = predicted_clean // num_colors
                color_predictions_clean = predicted_clean % num_colors

                correct_digit_clean += (digit_predictions_clean == digit_labels_clean).sum().item()
                correct_color_clean += (color_predictions_clean == color_labels_clean).sum().item()
                loss_values_clean.extend(per_sample_loss_weighted[idx_clean].detach().cpu().numpy())

            loss_values.extend(per_sample_loss_weighted.detach().cpu().numpy())

    avg_loss = np.mean(loss_values) if loss_values else float('nan')
    var_loss = np.var(loss_values) if loss_values else float('nan')
    avg_loss_noisy = np.mean(loss_values_noisy) if loss_values_noisy else float('nan')
    var_loss_noisy = np.var(loss_values_noisy) if loss_values_noisy else float('nan')
    avg_loss_clean = np.mean(loss_values_clean) if loss_values_clean else float('nan')
    var_loss_clean = np.var(loss_values_clean) if loss_values_clean else float('nan')

    metrics = {
        'avg_loss': avg_loss,
        'var_loss': var_loss,
        'accuracy_total': 100. * correct_total / total_samples if total_samples > 0 else float('nan'),
        'accuracy_noisy': 100. * correct_noisy / total_noisy if total_noisy > 0 else float('nan'),
        'accuracy_clean': 100. * correct_clean / total_clean if total_clean > 0 else float('nan'),
        'avg_loss_noisy': avg_loss_noisy,
        'var_loss_noisy': var_loss_noisy,
        'avg_loss_clean': avg_loss_clean,
        'var_loss_clean': var_loss_clean,
        'accuracy_digit_total': 100. * correct_digit_total / total_samples if total_samples > 0 else float('nan'),
        'accuracy_color_total': 100. * correct_color_total / total_samples if total_samples > 0 else float('nan'),
        'accuracy_digit_noisy': 100. * correct_digit_noisy / total_noisy if total_noisy > 0 else float('nan'),
        'accuracy_color_noisy': 100. * correct_color_noisy / total_noisy if total_noisy > 0 else float('nan'),
        'accuracy_digit_clean': 100. * correct_digit_clean / total_clean if total_clean > 0 else float('nan'),
        'accuracy_color_clean': 100. * correct_color_clean / total_clean if total_clean > 0 else float('nan'),
        'total_samples': total_samples,
        'total_noisy': total_noisy,
        'total_clean': total_clean,
        'correct_total': correct_total,
        'correct_noisy': correct_noisy,
        'correct_clean': correct_clean
    }

    return metrics



# ===========================
# Test: 通常
# ===========================
def test_model_standard(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    error = 100. - accuracy

    return {
        "avg_loss": avg_loss,
        "test_accuracy": accuracy,
        "test_total_samples": total,
        "test_correct": correct,
        "test_error": error
    }

# ===========================
# Test: 分解型（digit/color）
# ===========================
def test_model_distr_colored(model, test_loader, device, num_colors, num_digits):
    model.eval()
    correct = 0
    total = 0
    correct_digit = 0
    correct_color = 0
    test_loss = 0.0
    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            digit_labels = labels // num_colors
            color_labels = labels % num_colors
            digit_preds = predicted // num_colors
            color_preds = predicted % num_colors

            correct_digit += (digit_preds == digit_labels).sum().item()
            correct_color += (color_preds == color_labels).sum().item()

    return {
        "avg_loss": test_loss / len(test_loader),
        "accuracy_total": 100. * correct / total,
        "accuracy_digit_total": 100. * correct_digit / total,
        "accuracy_color_total": 100. * correct_color / total,
        "total_samples": total,
        "correct": correct,
        "error": 100. - 100. * correct / total
    }



def compute_class_centroid_and_variance(dataset, labels, class_label):
    """
    Compute the centroid (mean) and variance of the data points belonging to a specific class label.

    Args:
        dataset (Dataset): The dataset containing the images.
        labels (Tensor): Tensor of labels corresponding to the dataset.
        class_label (int): The class label for which to compute the centroid and variance.

    Returns:
        tuple: Centroid tensor and variance tensor.
    """
    indices = torch.where(labels == class_label)[0]

    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label}")

    images = torch.stack([dataset[i][0] for i in indices])

    centroid = images.mean(dim=0)
    variance = images.var(dim=0)

    return centroid, variance

def find_closest_data_point_to_centroid(centroid, dataset, labels, class_label, mode='no_noise', noise_info=None):
    """
    Find the data point with the specified class label that is closest to the centroid.

    Args:
        centroid (Tensor): The centroid tensor computed from compute_class_centroid_and_variance.
        dataset (Dataset): The dataset containing the images.
        labels (Tensor): Tensor of labels corresponding to the dataset.
        class_label (int): The class label to search for.
        mode (str): 'no_noise' to consider clean data points, 'noise' to consider noisy data points.
        noise_info (Tensor): Tensor indicating which data points are noisy (1) or clean (0).

    Returns:
        tuple: The closest data point tensor, its index, and the distance to the centroid.
    """
    if mode not in ['no_noise', 'noise']:
        raise ValueError("Mode must be 'no_noise' or 'noise'")

    indices = torch.where(labels == class_label)[0]

    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label}")

    if noise_info is not None:
        if mode == 'no_noise':
            indices = indices[noise_info[indices] == 0]
        elif mode == 'noise':
            indices = indices[noise_info[indices] == 1]

    if len(indices) == 0:
        raise ValueError(f"No data points found for class label {class_label} under mode '{mode}'")

    images = torch.stack([dataset[i][0] for i in indices])

    distances = torch.norm(images.view(len(images), -1) - centroid.view(-1), dim=1)

    min_distance, min_index = torch.min(distances, dim=0)

    closest_data_point = dataset[indices[min_index]][0]
    data_point_index = indices[min_index].item()

    return closest_data_point, data_point_index, min_distance.item()


def select_n2(n1, idx1, target, mode, original_targets, y_train_noisy, noise_info, train_dataset):
    """
    n1を基にn2を選択する関数。

    Args:
        n1 (torch.Tensor): データ点n1のテンソル (C, H, W)。
        idx1 (int): n1が格納されているインデックス。
        target (str): 'color', 'digit', 'combined' のいずれか。
        mode (str): 'noise' または 'no_noise'。
        original_targets (torch.Tensor または np.ndarray): ノイズ付与前のラベル情報。
        y_train_noisy (torch.Tensor または np.ndarray): ノイズ付与後のラベル情報。
        noise_info (torch.Tensor または np.ndarray): ノイズが付与されたかのフラグ (1: ノイズ有, 0: ノイズ無)。
        train_dataset (torch.utils.data.Dataset): トレーニングデータセット。

    Returns:
        tuple: (n2, idx2) - 選択されたデータ点n2とそのインデックス。
               条件を満たすn2が存在しない場合は (None, None) を返す。
    """
    
    # NumPy配列の場合はテンソルに変換
    if isinstance(original_targets, np.ndarray):
        original_targets = torch.from_numpy(original_targets)
    if isinstance(y_train_noisy, np.ndarray):
        y_train_noisy = torch.from_numpy(y_train_noisy)
    if isinstance(noise_info, np.ndarray):
        noise_info = torch.from_numpy(noise_info)

    # ターゲットに応じてラベルを抽出
    if target == 'digit':
        # 10の位がdigitラベル
        n1_label = original_targets[idx1] // 10
        labels_original = original_targets // 10
        labels_noisy = y_train_noisy // 10
    elif target == 'color':
        # 1の位がcolorラベル
        n1_label = original_targets[idx1] % 10
        labels_original = original_targets % 10
        labels_noisy = y_train_noisy % 10
    elif target == 'combined':
        # combinedラベル
        n1_label = original_targets[idx1]
        labels_original = original_targets
        labels_noisy = y_train_noisy
    else:
        raise ValueError("Invalid target. Must be one of ['color', 'digit', 'combined'].")

    # モードに応じて条件を設定
    if mode == 'noise':
        # ノイズモード: 元のラベルはn1_labelだが、ノイズ後に異なるラベルに変更されたデータ点
        condition = (labels_original == n1_label) & (labels_noisy != labels_original)
    elif mode == 'no_noise':
        # ノイズなしモード: 元のラベルがn1_labelで、ノイズが付与されていないデータ点
        condition = (labels_original == n1_label) & (noise_info == 0)
    else:
        raise ValueError("Invalid mode. Must be one of ['noise', 'no_noise'].")

    # 条件を満たすインデックスを取得し、n1自身のインデックスを除外
    candidate_indices = torch.where(condition)[0]
    candidate_indices = candidate_indices[candidate_indices != idx1]

    # 条件を満たすデータ点が存在しない場合
    if candidate_indices.numel() == 0:
        print("条件を満たすn2が見つかりませんでした。")
        return None, None

    # 条件を満たすデータ点を取得
    # データセットがインデックスでアクセス可能であることを前提としています
    candidates = torch.stack([train_dataset[i][0] for i in candidate_indices])

    # n1と各候補データ点のユークリッド距離を計算
    # n1と候補をフラット化して計算
    n1_flat = n1.view(-1)
    candidates_flat = candidates.view(candidates.size(0), -1)

    # 各候補との距離を計算
    distances = torch.norm(candidates_flat - n1_flat, p=2, dim=1)

    # 最小距離のインデックスを取得
    min_dist, min_idx = torch.min(distances, dim=0)
    selected_idx = candidate_indices[min_idx].item()
    n2 = train_dataset[selected_idx][0]

    return n2, selected_idx,min_dist


