# logger.py

import os
import csv
import wandb
import math
import numpy as np

def setup_wandb(args, experiment_name):
    """
    Initialize WandB run.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.
        experiment_name (str): Name of the experiment.

    Returns:
        wandb.Run: The initialized WandB run.
    """
    if args.wandb:
        run = wandb.init(
            project=args.wandb_project,
            name=experiment_name,
            entity=args.wandb_entity,
            config=args
        )
        return run
    return None


def setup_alpha_csv_logging_save_dir(save_dir, epoch, n1, n2, mode1, mode2):
    """
    Set up directory and path for alpha test CSV logging.

    Args:
        save_dir (str): 保存先のディレクトリパス.
        epoch (int): 現在のエポック.
        n1, n2, mode1, mode2: 今回は未使用（将来的な拡張用）。

    Returns:
        tuple: (alpha_csv_path, raw_probs_csv_path)
    """
    import os
    import csv

    os.makedirs(save_dir, exist_ok=True)
    alpha_csv_path = os.path.join(save_dir, f'alpha_log_epoch_{epoch}.csv')
    raw_probs_csv_path = os.path.join(save_dir, f'raw_probabilities_epoch_{epoch}.csv')

    # ヘッダー書き込み処理を共通化
    def write_header_if_needed(path, header):
        if not os.path.isfile(path):
            with open(path, 'w', newline='') as f:
                csv.writer(f).writerow(header)

    # alpha_logs用のヘッダー
    alpha_header = (
        ["alpha", "predicted_digit", "predicted_color", "predicted_combined",
         "digit_label_match", "color_label_match"] +
        [f"digit_probability_{i}" for i in range(10)] +
        [f"color_probability_{i}" for i in range(10)]
    )
    write_header_if_needed(alpha_csv_path, alpha_header)

    # raw_probabilities用のヘッダー（num_digits * num_colors = 100 と仮定）
    raw_header = ["alpha"] + [f"probability_{i}" for i in range(100)]
    write_header_if_needed(raw_probs_csv_path, raw_header)

    return alpha_csv_path, raw_probs_csv_path

def setup_alpha_csv_logging_save(experiment_name, epoch,n1,n2,mode1,mode2):
    """
    Set up directory and path for alpha test CSV logging.

    Args:
        experiment_name (str): Name of the experiment.
        epoch (int): Current epoch.

    Returns:
        str: Path to the alpha CSV file.
    """
    alpha_csv_dir = f"./alpha_test/{experiment_name}/{mode1}_{mode2}/{n1}/{n2}"
    os.makedirs(alpha_csv_dir, exist_ok=True)
    alpha_csv_path = os.path.join(alpha_csv_dir, f'alpha_log_epoch_{epoch}.csv')
    if not os.path.isfile(alpha_csv_path):
        with open(alpha_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header for alpha_logs
            writer.writerow([
                "alpha",
                #"digit_loss",
                #"color_loss",
                #"combined_loss",
                "predicted_digit",
                "predicted_color",
                "predicted_combined",
                "digit_label_match",
                "color_label_match",
                "digit_probability_0", "digit_probability_1", "digit_probability_2",
                "digit_probability_3", "digit_probability_4", "digit_probability_5",
                "digit_probability_6", "digit_probability_7", "digit_probability_8",
                "digit_probability_9",  # Adjust based on num_digits if necessary
                "color_probability_0", "color_probability_1", "color_probability_2",
                "color_probability_3", "color_probability_4", "color_probability_5",
                "color_probability_6", "color_probability_7", "color_probability_8",
                "color_probability_9"  # Adjust based on num_colors if necessary
            ])
    return alpha_csv_path

def setup_alpha_csv_logging_save2(experiment_name, epoch,n1,n2,mode1,mode2):
    """
    Set up directory and path for alpha test CSV logging.

    Args:
        experiment_name (str): Name of the experiment.
        epoch (int): Current epoch.

    Returns:
        str: Path to the alpha CSV file.
    """
    alpha_csv_dir = f"./alpha_test/{experiment_name}/{mode1}_{mode2}/{n2}/{n1}"
    os.makedirs(alpha_csv_dir, exist_ok=True)
    alpha_csv_path = os.path.join(alpha_csv_dir, f'alpha_log_epoch_{epoch}.csv')
    if not os.path.isfile(alpha_csv_path):
        with open(alpha_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header for alpha_logs
            writer.writerow([
                "alpha",
                #"digit_loss",
                #"color_loss",
                #"combined_loss",
                "predicted_digit",
                "predicted_color",
                "predicted_combined",
                "digit_label_match",
                "color_label_match",
                "digit_probability_0", "digit_probability_1", "digit_probability_2",
                "digit_probability_3", "digit_probability_4", "digit_probability_5",
                "digit_probability_6", "digit_probability_7", "digit_probability_8",
                "digit_probability_9",  # Adjust based on num_digits if necessary
                "color_probability_0", "color_probability_1", "color_probability_2",
                "color_probability_3", "color_probability_4", "color_probability_5",
                "color_probability_6", "color_probability_7", "color_probability_8",
                "color_probability_9"  # Adjust based on num_colors if necessary
            ])
    return alpha_csv_path

def setup_alpha_csv_logging(experiment_name, epoch):
    """
    Set up directory and path for alpha test CSV logging.

    Args:
        experiment_name (str): Name of the experiment.
        epoch (int): Current epoch.

    Returns:
        str: Path to the alpha CSV file.
    """
    alpha_csv_dir = f"./csv/combine/alpha_test/{experiment_name}"
    os.makedirs(alpha_csv_dir, exist_ok=True)
    alpha_csv_path = os.path.join(alpha_csv_dir, f'alpha_log_epoch_{epoch}.csv')
    if not os.path.isfile(alpha_csv_path):
        with open(alpha_csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header for alpha_logs
            writer.writerow([
                "alpha",
                "digit_loss",
                "color_loss",
                "combined_loss",
                "predicted_digit",
                "predicted_color",
                "predicted_combined",
                "digit_label_match",
                "color_label_match",
                "digit_probability_0", "digit_probability_1", "digit_probability_2",
                "digit_probability_3", "digit_probability_4", "digit_probability_5",
                "digit_probability_6", "digit_probability_7", "digit_probability_8",
                "digit_probability_9",  # Adjust based on num_digits if necessary
                "color_probability_0", "color_probability_1", "color_probability_2",
                "color_probability_3", "color_probability_4", "color_probability_5",
                "color_probability_6", "color_probability_7", "color_probability_8",
                "color_probability_9"  # Adjust based on num_colors if necessary
            ])
    return alpha_csv_path
def log_alpha_test_results_save(alpha_logs, alpha_csv_path, raw_probs_csv_path):
    """
    Log alpha test results to CSV files.

    Args:
        alpha_logs (dict): Dictionary containing alpha test logs.
        alpha_csv_path (str): Path to the main CSV file.
        raw_probs_csv_path (str): Path to the raw probabilities CSV file.
    """
    import csv

    # メインのアルファログの全行をリストにまとめる
    main_rows = []
    for i, alpha in enumerate(alpha_logs['alpha_values']):
        row = [
            alpha,
            alpha_logs['predicted_digits'][i],
            alpha_logs['predicted_colors'][i],
            alpha_logs['predicted_combined'][i],
            alpha_logs['digit_label_matches'][i],
            alpha_logs['color_label_matches'][i]
        ]
        # 各クラスの確率を連結
        row.extend(alpha_logs['digit_probabilities'][i])
        row.extend(alpha_logs['color_probabilities'][i])
        main_rows.append(row)

    # 全行をまとめて書き込み
    with open(alpha_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(main_rows)

    # 生の出力確率値の全行をリストにまとめる
    raw_rows = []
    for i, alpha in enumerate(alpha_logs['alpha_values']):
        # alphaに続けてraw_probabilitiesを連結（既にリスト化済み）
        row = [alpha] + alpha_logs['raw_probabilities'][i]
        raw_rows.append(row)

    # こちらも全行をまとめて書き込み
    with open(raw_probs_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(raw_rows)

def log_alpha_test_results(alpha_logs, alpha_csv_path):
    """
    Log alpha test results to CSV.

    Args:
        alpha_logs (dict): Dictionary containing alpha test logs.
        alpha_csv_path (str): Path to the CSV file.
    """
    with open(alpha_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for i, alpha in enumerate(alpha_logs['alpha_values']):
            writer.writerow([
                alpha,
                alpha_logs['digit_losses'][i],
                alpha_logs['color_losses'][i],
                alpha_logs['combined_losses'][i],
                alpha_logs['predicted_digits'][i],
                alpha_logs['predicted_colors'][i],
                alpha_logs['predicted_combined'][i],
                alpha_logs['digit_label_matches'][i],
                alpha_logs['color_label_matches'][i],
                *alpha_logs['digit_probabilities'][i],  # Log each digit probability
                *alpha_logs['color_probabilities'][i]   # Log each color probability
            ])

def log_to_wandb(wandb_run, log_data):
    """
    Log data to WandB.

    Args:
        wandb_run (wandb.Run): The WandB run.
        log_data (dict): Data to log.
    """
    if wandb_run:
        wandb.log(log_data)

def write_alpha_logs_with_raw_probs(alpha_csv_path, raw_probs_csv_path, alpha_logs):
    """
    Write alpha interpolation test results to CSV files.

    Args:
        alpha_csv_path (str): Path to the main alpha log CSV file.
        raw_probs_csv_path (str): Path to the raw probabilities CSV file.
        alpha_logs (dict): Dictionary containing the test results.
    """
    # メインのアルファログを書き込み
    with open(alpha_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for i, alpha in enumerate(alpha_logs['alpha_values']):
            row = [
                alpha,
                alpha_logs['predicted_digits'][i],
                alpha_logs['predicted_colors'][i],
                alpha_logs['predicted_combined'][i],
                alpha_logs['digit_label_matches'][i],
                alpha_logs['color_label_matches'][i]
            ]
            # Add digit probabilities
            row.extend(alpha_logs['digit_probabilities'][i])
            # Add color probabilities
            row.extend(alpha_logs['color_probabilities'][i])
            writer.writerow(row)
    
    # 生の出力確率値を書き込み
    with open(raw_probs_csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        for i, alpha in enumerate(alpha_logs['alpha_values']):
            row = [alpha] + alpha_logs['raw_probabilities'][i].tolist()
            writer.writerow(row)
