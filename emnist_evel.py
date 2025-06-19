import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt

def plot_predictions_over_epochs(csv_dir, noisy_label_y, noisy_label_x):
    """
    指定ディレクトリ内の alpha_log_epoch_*.csv をすべて読み込み、
    エポックごとに「predicted_digit == noisy_label_y (または x)」だった割合を計算してプロットする。
    
    Args:
        csv_dir (str): CSVファイルが格納されているディレクトリパス
        noisy_label_y (int): 予測として見たいラベル y
        noisy_label_x (int): 予測として見たいラベル x
    """
    epoch_list = []
    frac_y_list = []  # predicted_digit == noisy_label_y の割合
    frac_x_list = []  # predicted_digit == noisy_label_x の割合

    # alpha_log_epoch_XXX.csv を順に取得
    csv_files = sorted(glob.glob(os.path.join(csv_dir, 'alpha_log_epoch_*.csv')), 
                       key=lambda path: int(re.findall(r'epoch_(\d+)', os.path.basename(path))[0]))

    for csv_file in csv_files:
        # ファイル名からエポック番号を抽出 (例: alpha_log_epoch_12.csv -> epoch=12)
        m = re.search(r'alpha_log_epoch_(\d+)\.csv', os.path.basename(csv_file))
        if not m:
            continue
        epoch = int(m.group(1))

        # CSV 読み込み
        df = pd.read_csv(csv_file)

        # predicted_digit == noisy_label_y の割合を計算
        frac_y = (df['predicted_digit'] == noisy_label_y).mean()

        # predicted_digit == noisy_label_x の割合を計算
        frac_x = (df['predicted_digit'] == noisy_label_x).mean()

        epoch_list.append(epoch)
        frac_y_list.append(frac_y)
        frac_x_list.append(frac_x)

    # -----------------
    # Plot
    # -----------------
    plt.figure(figsize=(8,5))
    plt.plot(epoch_list, frac_y_list, label=f'Pred = {noisy_label_y}')
    plt.plot(epoch_list, frac_x_list, label=f'Pred = {noisy_label_x}')
    
    plt.xlabel('Epoch')
    plt.ylabel('Fraction of alpha that predicted each label')
    plt.title(f"Prediction fractions over epochs\n(y={noisy_label_y}, x={noisy_label_x})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("test2.png")
    print("sabve_fig")

if __name__ == "__main__":
    # 例: CSV格納ディレクトリと、観察したいラベルを指定
    csv_directory = "alpha_test/EMNIST/cnn_5layers_emnist_digits_variance0_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise/1/3/csv"
    noisy_label_y = 3   # 例: {noisy_label_y} = 1
    noisy_label_x = 1   # 例: {noisy_label_x} = 0

    plot_predictions_over_epochs(csv_directory, noisy_label_y, noisy_label_x)
