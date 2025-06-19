import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def convert_variance(variance):
    if variance == "N/A":
        return variance
    return f"{float(variance) / (255.0 ** 2):.6f}"

def compile_images(base_dir, models, label_noise_rates, variances):
    for model in models:
        # モデルごとのディレクトリを作成
        model_dir = os.path.join(base_dir, f"{model}_compiled")
        os.makedirs(model_dir, exist_ok=True)

        # モデルごとにPDFファイルを作成
        pdf_path = os.path.join(model_dir, f"{model}_results.pdf")
        with PdfPages(pdf_path) as pdf:
            # グラフを格納する2D配列を作成
            fig, axs = plt.subplots(len(label_noise_rates) * 2, len(variances), figsize=(5 * len(variances), 10 * len(label_noise_rates)))
            fig.suptitle(f"{model}", fontsize=32)

            # 凡例用のラベルとハンドルを格納するリスト
            handles_acc, labels_acc = None, None
            handles_loss, labels_loss = None, None

            for i, lnr in enumerate(label_noise_rates):
                for j, var in enumerate(variances):
                    if var == "N/A":
                        pattern = f"{base_dir}/{model}_colored_emnist_combined_*_LabelNoiseRate{lnr}*/log.csv"
                    else:
                        pattern = f"{base_dir}/{model}_distribution_colored_emnist_Variance{var}_*_LabelNoiseRate{lnr}*/log.csv"
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        df = pd.read_csv(matching_files[0])
                        epochs = df['epoch']
                        train_loss = df[' train_loss']
                        test_loss = df[' test_loss']
                        train_combined_acc = df[' train_combined_acc']
                        test_combined_acc = df[' test_combined_acc']
                        train_digit_acc = df[' train_digit_acc']
                        test_digit_acc = df[' test_digit_acc']
                        train_color_acc = df[' train_color_acc']
                        test_color_acc = df[' test_color_acc']

                        loss_min, loss_max = min(train_loss.min(), test_loss.min()), max(train_loss.max(), test_loss.max())
                        
                        # 訓練とテストの正解率のグラフ
                        ax_acc = axs[i * 2, j]
                        ax_acc.plot(epochs, train_combined_acc, color='blue')
                        ax_acc.plot(epochs, test_combined_acc, color='blue', linestyle='dashed')
                        ax_acc.plot(epochs, train_digit_acc, color='green')
                        ax_acc.plot(epochs, test_digit_acc, color='green', linestyle='dashed')
                        ax_acc.plot(epochs, train_color_acc, color='red')
                        ax_acc.plot(epochs, test_color_acc, color='red', linestyle='dashed')
                        ax_acc.set_xscale('log')
                        ax_acc.set_xlim(1, 1000)
                        ax_acc.set_ylim(0, 1)
                        ax_acc.set_xlabel('Epoch')
                        ax_acc.set_ylabel('Accuracy')
                        ax_acc.set_title(f'Variance {convert_variance(var)}, Label Noise {lnr}', fontsize=10, pad=10)
                        ax_acc.grid(True)

                        # 損失のグラフ
                        ax_loss = axs[i * 2 + 1, j]
                        ax_loss.plot(epochs, train_loss, color='blue')
                        ax_loss.plot(epochs, test_loss, color='blue', linestyle='dashed')
                        ax_loss.set_xscale('log')
                        ax_loss.set_xlim(1, 1000)
                        ax_loss.set_ylim(loss_min, loss_max)
                        ax_loss.set_xlabel('Epoch')
                        ax_loss.set_ylabel('Loss')
                        ax_loss.set_title(f'Variance {convert_variance(var)}, Label Noise {lnr}', fontsize=10, pad=10)
                        ax_loss.grid(True)
                    else:
                        axs[i * 2, j].axis('off')
                        axs[i * 2 + 1, j].axis('off')

            # 凡例を右上にまとめて表示
            handles, labels = axs[0, 0].get_legend_handles_labels()
            fig.legend(handles, ['Train Combined Accuracy', 'Test Combined Accuracy', 'Train Digit Accuracy', 'Test Digit Accuracy', 'Train Color Accuracy', 'Test Color Accuracy', 'Train Loss', 'Test Loss'],
                       loc='upper right', fontsize='small', bbox_to_anchor=(1.1, 1))

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            pdf.savefig(fig, dpi=600)  # 高解像度で保存
            plt.close()

            print(f"Saved: {pdf_path}")

# スクリプトのメイン部分
base_dir = "/workspace/csv/combine"
models = ["cnn_2layers", "cnn_5layers", "resnet18"]
label_noise_rates = ["0.0", "0.1", "0.2", "0.4", "0.8"]
variances = ["N/A", "100", "1000", "10000", "100000"]  # N/A for non-distribution models

compile_images(base_dir, models, label_noise_rates, variances)