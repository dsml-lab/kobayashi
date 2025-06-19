import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def compile_images(base_dir, models, label_noise_rates, variances, metrics):
    for model in models:
        # モデルごとのディレクトリを作成
        model_dir = os.path.join(base_dir, f"{model}_compiled")
        os.makedirs(model_dir, exist_ok=True)

        # モデルごとにPDFファイルを作成
        pdf_path = os.path.join(model_dir, f"{model}_results.pdf")
        with PdfPages(pdf_path) as pdf:
            for metric in metrics:
                # 結果を格納する2D配列を作成
                results = [[None for _ in range(len(variances) + 1)] for _ in range(len(label_noise_rates) + 1)]
                
                # ヘッダーを設定
                results[0][0] = "LNR / Var"
                for i, var in enumerate(variances):
                    results[0][i+1] = f"Var {var}"
                for i, lnr in enumerate(label_noise_rates):
                    results[i+1][0] = f"LNR {lnr}"
                
                # 画像を読み込んで結果配列に格納
                for i, lnr in enumerate(label_noise_rates):
                    for j, var in enumerate(variances):
                        if var == "N/A":
                            pattern = f"{base_dir}/{model}_colored_emnist_combined_*_LabelNoiseRate{lnr}*/{metric}.png"
                        else:
                            pattern = f"{base_dir}/{model}_distribution_colored_emnist_Variance{var}_*_LabelNoiseRate{lnr}*/{metric}.png"
                        matching_files = glob.glob(pattern)
                        if matching_files:
                            results[i+1][j+1] = Image.open(matching_files[0])
                
                # 結果を表示してPDFに保存
                fig, axs = plt.subplots(len(label_noise_rates) + 1, len(variances) + 1, 
                                        figsize=(12 * (len(variances) + 1), 12 * (len(label_noise_rates) + 1)))
                fig.suptitle(f"{model} - {metric}", fontsize=32)
                
                for i in range(len(label_noise_rates) + 1):
                    for j in range(len(variances) + 1):
                        ax = axs[i, j]
                        if i == 0 or j == 0:
                            ax.text(0.5, 0.5, results[i][j], ha='center', va='center', fontsize=20)
                        elif results[i][j] is not None:
                            ax.imshow(results[i][j])
                        ax.axis('off')
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()

            print(f"Saved: {pdf_path}")

# スクリプトのメイン部分
base_dir = "/workspace/csv/combine"
models = ["cnn_2layers", "cnn_5layers", "resnet18"]
label_noise_rates = ["0.0", "0.1", "0.2", "0.4", "0.8"]
variances = ["N/A", "100", "1000", "10000", "100000"]  # N/A for non-distribution models
metrics = ["train_accuracies", "train_loss", "test_accuracies", "test_loss"]

compile_images(base_dir, models, label_noise_rates, variances, metrics)