import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# フォントをArialに設定し、サイズを1.5倍に調整
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = matplotlib.rcParams['font.size'] * 1.5

# データの読み込みパスを指定してください
datapath = "csv/combine/split_noise/cnn_5layers_distribution_colored_emnist_variance10000_combined_lr0.01_batch256_epoch1300_LabelNoiseRate0.5_Optimsgd_distance8.4839_xlabel50_ylabelafter39/log.csv"
data = pd.read_csv(datapath)

# エポックを x 軸に設定
epochs = data['epoch']

# 色の設定
colors = {'digit': 'blue', 'color': 'red', 'both': 'green'}

# 画像を生成するエポックのリスト
highlight_epochs = [50, 150, 500, 1300]

# 各エポックを強調する画像を生成
for ep in highlight_epochs:
    fig, axes = plt.subplots(4, 1, figsize=(10, 20))
    fig.tight_layout(pad=5.0)

    for i, ylabel in enumerate(["Test Error (%)", "Train Error (%)", "Train Error Clean/Noisy (%)", "Loss"]):
        # データプロットと軸設定
        axes[i].plot(epochs, 100 - data['test_accuracy'] if i == 0 else
                     100 - data['train_accuracy'] if i == 1 else
                     100 - data['train_accuracy_clean'], label='test both' if i == 0 else
                     'train both' if i == 1 else 'train clean both', color=colors['both'])
        axes[i].set_xscale('log')
        axes[i].set_ylabel(ylabel)
        axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        
        # エポックラインとラベル
        for other_ep in highlight_epochs:
            line_color = 'purple' if other_ep == ep else 'gray'
            fontsize = matplotlib.rcParams['font.size'] * 2 if other_ep == ep else matplotlib.rcParams['font.size']
            axes[i].axvline(x=other_ep, color=line_color, linestyle='--', linewidth=1.5)
            axes[i].text(other_ep, axes[i].get_ylim()[1], f'{other_ep}', color=line_color, ha='center', va='bottom', fontweight='bold', fontsize=fontsize)

    plt.savefig(f"test_epoch_{ep}.png", bbox_inches='tight')
    plt.close(fig)

# すべてのエポックラベルを灰色にした画像を追加
fig, axes = plt.subplots(4, 1, figsize=(10, 20))
fig.tight_layout(pad=5.0)

for i, ylabel in enumerate(["Test Error (%)", "Train Error (%)", "Train Error Clean/Noisy (%)", "Loss"]):
    # データプロットと軸設定
    axes[i].plot(epochs, 100 - data['test_accuracy'] if i == 0 else
                 100 - data['train_accuracy'] if i == 1 else
                 100 - data['train_accuracy_clean'], label='test both' if i == 0 else
                 'train both' if i == 1 else 'train clean both', color=colors['both'])
    axes[i].set_xscale('log')
    axes[i].set_ylabel(ylabel)
    axes[i].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # すべてのエポックラベルを灰色に設定
    for other_ep in highlight_epochs:
        axes[i].axvline(x=other_ep, color='gray', linestyle='--', linewidth=1.5)
        axes[i].text(other_ep, axes[i].get_ylim()[1], f'{other_ep}', color='gray', ha='center', va='bottom', fontweight='bold', fontsize=matplotlib.rcParams['font.size'] * 2)

plt.savefig("test_all_gray_epochs.png", bbox_inches='tight')
plt.close(fig)
