import numpy as np
import matplotlib.pyplot as plt
import os

def load_colored_emnist_data(seed, variance, correlation):
    # 指定されたベースパスでデータをロード
    base_path = f'data/distribution_colored_EMNIST_Seed{seed}_Var{variance}_Corr{correlation}'
    x_train_colored = np.load(f'{base_path}/x_train_colored.npy')
    y_train_digits = np.load(f'{base_path}/y_train_digits.npy')
    y_train_colors = np.load(f'{base_path}/y_train_colors.npy')
    y_train_combined = np.load(f'{base_path}/y_train_combined.npy')
    return x_train_colored, y_train_digits, y_train_colors, y_train_combined

def display_and_save_colored_image(x_train_colored, y_train_digits, y_train_colors, y_train_combined, index, save_path="colored_image.png"):
    # 指定インデックスの画像とラベルを取得
    image = x_train_colored[index]
    digit_label = y_train_digits[index]
    color_label = y_train_colors[index]
    combined_label = y_train_combined[index]
    
    # 画像表示設定
    plt.imshow(image)
    plt.axis('off')
    
    # ラベルを画像の下に配置
    plt.figtext(0.5, 0.01, f'Digit Label: {digit_label}, Color Label: {color_label}, Combined Label: {combined_label}', 
                wrap=True, horizontalalignment='center', fontsize=12)
    
    # 画像とラベルを保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 変数はデータセット作成時と同じに設定
    seed = 42
    variance = 1000
    correlation = 0.5
    
    # データのロード
    x_train_colored, y_train_digits, y_train_colors, y_train_combined = load_colored_emnist_data(seed, variance, correlation)
    
    # 表示したいインデックスと保存先を指定
    display_index = 2745  # 任意のインデックスに変更可能
    save_path = f"your_path.png"  # 保存ファイルパス
    display_and_save_colored_image(x_train_colored, y_train_digits, y_train_colors, y_train_combined, display_index, save_path)