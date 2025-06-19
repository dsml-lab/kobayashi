import os
import pandas as pd
import matplotlib.pyplot as plt

# ディレクトリのリスト
directories = [
    'csv/combine/cnn_2layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.0',
    'csv/combine/cnn_2layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.1',
    'csv/combine/cnn_2layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.2',
    'csv/combine/cnn_2layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.4',
    'csv/combine/cnn_2layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.8',
    'csv/combine/cnn_5layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.0',
    'csv/combine/cnn_5layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.1',
    'csv/combine/cnn_5layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.2',
    'csv/combine/cnn_5layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.4',
    'csv/combine/cnn_5layers_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.8',
    'csv/combine/resnet18_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.0',
    'csv/combine/resnet18_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.1',
    'csv/combine/resnet18_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.2',
    'csv/combine/resnet18_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.4',
    'csv/combine/resnet18_colored_emnist_combined_lr0.0001_batch256_epoch1000_LabelNoiseRate0.8'
]

# それぞれのディレクトリに対してプロットを作成
for directory in directories:
    file_path = os.path.join(directory, 'log.csv')
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        
        plt.figure(figsize=(12, 6))

        # 左のプロット (トレーニングデータ)
        plt.subplot(1, 2, 1)
        plt.plot(data['epoch'], data['train_combined_acc'], label='train_combined_acc')
        plt.plot(data['epoch'], data['train_digit_acc'], label='train_digit_acc')
        plt.plot(data['epoch'], data['train_color_acc'], label='train_color_acc')
        plt.xscale('log')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch (log scale)')
        plt.ylabel('Accuracy')
        plt.legend()

        # 右のプロット (テストデータ)
        plt.subplot(1, 2, 2)
        plt.plot(data['epoch'], data['test_combined_acc'], label='test_combined_acc')
        plt.plot(data['epoch'], data['test_digit_acc'], label='test_digit_acc')
        plt.plot(data['epoch'], data['test_color_acc'], label='test_color_acc')
        plt.xscale('log')
        plt.title('Test Accuracy')
        plt.xlabel('Epoch (log scale)')
        plt.ylabel('Accuracy')
        plt.legend()

        # タイトルを設定
        parts = directory.split('_')
        model_info = f'train_{parts[0]}_{parts[-1]}'
        plt.suptitle(model_info)

        # プロットを保存
        plt.savefig(os.path.join(directory, 'combined_accuracy_plot.png'))

        # プロットを表示
        plt.show()
