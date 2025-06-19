import os
import shutil

def copy_fig_and_log(source_root, destination_root):
    """
    指定されたソースディレクトリ内のすべてのfig_and_logディレクトリを
    デスティネーションディレクトリに階層を維持してコピーします。

    :param source_root: コピー元のno_noise_no_noiseディレクトリのパス
    :param destination_root: コピー先のディレクトリのパス
    """
    for dirpath, dirnames, filenames in os.walk(source_root):
        # ディレクトリ名が 'fig_and_log' の場合
        if os.path.basename(dirpath) == 'fig_and_log':
            # 相対パスを取得
            relative_path = os.path.relpath(dirpath, source_root)
            # コピー先のパスを作成
            destination_path = os.path.join(destination_root, relative_path)
            
            # コピー先ディレクトリが存在しない場合は作成
            os.makedirs(destination_path, exist_ok=True)
            
            # ファイルをコピー
            for file in filenames:
                source_file = os.path.join(dirpath, file)
                destination_file = os.path.join(destination_path, file)
                shutil.copy2(source_file, destination_file)
                print(f"Copied: {source_file} -> {destination_file}")

if __name__ == "__main__":
    # コピー元のno_noise_no_noiseディレクトリのパス
    var="3162"
    source_root = f"/workspace/alpha_test/cnn_5layers_distribution_colored_emnist_variance{var}_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise"
    
    # コピー先のディレクトリのパス（適宜変更してください）
    destination_root = f"./workspace/path/to/destination{var}/noise_no_noise"

    # コピー先ディレクトリが存在しない場合は作成
    os.makedirs(destination_root, exist_ok=True)

    # コピー処理を実行
    copy_fig_and_log(source_root, destination_root)

    print("全てのfig_and_logディレクトリのファイルをコピーしました。")