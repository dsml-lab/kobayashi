import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import logging
import fnmatch

plt.rcParams["figure.dpi"] = 300

def create_combined_figure(root_path, fig_dir_name='fig_and_log', 
                          smoothing=False, directories=None, output_suffix=''):
    """
    各サブディレクトリの fig_and_log 内の指定された画像を抜き出し、n行×5列のレイアウトで一つの大きな図を作成します。
    指定されたディレクトリのみを処理します。

    :param root_path: ルートディレクトリの絶対パス（例: /path/to/no_noise_no_noise/）
    :param fig_dir_name: 図とログが格納されているサブディレクトリ名
    :param smoothing: Trueの場合、スムージングされた画像を使用。Falseの場合、元の画像を使用。
    :param directories: 処理するサブディレクトリの絶対パスのリスト。Noneの場合、すべてのサブディレクトリを処理。
    :param output_suffix: 出力ファイル名に追加するサフィックス（例: 'group1'）
    """
    # ログの設定
    logging.basicConfig(filename=os.path.join(root_path, 'combined_figure.log'), level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    # ルートパスから variance と label_noise_rate を抽出
    variance_match = re.search(r'variance(\d+)', root_path, re.IGNORECASE)
    label_noise_match = re.search(r'LabelNoiseRate([0-9.]+)', root_path, re.IGNORECASE)
    
    if variance_match:
        variance = variance_match.group(1)
    else:
        variance = 'unknown'
        logging.warning("variance が root_path から抽出できませんでした。'unknown' を使用します。")

    if label_noise_match:
        label_noise_rate = label_noise_match.group(1)
    else:
        label_noise_rate = 'unknown'
        logging.warning("label_noise_rate が root_path から抽出できませんでした。'unknown' を使用します。")

    # *** 保存先ディレクトリを変更 ***
    # combine_fig/{root_directory}/ を保存先として設定
    # root_directory は root_path のベースネームとします
    combined_fig_dir = os.path.join('combine_fig', os.path.basename(root_path))
    os.makedirs(combined_fig_dir, exist_ok=True)
    # ***

    # サブディレクトリのリストを取得
    if directories is not None:
        # directories は絶対パスのリスト
        subdirs = directories
        # 存在しないディレクトリをチェック
        missing_dirs = [d for d in directories if not os.path.isdir(os.path.join(d, fig_dir_name))]
        if missing_dirs:
            warning_msg = f"警告: 以下の指定ディレクトリが存在しません: {', '.join(missing_dirs)}"
            print(warning_msg)
            logging.warning(warning_msg)
            # 存在するディレクトリのみを残す
            subdirs = [d for d in directories if os.path.isdir(os.path.join(d, fig_dir_name))]
    else:
        # ディレクトリ名が数字であるサブディレクトリのみを取得
        subdirs = sorted(
            [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d.isdigit()],
            key=lambda x: int(x)
        )

    print(f"Processing {len(subdirs)} directories for {output_suffix}")

    # 各サブディレクトリ内の画像パスを収集
    image_paths = []
    for subdir in subdirs:
        fig_and_log_path = os.path.join(subdir, fig_dir_name)
        print(f"Checking directory: {fig_and_log_path}")
        if os.path.isdir(fig_and_log_path):
            # 1. selected_data_points_*.png (部分一致)
            selected_data_points = None
            for file in os.listdir(fig_and_log_path):
                if fnmatch.fnmatch(file, 'selected_data_points_*.png'):
                    selected_data_points = os.path.join(fig_and_log_path, file)
                    break

            # 2. eval_alpha_color.png または eval_alpha_color_smoothing.png
            eval_alpha_color = os.path.join(fig_and_log_path, 's_eval_alpha_color_smoothing.png' if smoothing else 's_eval_alpha_color.png')
            
            # 3. eval_epoch_color.png
            eval_epoch_color = os.path.join(fig_and_log_path, 's_eval_epoch_color.png')
            
            # 4. eval_alpha_digit.png または eval_alpha_digit_smoothing.png
            eval_alpha_digit = os.path.join(fig_and_log_path, 'eval_alpha_digit_smoothing.png' if smoothing else 's_eval_alpha_digit.png')
            
            # 5. eval_epoch_digit.png
            eval_epoch_digit = os.path.join(fig_and_log_path, 's_eval_epoch_digit.png')
            
            # 画像が存在するか確認
            required_files = {
                'selected_data_points': selected_data_points,
                'eval_alpha_color': eval_alpha_color,
                'eval_epoch_color': eval_epoch_color,
                'eval_alpha_digit': eval_alpha_digit,
                'eval_epoch_digit': eval_epoch_digit
            }
            missing_files = [key for key, path in required_files.items() if not path or not os.path.isfile(path)]
            
            if not missing_files:
                image_paths.append(required_files)
            else:
                warning_msg = f"警告: {fig_and_log_path} 内に以下の画像ファイルが不足しています: {', '.join(missing_files)}"
                print(warning_msg)
                logging.warning(warning_msg)
        else:
            warning_msg = f"警告: {fig_dir_name} ディレクトリが {fig_and_log_path} 内に存在しません。"
            print(warning_msg)
            logging.warning(warning_msg)
    
    num_subdirs = len(image_paths)
    if num_subdirs == 0:
        error_msg = "エラー: 有効な画像が見つかりませんでした。"
        print(error_msg)
        logging.error(error_msg)
        return

    # 各サブディレクトリごとに5つの画像を配置
    images_per_subdir = 5
    total_rows = num_subdirs
    total_cols = images_per_subdir

    # 図のサイズを調整（各画像に対して4インチ幅、3インチ高さ）
    plt.figure(figsize=(4 * total_cols, 3 * total_rows))  
    
    # 画像の配置順序を定義（左から右へ: selected_data_points, eval_alpha_color, eval_epoch_color, eval_alpha_digit, eval_epoch_digit）
    image_order = ['selected_data_points', 'eval_alpha_color', 'eval_epoch_color', 'eval_alpha_digit', 'eval_epoch_digit']
    
    for row_idx, data in enumerate(image_paths):
        # 対応するディレクトリパスを取得
        if directories is not None:
            subdir = subdirs[row_idx]  # 絶対パス
            display_dir = os.path.relpath(subdir, root_path)  # 表示用に相対パス
        else:
            subdir = os.path.join(root_path, subdirs[row_idx])  # 絶対パス
            display_dir = subdirs[row_idx]  # 表示用に名前だけ

        for col_idx, key in enumerate(image_order):
            img_path = data[key]
            try:
                img = Image.open(img_path)
                ax = plt.subplot(total_rows, total_cols, row_idx * total_cols + col_idx + 1)
                ax.imshow(img)
                ax.axis('off')
                if col_idx == 0:
                    ax.set_ylabel(f"Dir {display_dir}", fontsize=12, rotation=0, labelpad=40, va='center')
            except Exception as e:
                error_msg = f"エラー: {img_path} の読み込みに失敗しました。 {e}"
                print(error_msg)
                logging.error(error_msg)
    
    plt.tight_layout()
    
    # 出力ファイル名の生成
    output_filename = f"variance_{variance}_label_noise_rate{label_noise_rate}_{output_suffix}.png"
    output_path = os.path.join(combined_fig_dir, output_filename)
    
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    success_msg = f"大きな図を {output_path} として保存しました。"
    print(success_msg)
    logging.info(success_msg)

def traverse_and_group(root_path):
    """
    親ディレクトリと子ディレクトリの1の位および10の位が一致するグループに分類します。

    :param root_path: ルートディレクトリの絶対パス
    :return: グループ1とグループ2のリスト
    """
    group1_total = []  # 親ディレクトリと子ディレクトリの1の位が一致するグループ
    group2_total = []  # 親ディレクトリと子ディレクトリの10の位が一致するグループ

    # 親ディレクトリ内のすべてのサブディレクトリを取得（数字のみ）
    all_subdirs = sorted(
        [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d)) and d.isdigit()],
        key=lambda x: int(x)
    )

    for parent_dir in all_subdirs:
        parent_path = os.path.join(root_path, parent_dir)
        parent_num = int(parent_dir)
        parent_ones = parent_num % 10
        parent_tens = (parent_num // 10) % 10  # 10の位を取得

        # 子ディレクトリを取得（数字のみ）
        child_subdirs = sorted(
            [d for d in os.listdir(parent_path) if os.path.isdir(os.path.join(parent_path, d)) and d.isdigit()],
            key=lambda x: int(x)
        )

        for child_dir in child_subdirs:
            child_num = int(child_dir)
            child_ones = child_num % 10
            child_tens = (child_num // 10) % 10

            # グループ1: 親の1の位と子の1の位が一致
            if parent_ones == child_ones:
                group1_total.append(os.path.join(parent_path, child_dir))

            # グループ2: 親の10の位と子の10の位が一致
            if parent_tens == child_tens:
                group2_total.append(os.path.join(parent_path, child_dir))

    return group1_total, group2_total

def find_group3(root_path, group1, group2):
    """
    グループ1およびグループ2に該当しないディレクトリをグループ3として分類します。

    :param root_path: ルートディレクトリの絶対パス
    :param group1: グループ1の絶対パスリスト
    :param group2: グループ2の絶対パスリスト
    :return: グループ3のリスト
    """
    group1_set = set(group1)
    group2_set = set(group2)
    group3 = []

    # 全ての親ディレクトリと子ディレクトリのパスを取得
    all_subdirs = []
    for parent_dir in os.listdir(root_path):
        parent_path = os.path.join(root_path, parent_dir)
        if os.path.isdir(parent_path) and parent_dir.isdigit():
            for child_dir in os.listdir(parent_path):
                child_path = os.path.join(parent_path, child_dir)
                if os.path.isdir(child_path) and child_dir.isdigit():
                    all_subdirs.append(child_path)

    for subdir in all_subdirs:
        if subdir not in group1_set and subdir not in group2_set:
            group3.append(subdir)

    return group3

# 使用例
if __name__ == "__main__":
    # ルートディレクトリを絶対パスに設定
    root_directory = os.path.abspath("/workspace/alpha_test/seed/lr_0.2/ori_closet2_seed43_cnn_5layers_distribution_colored_emnist_variance1000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/noise_no_noise")

    # ディレクトリのグルーピング
    group1, group2 = traverse_and_group(root_directory)

    # グループ3の取得
    group3 = find_group3(root_directory, group1, group2)

    # グループ1の処理
    if group1:
        create_combined_figure(
            root_path=root_directory, 
            fig_dir_name='fig_and_log', 
            smoothing=False,
            directories=group1,
            output_suffix='group1'
        )
    else:
        print("グループ1に該当するディレクトリがありません。")

    # グループ2の処理
    if group2:
        create_combined_figure(
            root_path=root_directory, 
            fig_dir_name='fig_and_log', 
            smoothing=False,
            directories=group2,
            output_suffix='group2'
        )
    else:
        print("グループ2に該当するディレクトリがありません。")

    # グループ3の処理
    if group3:
        create_combined_figure(
            root_path=root_directory, 
            fig_dir_name='fig_and_log', 
            smoothing=False,
            directories=group3,
            output_suffix='group3'
        )
    else:
        print("グループ3に該当するディレクトリがありません。")
