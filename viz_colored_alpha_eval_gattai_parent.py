import os
import re
from PIL import Image
import matplotlib.pyplot as plt
import logging
import fnmatch

plt.rcParams["figure.dpi"] = 300

def create_combined_figure(root_path, fig_dir_name='fig_and_log', 
                          smoothing=False, first_dir=None, end_dir=None):
    """
    各サブディレクトリの fig_and_log 内の指定された画像を抜き出し、n行×5列のレイアウトで一つの大きな図を作成します。
    指定された範囲のサブディレクトリのみを処理します。

    :param root_path: ルートディレクトリのパス（例: .../no_noise_no_noise）
    :param fig_dir_name: 図とログが格納されているサブディレクトリ名
    :param smoothing: Trueの場合、スムージングされた画像を使用。Falseの場合、元の画像を使用。
    :param first_dir: 処理を開始するサブディレクトリ名（例: '1'）
    :param end_dir: 処理を終了するサブディレクトリ名（例: '5'）
    """
    # ログの設定
    combined_fig_dir = os.path.join(root_path, 'combined_fig')
    os.makedirs(combined_fig_dir, exist_ok=True)
    log_file = os.path.join(combined_fig_dir, 'combined_figure.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
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

    # サブディレクトリ名のリストを取得
    all_subdirs = sorted(
        [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))],
        key=lambda x: (0, int(x)) if x.isdigit() else (1, x)
    )

    # first_dir と end_dir が指定されている場合、その範囲でサブディレクトリをフィルタリング
    if first_dir and end_dir:
        try:
            first_index = all_subdirs.index(first_dir)
            end_index = all_subdirs.index(end_dir)
            if first_index > end_index:
                first_index, end_index = end_index, first_index  # 順序が逆の場合は入れ替え
            selected_subdirs = all_subdirs[first_index:end_index + 1]
            logging.info(f"サブディレクトリの範囲を '{first_dir}' から '{end_dir}' に設定しました。")
        except ValueError as e:
            error_msg = f"エラー: 指定された 'first_dir' または 'end_dir' が存在しません。 {e}"
            print(error_msg)
            logging.error(error_msg)
            return
    elif first_dir or end_dir:
        warning_msg = "警告: 'first_dir' と 'end_dir' は両方とも指定してください。範囲指定を無視します。"
        print(warning_msg)
        logging.warning(warning_msg)
        selected_subdirs = all_subdirs
    else:
        selected_subdirs = all_subdirs

    # 各サブディレクトリ内の画像パスを収集
    image_paths = []
    for subdir in selected_subdirs:
        # 内部のディレクトリ（例: '1/1', '2/2' など）を対象とする
        inner_dir = os.path.join(root_path, subdir, subdir)
        fig_and_log_path = os.path.join(inner_dir, fig_dir_name)
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
            eval_alpha_digit = os.path.join(fig_and_log_path, 's_eval_alpha_digit_smoothing.png' if smoothing else 's_eval_alpha_digit.png')
            
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
            warning_msg = f"警告: {fig_dir_name} ディレクトリが {inner_dir} 内に存在しません。"
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
        subdir = selected_subdirs[row_idx]
        for col_idx, key in enumerate(image_order):
            img_path = data[key]
            try:
                img = Image.open(img_path)
                ax = plt.subplot(total_rows, total_cols, row_idx * total_cols + col_idx + 1)
                ax.imshow(img)
                ax.axis('off')
                if col_idx == 0:
                    ax.set_ylabel(f"Dir {subdir}", fontsize=12, rotation=0, labelpad=40, va='center')
            except Exception as e:
                error_msg = f"エラー: {img_path} の読み込みに失敗しました。 {e}"
                print(error_msg)
                logging.error(error_msg)

    plt.tight_layout()

    # 出力ファイル名の生成
    if first_dir and end_dir:
        output_filename = f"variance_{variance}_label_noise_rate{label_noise_rate}_{first_dir}_to_{end_dir}.png"
    else:
        output_filename = f"variance_{variance}_label_noise_rate{label_noise_rate}_all.png"
    output_path = os.path.join(combined_fig_dir, output_filename)

    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    success_msg = f"大きな図を {output_path} として保存しました。"
    print(success_msg)
    logging.info(success_msg)

# 使用例
if __name__ == "__main__":
    root_directory = '/workspace/alpha_test/test_closet_cnn_5layers_distribution_colored_emnist_variance3162_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd_Momentum0.0/no_noise_no_noise'

    # 範囲ごとに create_combined_figure を呼び出す
    ranges = [
        ('1', '5'),
        ('6', '9'),
        ('10', '50'),
        ('60', '90')
    ]

    for first, end in ranges:
        create_combined_figure(
            root_path=root_directory, 
            fig_dir_name='fig_and_log', 
            smoothing=False,
            first_dir=first,      # 開始ディレクトリ名
            end_dir=end           # 終了ディレクトリ名
        )
