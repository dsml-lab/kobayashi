import os
import pandas as pd

def process_csv_file(file_path, epoch, tolerance=0.01):
    """
    指定されたCSVファイルを処理し、新しい列を追加し、合計値を検証します。

    Parameters:
    - file_path (str): 処理するCSVファイルのパス。
    - epoch (str): エポック番号（ファイル名から取得）。
    - tolerance (float): 合計値の許容誤差。
    """
    try:
        df = pd.read_csv(file_path)

        # color_probability_0 ~ color_probability_9 の合計を計算
        color_prob_cols = [f"color_probability_{i}" for i in range(10)]
        if all(col in df.columns for col in color_prob_cols):
            df['sum_color_prob'] = df[color_prob_cols].sum(axis=1)
        else:
            missing_color_cols = [col for col in color_prob_cols if col not in df.columns]
            print(f"[Warning] Epoch {epoch}: 一部の color_probability 列が見つかりません: {', '.join(missing_color_cols)}")
            df['sum_color_prob'] = None  # 欠損値として設定

        # digit_probability_0 ~ digit_probability_9 の合計を計算
        digit_prob_cols = [f"digit_probability_{i}" for i in range(10)]
        if all(col in df.columns for col in digit_prob_cols):
            df['sum_digit_prob'] = df[digit_prob_cols].sum(axis=1)
        else:
            missing_digit_cols = [col for col in digit_prob_cols if col not in df.columns]
            print(f"[Warning] Epoch {epoch}: 一部の digit_probability 列が見つかりません: {', '.join(missing_digit_cols)}")
            df['sum_digit_prob'] = None  # 欠損値として設定

        # 合計値の検証
        invalid_rows = pd.DataFrame()

        if 'sum_color_prob' in df.columns and df['sum_color_prob'].notnull().any():
            invalid_color = (df['sum_color_prob'] - 1).abs() > tolerance
            invalid_rows = invalid_rows.append(df[invalid_color])

        if 'sum_digit_prob' in df.columns and df['sum_digit_prob'].notnull().any():
            invalid_digit = (df['sum_digit_prob'] - 1).abs() > tolerance
            invalid_rows = invalid_rows.append(df[invalid_digit])

        # 重複行を削除
        invalid_rows = invalid_rows.drop_duplicates()

        if not invalid_rows.empty:
            for _, row in invalid_rows.iterrows():
                alpha = row.get('alpha', 'N/A')
                sum_color = row.get('sum_color_prob', 'N/A')
                sum_digit = row.get('sum_digit_prob', 'N/A')
                print(f"Epoch: {epoch}, Alpha: {alpha}, sum_color_prob: {sum_color:.5f}, sum_digit_prob: {sum_digit:.5f}")

        # CSVファイルに上書き保存
        df.to_csv(file_path, index=False)

    except Exception as e:
        print(f"[Error] Epoch {epoch}: CSVファイルの処理中にエラーが発生しました: {e}")

def main():
    #list = ["10","20","30","40","50","60","70","80","90"]
    #for i in list:
    base_path = f"alpha_test/v1_cnn_5layers_distribution_colored_emnist_variance10000_combined_lr0.01_batch256_epoch2000_LabelNoiseRate0.5_Optimsgd/no_noise_noise/90/71"
    
    if not os.path.exists(base_path):
        print(f"[Error] 指定されたベースパスが存在しません: {base_path}")
        return

    #csv_dir = os.path.join(base_path, "csv")
    csv_dir = base_path
    if not os.path.exists(csv_dir):
        print(f"[Error] CSVディレクトリが存在しません: {csv_dir}")
        return

    # CSVファイルの一覧を取得
    csv_files = [f for f in os.listdir(csv_dir) if f.startswith("alpha_log_epoch_") and f.endswith(".csv")]
    if not csv_files:
        print("[Info] 該当するCSVファイルが見つかりません。")
        return

    # エポック番号のリストを作成
    epochs = [f.replace("alpha_log_epoch_", "").replace(".csv", "") for f in csv_files]
    epochs = sorted(epochs, key=lambda x: int(x) if x.isdigit() else float(x) if x.replace('.', '', 1).isdigit() else x)

    # 各CSVファイルを処理
    for csv_file, epoch in zip(csv_files, epochs):
        selected_path = os.path.join(csv_dir, csv_file)
        process_csv_file(selected_path, epoch)

    # 最終メッセージは出力しません

if __name__ == "__main__":
    main()
