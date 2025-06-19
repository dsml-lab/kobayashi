import os
import pandas as pd

def evaluate_label_changes_vectorized(start_epoch, end_epoch, ratio=True, csv_dir="."):
    """
    指定したエポック範囲内のCSVファイルから、各サンプルの予測ラベルの変化数をベクトル演算で高速に計算する関数。
    
    各CSVファイルは "test_inferenc_epoch_{epoch}.csv" という名前で存在するものとする。
    読み込む際は、"TestData_Index" と "Predicted_Label" の2列のみを対象とする。

    Parameters
    ----------
    start_epoch : int
        評価開始エポック
    end_epoch : int
        評価終了エポック
    ratio : bool, default True
        Trueの場合、変化数を (エポック数 - 1) で割った比率として返す。
        Falseの場合は変化数そのままを返す。
    csv_dir : str, default "."
        CSVファイルが存在するディレクトリのパス

    Returns
    -------
    pd.DataFrame or None
        DataFrameは "TestData_Index" と "Label_Change" の列を持つ。
        十分なデータが取得できなかった場合は None を返す。
    """
    predictions_list = []
    test_data_index = None
    epochs_found = []

    for epoch in range(start_epoch, end_epoch + 1):
        csv_filename = os.path.join(csv_dir, f"test_inferenc_epoch_{epoch}.csv")
        if not os.path.exists(csv_filename):
            print(f"Warning: {csv_filename} が存在しません。epoch {epoch} はスキップします。")
            continue

        try:
            # 必要な2列のみ読み込む
            df = pd.read_csv(csv_filename, usecols=["TestData_Index", "Predicted_Label"])
        except Exception as e:
            print(f"Error: {csv_filename} の読み込みに失敗しました。{e}")
            continue

        if test_data_index is None:
            # 最初に読み込んだ TestData_Index を基準として保持
            test_data_index = df["TestData_Index"].sort_values().reset_index(drop=True)
        else:
            # すでに読み込んだ test_data_index と一致しているかチェック（ソート済みである前提）
            current_index = df["TestData_Index"].sort_values().reset_index(drop=True)
            if not test_data_index.equals(current_index):
                print(f"Warning: {csv_filename} の TestData_Index が基準と一致しません。")
                continue

        # TestData_Index でソートして Predicted_Label のみ取得
        predictions_list.append(df.sort_values("TestData_Index").reset_index(drop=True)["Predicted_Label"])
        epochs_found.append(epoch)

    if len(predictions_list) < 2:
        print("評価するための十分なエポックのCSVファイルが読み込めませんでした。")
        return None

    # 各エポックの予測ラベルを1つの DataFrame にまとめる（各列が1エポック分）
    pred_df = pd.concat(predictions_list, axis=1)
    pred_df.columns = [f"epoch_{e}" for e in epochs_found]

    # 隣接するエポック間でラベルが変わっているかを一括で判定
    # diff() は最初の列でNaNになるので、その後の列について比較する
    changes = (pred_df.diff(axis=1).iloc[:, 1:] != 0).astype(int)
    change_counts = changes.sum(axis=1)
    num_transitions = changes.shape[1]

    if ratio and num_transitions > 0:
        label_change = change_counts / num_transitions
    else:
        label_change = change_counts

    result_df = pd.DataFrame({
        "TestData_Index": test_data_index,
        "Label_Change": label_change
    })

    return result_df

def evaluate_label_changes_for_ranges_vectorized(epoch_ranges, ratio=True, csv_dir="."):
    """
    複数のエポック範囲に対して、ベクトル演算を利用した評価を行い、その結果を
    処理対象のCSVファイルが存在するフォルダの親ディレクトリ直下に作成した temporal_stability フォルダ内に、
    各エポック範囲ごとにファイル名に範囲情報を含めたCSVとして保存する関数。

    Parameters
    ----------
    epoch_ranges : list of tuple
        各タプル (start_epoch, end_epoch) の形式でエポック範囲を指定。例: [(1, 1000), (1001, 2000)]
    ratio : bool, default True
        True の場合、変化数を (エポック数 - 1) で割った比率として計算する。
        False の場合は変化数そのままを使用する。
    csv_dir : str, default "."
        各エポックのCSVファイルが存在するディレクトリのパス

    Returns
    -------
    pd.DataFrame or None
        複数範囲の結果をまとめた DataFrame を返す。十分なデータがなければ None を返す。
    """
    # csv_dir の親ディレクトリ直下に temporal_stability フォルダを作成
    parent_dir = os.path.dirname(os.path.abspath(csv_dir))
    output_dir = os.path.join(parent_dir, "temporal_stability")
    os.makedirs(output_dir, exist_ok=True)

    combined_results = []
    for start_epoch, end_epoch in epoch_ranges:
        print(f"Evaluating epoch range: {start_epoch} - {end_epoch}")
        df_result = evaluate_label_changes_vectorized(start_epoch, end_epoch, ratio=ratio, csv_dir=csv_dir)
        if df_result is None:
            print(f"Epoch range {start_epoch}-{end_epoch} の評価結果が得られませんでした。")
            continue

        # 結果に範囲情報を追加（任意）
        df_result["Epoch_Range"] = f"{start_epoch}-{end_epoch}"
        # ファイル名にエポック範囲を含めたパス
        output_file = os.path.join(output_dir, f"label_change_{start_epoch}-{end_epoch}.csv")
        try:
            df_result.to_csv(output_file, index=False)
            print(f"Epoch range {start_epoch}-{end_epoch} の結果を保存しました: {output_file}")
        except Exception as e:
            print(f"CSV の出力に失敗しました: {e}")
        combined_results.append(df_result)

    if combined_results:
        final_df = pd.concat(combined_results, ignore_index=True)
        combined_output_file = os.path.join(output_dir, "combined_label_changes.csv")
        try:
            final_df.to_csv(combined_output_file, index=False)
            print(f"全範囲の結果をまとめたCSVを保存しました: {combined_output_file}")
        except Exception as e:
            print(f"Combined CSV の出力に失敗しました: {e}")
        return final_df
    else:
        print("有効な評価結果が得られませんでした。")
        return None

# 利用例:
if __name__ == "__main__":
    # 複数のエポック範囲で評価を実施（例：1～1000 と 1001～2000）
    epoch_ranges = [(1, 30), (30,60),(60,140),(140,1000)]
    # csv_dir には処理対象のCSVファイルが格納されているディレクトリのパスを指定
    result_df = evaluate_label_changes_for_ranges_vectorized(epoch_ranges, ratio=True, csv_dir="/workspace/test_inference/seed_42width4_cnn_5layers_distribution_colored_emnist_variance0_combined_lr0.01_batch256_epoch1000_LabelNoiseRate0.2_Optimsgd_Momentum0.0/csv")
