import os
import torch
def compare_batches(dir1, dir2):
    # 各ディレクトリから "batch_*.pt" ファイルを昇順で取得
    files1 = sorted([f for f in os.listdir(dir1) if f.startswith("batch_") and f.endswith(".pt")])
    files2 = sorted([f for f in os.listdir(dir2) if f.startswith("batch_") and f.endswith(".pt")])
    
    # 両ディレクトリに共通するファイル名を抽出
    common_files = sorted(set(files1).intersection(files2))
    if not common_files:
        print("共通のバッチファイルが見つかりません。")
        return
    
    for filename in common_files:
        print(f"--- {filename} の内容比較 ---")
        file_path1 = os.path.join(dir1, filename)
        file_path2 = os.path.join(dir2, filename)
        data1 = torch.load(file_path1)
        data2 = torch.load(file_path2)
        
        # 比較するキー
        keys = ["inputs", "labels", "noise_flags"]
        for key in keys:
            v1 = data1.get(key)
            v2 = data2.get(key)
            
            if v1 is None or v2 is None:
                print(f"キー '{key}' はどちらかのディレクトリに存在しません (dir1: {v1 is not None}, dir2: {v2 is not None})")
                continue
            
            # テンソルの場合、shape を出力
            shape1 = v1.shape if hasattr(v1, "shape") else "N/A"
            shape2 = v2.shape if hasattr(v2, "shape") else "N/A"
            
            # サンプルの比較
            if v1.dtype.is_floating_point:
                same = torch.allclose(v1, v2, atol=1e-6)
            else:
                same = torch.equal(v1, v2)
            
            # print(f"キー '{key}':")
            # print(f"  dir1 shape: {shape1}")
            # print(f"  dir2 shape: {shape2}")
            print(f"  一致: {same}")
            
            # 先頭のサンプル（例: 先頭5個）の内容を表示（リストに変換して出力）
            if hasattr(v1, "tolist"):
                samples1 = v1.tolist()[:5]
                samples2 = v2.tolist()[:5]
                # print(f"  dir1 先頭5サンプル: {samples1}")
                # print(f"  dir2 先頭5サンプル: {samples2}")
        print("\n")
        

if __name__ == "__main__":
    # ここにそれぞれのディレクトリパスを指定してください
    dir1 = "/workspace/save_model/cifar10/noise_0.0/seed_42width1_resnet18k_cifar10_variance0_color_lr0.0001_batch128_epoch3_LabelNoiseRate0.0_Optimadam_Momentum0.0/csv/batch_log/epoch_3"
    dir2 = "/workspace/save_model/cifar10/noise_0.0/seed_42width2_resnet18k_cifar10_variance0_color_lr0.0001_batch128_epoch3_LabelNoiseRate0.0_Optimadam_Momentum0.0/csv/batch_log/epoch_3"
    compare_batches(dir1, dir2)
