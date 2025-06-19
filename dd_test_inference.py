# test_inference.py
import os
import csv
import torch
import torch.nn.functional as F
import warnings

from config import parse_args_model_save
from models import load_models
from datasets import load_datasets
from utils import set_seed, set_device

def run_test_inference(save_fields=None):
    """
    save_fields: list of str, 保存する項目を指定する
      指定例: ["index", "label", "loss", "prob", "true", "correct"]
      - "index": サンプル番号
      - "label": 予測ラベル
      - "loss": loss値
      - "prob": 各クラスの確率
      - "true": データ点に付いている正解ラベル
      - "correct": 予測が正解ラベルと一致しているかどうか
    """
    warnings.filterwarnings("ignore")
    
    # 引数のパース、シード設定、デバイス設定
    args = parse_args_model_save()
    set_seed(args.fix_seed)
    device = set_device(args.gpu)
    print(f"Using device: {device}")
    
    # 保存項目の指定（引数がNoneならデフォルト値を使用）
    if save_fields is None:
        save_fields = ["index", "label", "loss", "prob", "true", "correct"]
    save_fields = [field.strip().lower() for field in save_fields]
    print("Selected save fields:", save_fields)
    
    # テストデータの読み込み
    print("Loading test dataset...")
    _, test_dataset, imagesize, num_classes, in_channels = load_datasets(
        args.dataset, args.target, args.gray_scale, args
    )
    
    # モデルの初期化（重みは各エポック毎にチェックポイントから読み込む）
    model = load_models(in_channels, args, imagesize, num_classes)
    model.to(device)
    
    # experiment_name の生成（main.pyと同一の形式）
    experiment_name = (
        f'seed_{args.fix_seed}width{args.model_width}_{args.model}_'
        f'{args.dataset}_variance{args.variance}_{args.target}_'
        f'lr{args.lr}_batch256_epoch{args.epoch}_'
        f'LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}'
    )
    tyukan_dir = "noise_rate=0_sigma=0"  # 固定値
    
    # DataLoaderの作成（テスト時はshuffleはFalse）
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # 1～指定されたエポックまで繰り返し推論を実行
    for current_epoch in range(1, args.epoch + 1):
        checkpoint_path = os.path.join(
            "save_model", "Colored_EMSNIT", tyukan_dir, experiment_name,
            f"model_epoch_{current_epoch}.pth"
        )
        if not os.path.exists(checkpoint_path):
            print(f"Error: checkpoint '{checkpoint_path}' does not exist. Skipping epoch {current_epoch}.")
            continue
        
        # チェックポイントの読み込み
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")
        
        results = []
        sample_index = 0
        
        with torch.no_grad():
            for batch in test_loader:
                # (inputs, labels) のタプルが返ると想定
                inputs, labels = batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)  # shape: (batch_size, num_classes)
                softmax_outputs = F.softmax(outputs, dim=1)
                predicted = torch.argmax(softmax_outputs, dim=1)
                losses = F.cross_entropy(outputs, labels, reduction='none')
                
                batch_size = inputs.size(0)
                for i in range(batch_size):
                    result = {}
                    if "index" in save_fields:
                        result["TestData_Index"] = sample_index
                    if "label" in save_fields:
                        result["Predicted_Label"] = predicted[i].item()
                    if "loss" in save_fields:
                        result["Loss"] = losses[i].item()
                    if "prob" in save_fields:
                        for cls in range(num_classes):
                            result[f"Prob_Class_{cls}"] = softmax_outputs[i, cls].item()
                    if "true" in save_fields:
                        result["True_Label"] = labels[i].item()
                    if "correct" in save_fields:
                        result["Correct"] = (predicted[i].item() == labels[i].item())
                    results.append(result)
                    sample_index += 1
        
        # ヘッダーを保存項目に合わせて構築
        header = []
        if "index" in save_fields:
            header.append("TestData_Index")
        if "label" in save_fields:
            header.append("Predicted_Label")
        if "loss" in save_fields:
            header.append("Loss")
        if "prob" in save_fields:
            for cls in range(num_classes):
                header.append(f"Prob_Class_{cls}")
        if "true" in save_fields:
            header.append("True_Label")
        if "correct" in save_fields:
            header.append("Correct")
        
        output_csv = f"test_inferenc_epoch_{current_epoch}.csv"
        output_dir = os.path.join("test_inference", experiment_name, "csv")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, output_csv)
        
        with open(output_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=header)
            writer.writeheader()
            for row in results:
                writer.writerow(row)
        
        print(f"Test inference results saved to {output_path}")

if __name__ == "__main__":
    # 例: 以下のように保存項目を関数引数で指定可能
    run_test_inference(save_fields=["index", "label", "loss", "true", "correct"])
