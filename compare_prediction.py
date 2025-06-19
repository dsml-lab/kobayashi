import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import load_models
from datasets import load_datasets
from utils import set_seed, set_device
from config import parse_args_model_save
from pathlib import Path

def get_correct_indices(model, dataloader, device):
    model.eval()
    correct_indices = []
    idx_offset = 0
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct = (preds == y).cpu().numpy()
            indices = np.arange(idx_offset, idx_offset + len(y))
            correct_indices.extend(indices[correct])
            idx_offset += len(y)
    return np.array(correct_indices)

def save_correct_indices(model_path, model, test_loader, device, output_path):
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    indices = get_correct_indices(model, test_loader, device)
    np.save(output_path, indices)
    print(f"[✓] Saved correct indices to {output_path}")
    return indices

def compare_correct_indices(path1, path2):
    idx1 = set(np.load(path1))
    idx2 = set(np.load(path2))
    gained = sorted(list(idx2 - idx1))
    lost = sorted(list(idx1 - idx2))
    print(f"[Gained] {len(gained)} samples became correct in epoch2")
    print(f"[Lost]   {len(lost)} samples became incorrect in epoch2")
    return gained, lost

def main():
    args = parse_args_model_save()
    set_seed(args.fix_seed)
    device = set_device(args.gpu)

    # 比較対象epoch（必要に応じて変更）
    epoch1 = 80
    epoch2 = 2000

    # === 実験名と保存ディレクトリ（compare_test配下） ===
    experiment_name = f"seed_{args.fix_seed}width{args.model_width}_{args.model}_{args.dataset}_variance{args.variance}_{args.target}_lr{args.lr}_batch{args.batch_size}_epoch{args.epoch}_LabelNoiseRate{args.label_noise_rate}_Optim{args.optimizer}_Momentum{args.momentum}"
    model_dir = Path("save_model") / args.dataset / f"noise_{args.label_noise_rate}" / experiment_name
    save_dir = Path("compare_test") / experiment_name
    save_dir.mkdir(parents=True, exist_ok=True)

    path1 = save_dir / f"correct_epoch_{epoch1}.npy"
    path2 = save_dir / f"correct_epoch_{epoch2}.npy"

    # === テストデータロード ===
    _, test_dataset, img_size, num_classes, in_channels = load_datasets(args.dataset, args.target, args.gray_scale, args)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # === モデル定義・保存インデックス生成 ===
    model = load_models(in_channels, args, img_size, num_classes).to(device)

    model_path1 = model_dir / f"model_epoch_{epoch1}.pth"
    model_path2 = model_dir / f"model_epoch_{epoch2}.pth"
    save_correct_indices(model_path1, model, test_loader, device, path1)
    save_correct_indices(model_path2, model, test_loader, device, path2)

    # === 比較・保存 ===
    gained, lost = compare_correct_indices(path1, path2)
    np.save(save_dir / f"gained_{epoch1}_to_{epoch2}.npy", gained)
    np.save(save_dir / f"lost_{epoch1}_to_{epoch2}.npy", lost)

if __name__ == "__main__":
    main()

# python compare_prediction.py \
#   --fix_seed 42 \
#   --model resnet18k \
#   --model_width 64 \
#   --epoch 4000 \
#   --dataset cifar10 \
#   --target color \
#   --label_noise_rate 0.2 \
#   --batch_size 128 \
#   --lr 0.0001 \
#   --optimizer adam \
#   --momentum 0.0 \
#   --loss cross_entropy \
#   --gpu 0 \
#   --num_workers 4 \
#   --variance 0 \
#   --correlation 0.5 \
#   --wandb \
#   --wandb_project kobayashi_save_model \
#   --wandb_entity dsml-kernel24
