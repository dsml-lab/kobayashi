import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from functorch import make_functional, vmap, grad

# def compute_fraction_of_loss_reduction_from_batch_func_v112(model, criterion, batch, device):
#     x, y, noise_flag = [b.to(device) for b in batch]
#     batch_size = x.size(0)

#     from functorch import make_functional, vmap, grad
#     fmodel, params = make_functional(model)

#     def compute_loss(p, x_i, y_i):
#         out = fmodel(p, x_i.unsqueeze(0))
#         loss_i = criterion(out, y_i.unsqueeze(0))
#         return loss_i.squeeze(0)

#     grads = vmap(grad(compute_loss), in_dims=(None, 0, 0))(params, x, y)
#     grads_flat = torch.cat([g.view(batch_size, -1) for g in grads], dim=1)  # (B, D)

#     # g_total は backward なしで、メモリ消費を抑える
#     with torch.no_grad():
#         model.zero_grad()
#         outputs = model(x)
#         losses = criterion(outputs, y)
#         losses.sum().backward()
#         g_total = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).detach()

#     # 勾配合計は一旦 CPU に逃がしてから和を取る（OOM 回避）
#     g_pristine = grads_flat[noise_flag == 0].cpu().sum(dim=0).to(device)
#     g_corrupt  = grads_flat[noise_flag == 1].cpu().sum(dim=0).to(device)

#     norm_sq = g_total.norm()**2 + 1e-8
#     fp = torch.dot(g_total, g_pristine) / norm_sq
#     fc = torch.dot(g_total, g_corrupt) / norm_sq

#     print(f"[DEBUG] fp = {fp.item():.6f}, fc = {fc.item():.6f}, sum = {fp.item() + fc.item():.6f}")
#     return fp.item(), fc.item()
# def compute_fraction_of_loss_reduction_from_batch(model, criterion, batch, device):
#     model.zero_grad()
#     inputs, labels, noise_flags = [x.to(device) for x in batch]
#     outputs = model(inputs)
#     loss = criterion(outputs, labels)
#     loss.mean().backward(retain_graph=True)
#     g_total = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).detach()

#     g_pristine = torch.zeros_like(g_total)
#     g_corrupt = torch.zeros_like(g_total)

#     for i in range(len(inputs)):
#         model.zero_grad()
#         output_i = model(inputs[i].unsqueeze(0))
#         loss_i = criterion(output_i, labels[i].unsqueeze(0))
#         loss_i.backward(retain_graph=True)
#         g_i = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).detach()
#         if noise_flags[i].item() == 0:
#             g_pristine += g_i
#         else:
#             g_corrupt += g_i

#     g_norm_sq = g_total.norm()**2 + 1e-8
#     f_p = torch.dot(g_total, g_pristine) / g_norm_sq
#     f_c = torch.dot(g_total, g_corrupt) / g_norm_sq
#     return f_p.item(), f_c.item()
import torch
import torch.nn as nn

def compute_fraction_of_loss_reduction_from_batch(model, criterion, batch, device):
    """
    Compute f_p and f_c as described in Coherent Gradients (ICLR 2020),
    and verify that f_p + f_c ≈ 1.

    Parameters:
    - model: PyTorch model
    - criterion: loss function (must be reduction='none')
    - batch: tuple of (inputs, labels, noise_flags) where noise_flags == 0 (pristine), 1 (corrupt)
    - device: computation device ('cuda' or 'cpu')

    Returns:
    - f_p: float, pristine gradient contribution
    - f_c: float, corrupt gradient contribution
    """
    model.zero_grad()
    inputs, labels, noise_flags = [x.to(device) for x in batch]

    # Compute full-batch gradient g_t
    outputs = model(inputs)
    losses = criterion(outputs, labels)  # shape: [batch_size]
    total_loss = losses.sum()
    total_loss.backward(retain_graph=True)
    g_t = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).detach()

    # Initialize gradients for pristine and corrupt subsets
    g_p = torch.zeros_like(g_t)
    g_c = torch.zeros_like(g_t)

    for i in range(len(inputs)):
        model.zero_grad()
        output_i = model(inputs[i].unsqueeze(0))
        loss_i = criterion(output_i, labels[i].unsqueeze(0))
        loss_i.backward()

        g_i = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).detach()

        if noise_flags[i].item() == 0:
            g_p += g_i
        else:
            g_c += g_i

    # Compute contributions
    g_t_norm_sq = torch.dot(g_t, g_t) + 1e-8  # avoid divide-by-zero
    f_p = torch.dot(g_t, g_p) / g_t_norm_sq
    f_c = torch.dot(g_t, g_c) / g_t_norm_sq

    # ✅ Check f_p + f_c ≈ 1
    sum_fc = f_p + f_c
    print(f"f_p (pristine contribution): {f_p:.6f}")
    print(f"f_c (corrupt  contribution): {f_c:.6f}")
    print(f"Sum (should be 1.0):         {sum_fc:.6f}")

    if abs(sum_fc - 1.0) < 1e-5:
        print("✅ f_p + f_c ≈ 1: Consistency confirmed.")
    else:
        print("⚠️ Warning: f_p + f_c != 1, check gradient computation.")

    return f_p.item(), f_c.item()


def evaluate_fp_fc_npy(
    model_class,
    npy_dir,
    model_dir,
    output_csv_path,
    device="cuda",
    batch_size=128,
    max_batches=5,
    epoch_start=0,
    epoch_end=None
):
    # === .npy ファイルから読み込み ===
    print("[INFO] Loading data from npy files...")
    x = np.load(os.path.join(npy_dir, "x_train.npy"))
    y = np.load(os.path.join(npy_dir, "y_train.npy"))
    noise_info = np.load(os.path.join(npy_dir, "noisy_info.npy"))
    print(f"[INFO] Loaded {len(x)} samples.")
    x = torch.tensor(x).float()
    y = torch.tensor(y).long()
    noise_info = torch.tensor(noise_info).long()

    dataset = TensorDataset(x, y, noise_info)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    criterion = nn.CrossEntropyLoss(reduction='none')

    # モデルファイル一覧を取得し、エポック範囲でフィルタ
    model_paths = sorted(
        [f for f in os.listdir(model_dir) if f.startswith("model_epoch_") and f.endswith(".pth")],
        key=lambda x: int(x.split("_")[-1].split(".")[0])
    )
    if epoch_end is not None:
        model_paths = [
            f for f in model_paths
            if epoch_start <= int(f.split("_")[-1].split(".")[0]) <= epoch_end
        ]
    else:
        model_paths = [
            f for f in model_paths
            if int(f.split("_")[-1].split(".")[0]) >= epoch_start
        ]

    results = []
    for model_file in tqdm(model_paths, desc="Evaluating fp/fc per epoch"):
        epoch = int(model_file.split("_")[-1].split(".")[0])
        model = model_class().to(device)
        state = torch.load(os.path.join(model_dir, model_file), map_location=device)
        model.load_state_dict(state["model_state_dict"])
        model.eval()

        f_p_list, f_c_list = [], []
        for i, batch in enumerate(loader):
            f_p, f_c = compute_fraction_of_loss_reduction_from_batch(model, criterion, batch, device)
            f_p_list.append(f_p)
            f_c_list.append(f_c)
            if i + 1 >= max_batches:
                break
        print(f"Epoch {epoch}: f_p = {np.mean(f_p_list)}, f_c = {np.mean(f_c_list)}")
        results.append({
            "epoch": epoch,
            "fp": sum(f_p_list) / len(f_p_list),
            "fc": sum(f_c_list) / len(f_c_list)
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv_path, index=False)
    print(f"[Done] Saved fp/fc results to: {output_csv_path}")


# ===== 実行部 =====
if __name__ == "__main__":
    import argparse
    from models import load_models
    from utils import set_device

    parser = argparse.ArgumentParser()
    parser.add_argument("--npy_dir", type=str, required=True)
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--output_csv", type=str, default="fp_fc_results.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_batches", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--fix_seed", type=int, default=42)
    parser.add_argument("--epoch_start", type=int, default=0)
    parser.add_argument("--epoch_end", type=int, default=None)

    parser.add_argument("--model", type=str, default="cnn_5layers")
    parser.add_argument("--model_width", type=int, default=4)
    parser.add_argument("--variance", type=int, default=0)
    parser.add_argument("--label_noise_rate", type=float, default=0.2)
    parser.add_argument("--target", type=str, default="combined")

    args = parser.parse_args()
    args.device = set_device(args.gpu)
    torch.manual_seed(args.fix_seed)

    # モデル構造（DataLoaderは自前で渡すので imagesize, in_channels は固定）
    model_class = lambda: load_models(
        3,                         # in_channels
        args,                      # args
        (32, 32),                  # img_size
        100 if args.target == "combined" else 10  # num_classes
    )

    evaluate_fp_fc_npy(
        model_class=model_class,
        npy_dir=args.npy_dir,
        model_dir=args.model_dir,
        output_csv_path=args.output_csv,
        device=args.device,
        batch_size=args.batch_size,
        max_batches=args.max_batches,
        epoch_start=args.epoch_start,
        epoch_end=args.epoch_end
    )