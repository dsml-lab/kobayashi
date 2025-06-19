# utils.py

import torch
import random
import numpy as np
import gc

def set_seed(seed):
    """
    Set the seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For GPU determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def set_device(gpu_id):
    """
    Sets the device for computation.

    Args:
        gpu_id (int): The ID of the GPU to use.

    Returns:
        torch.device: The selected device (GPU or CPU).
    """
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    return device

def clear_memory():
    """
    Clears CUDA cache and forces garbage collection.
    """
    torch.cuda.empty_cache()
    gc.collect()

def apply_transform(x, transform):
    """
    Applies a transformation to a batch of images.

    Args:
        x (np.ndarray or list): Batch of images.
        transform (torchvision.transforms.Compose): Transformation to apply.

    Returns:
        torch.Tensor: Transformed images.
    """
    transformed_x = []
    for img in x:
        img = transform(img)
        transformed_x.append(img)
    return torch.stack(transformed_x)

def compute_fraction_of_loss_reduction_from_batch(model, criterion, batch, device):
    """
    ミニバッチ中のpristine/corrupt別の勾配寄与率 (f_p, f_c) を返す。
    batch: (inputs, labels, noise_flags)
    """
    model.zero_grad()
    inputs, labels, noise_flags = [x.to(device) for x in batch]
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.mean().backward(retain_graph=True)
    g_total = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).detach()

    g_pristine = torch.zeros_like(g_total)
    g_corrupt = torch.zeros_like(g_total)

    for i in range(len(inputs)):
        model.zero_grad()
        output_i = model(inputs[i].unsqueeze(0))
        loss_i = criterion(output_i, labels[i].unsqueeze(0))
        loss_i.backward(retain_graph=True)
        g_i = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None]).detach()
        if noise_flags[i].item() == 0:
            g_pristine += g_i
        else:
            g_corrupt += g_i

    g_norm_sq = g_total.norm()**2 + 1e-8
    f_p = torch.dot(g_total, g_pristine) / g_norm_sq
    f_c = torch.dot(g_total, g_corrupt) / g_norm_sq
    return f_p.item(), f_c.item()