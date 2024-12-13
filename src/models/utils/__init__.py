import torch

from .adaln_zero import AdaLNZero


def ensure_all_on_device(model: torch.nn.Module, device: torch.device, depth=2):

    model = model.to(device)
    for module in model.modules():
        if depth > 0:
            ensure_all_on_device(module, device, depth - 1)
        module.to(device)


def ensure_all_same_dtype(model: torch.nn.Module, dtype: torch.dtype, depth=2):
    model = model.to(dtype)
    for module in model.modules():
        if depth > 0:
            ensure_all_same_dtype(module, dtype, depth - 1)
        module.to(dtype)
