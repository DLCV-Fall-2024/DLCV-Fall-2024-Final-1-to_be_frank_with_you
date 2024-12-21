from typing import List, Tuple, Dict, Any

import torch
from transformers.feature_extraction_utils import BatchFeature
from copy import deepcopy


def default(val, default_val):
    return val if val is not None else default_val


# Call `to` for tensors in List or Dict recursively
def container_to(container, target_dtypes: List[torch.dtype] = [], device: torch.device = None, dtype: torch.dtype = None):
    if isinstance(container, torch.Tensor):
        if container.dtype in target_dtypes:
            return container.to(device=device, dtype=dtype)
        else:
            return container.to(device=device)
    elif isinstance(container, (list, tuple)):
        return type(container)(
            container_to(item, target_dtypes, device=device, dtype=dtype) for item in container
        )
    elif isinstance(container, dict):
        return {
            key: container_to(value, target_dtypes, device=device, dtype=dtype)
            for key, value in container.items()
        }
    return container


# Concatenate tensors in different List or Dict recursively
def iterate_container(container: Dict | List | Tuple | torch.Tensor | Any):
    if isinstance(container, torch.Tensor):
        yield (container, [])
    elif isinstance(container, (list, tuple)):
        for i, item in enumerate(container):
            for tensor, paths in iterate_container(item):
                yield (tensor, [i] + paths)
    elif isinstance(container, dict):
        for key, value in container.items():
            for tensor, paths in iterate_container(value):
                yield (tensor, [key] + paths)
    else:
        raise ValueError(f"Unsupported type: {type(container)}")


def container_cat(containers: List[Dict | List | Tuple | torch.Tensor], dim: int = 0):
    if isinstance(containers[0], torch.Tensor):
        return torch.cat(containers, dim=dim)
    
    cated = deepcopy(containers[0])
    
    iters = [iterate_container(container) for container in containers]
    while True:
        iter_values = [next(iter, (None, [])) for iter in iters]
        if all(tensor is None for tensor, _ in iter_values):
            break
        tensors = [tensor for tensor, _ in iter_values]
        paths = iter_values[0][1]

        tensor = torch.cat(tensors, dim=dim)
        _cated = cated
        for path in paths[:-1]:
            _cated = _cated[path]
        _cated[paths[-1]] = tensor

    return cated

# if __name__ == "__main__":
#     C = {
#         "a": [torch.randn(1, 3), torch.randn(1, 2)],
#         "b": [torch.randn(1, 3), torch.randn(1, 2)],
#     }
#     print(container_cat([C, C], dim=0))


def batch_feature_to_dict(
    container: Dict | List | Tuple | torch.Tensor | BatchFeature | Any,
):
    if isinstance(container, BatchFeature) or isinstance(container, dict):
        return {key: batch_feature_to_dict(value) for key, value in container.items()}
    elif isinstance(container, (list, tuple)):
        return type(container)(batch_feature_to_dict(item) for item in container)
    elif isinstance(container, torch.Tensor):
        return container
    else:
        raise ValueError(f"Unsupported type: {type(container)}")
