import torch
import numpy as np

def to_tensor(data: np.ndarray) -> torch.Tensor:
    """
    Convert numpy array to PyTorch tensor.
    For complex arrays, the real and imaginary parts are stacked along the last
    dimension.
    Args:
        data: Input numpy array.
    Returns:
        PyTorch version of data.
    """
    if np.iscomplexobj(data):
        data = np.stack((data.real, data.imag), axis=-1)

    return torch.from_numpy(data)
