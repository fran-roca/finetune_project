import torch

def get_device() -> str:
    """
    Returns the appropriate device string: 'cuda' if an NVIDIA CUDA GPU is available, otherwise 'cpu'.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"
