import torch

device = "gpu" if torch.cuda.is_available() else "cpu"
