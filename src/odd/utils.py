import os, random
import numpy as np
import torch
from dataclasses import dataclass
import yaml

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    epochs: int = 50
    batch_size: int = 32
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    amp: bool = True
    seq_len: int = 96
    horizon: int = 72
    grid_size: int = 128
    num_workers: int = 2
    timesteps: int = 1000
    beta_schedule: str = "cosine"
    sample_steps: int = 50
    dim: int = 64
    dim_mults: tuple = (1,2,4)
    dropout: float = 0.1

def load_config(path: str) -> Config:
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
    base = Config()
    for k,v in raw.items():
        if hasattr(base, k):
            setattr(base, k, v)
    return base
