import os
import math
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

def _normalize_latlon(latlon):
    # map lat in [-90,90] and lon in [-180,180] to [-1,1]
    lat = latlon[...,0] / 90.0
    lon = latlon[...,1] / 180.0
    return np.stack([lat, lon], axis=-1)

class SyntheticDrift(Dataset):
    """Generate synthetic drift trajectories influenced by smooth currents and winds.
    Returns sequences of shape (T, 2) in normalized coords [-1,1].
    """
    def __init__(self, n=1024, seq_len=96, horizon=72, dt=1.0):
        self.n = n
        self.seq_len = seq_len
        self.horizon = horizon
        self.dt = dt
        rng = np.random.RandomState(1337)
        self.data = []
        for _ in range(n):
            # initial position around random center
            lat = rng.uniform(-0.5,0.5)
            lon = rng.uniform(-0.5,0.5)
            pos = np.array([lat,lon], dtype=np.float32)
            # smooth pseudo-current field
            def current(p, t):
                u = 0.05*np.sin(2*np.pi*(p[0]+0.01*t)) + 0.02*np.cos(2*np.pi*(p[1]-0.005*t))
                v = 0.05*np.cos(2*np.pi*(p[1]-0.008*t)) + 0.02*np.sin(2*np.pi*(p[0]+0.007*t))
                return np.array([u,v], dtype=np.float32)
            # wind-like perturbation
            def wind(t):
                return 0.01*np.array([np.cos(0.03*t), np.sin(0.025*t)], dtype=np.float32)
            T = seq_len + horizon
            traj = np.zeros((T,2), dtype=np.float32)
            p = pos.copy()
            for t in range(T):
                vel = current(p, t) + wind(t)
                noise = 0.005*rng.randn(2).astype(np.float32)
                p = p + vel*self.dt + noise
                # clamp to [-1,1]
                p = np.clip(p, -1.0, 1.0)
                traj[t] = p
            self.data.append(traj)
        self.data = np.stack(self.data, axis=0)

    def __len__(self): return self.n
    def __getitem__(self, i):
        traj = self.data[i]
        src = traj[:self.seq_len]
        tgt = traj[self.seq_len:self.seq_len+self.horizon]
        return {
            "src": torch.from_numpy(src).float(),  # (S,2)
            "tgt": torch.from_numpy(tgt).float(),  # (H,2)
        }

class NPZTraj(Dataset):
    """Load preprocessed trajectories saved as NPZ with arrays 'src' and 'tgt'."""
    def __init__(self, npz_dir, seq_len=96, horizon=72):
        self.files = sorted([os.path.join(npz_dir,f) for f in os.listdir(npz_dir) if f.endswith('.npz')])
        self.seq_len = seq_len
        self.horizon = horizon

    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        d = np.load(self.files[i])
        src = d['src'][:self.seq_len]
        tgt = d['tgt'][:self.horizon]
        return {"src": torch.tensor(src).float(), "tgt": torch.tensor(tgt).float()}

def make_dataloaders(data, batch_size=32, num_workers=2, seq_len=96, horizon=72):
    if data == "synthetic":
        ds = SyntheticDrift(n=1024, seq_len=seq_len, horizon=horizon)
        val = SyntheticDrift(n=128, seq_len=seq_len, horizon=horizon)
    else:
        ds = NPZTraj(data, seq_len=seq_len, horizon=horizon)
        val = ds
    train_loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, val_loader
