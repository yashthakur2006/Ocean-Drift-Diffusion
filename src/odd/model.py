import math
import torch
import torch.nn as nn
from einops import rearrange

def sinusoidal_embedding(timesteps, dim):
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device).float() / half)
    args = timesteps.float()[:,None] * freqs[None]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb

class ResBlock(nn.Module):
    def __init__(self, dim, dim_out, time_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, dim_out)
        )
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv1d(dim, dim_out, 3, padding=1)
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, dim_out),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim_out, dim_out, 3, padding=1)
        )
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, t):
        h = self.block1(x)
        h = h + self.mlp(t)[:,:,None]
        h = self.block2(h)
        return h + self.res_conv(x)

class Down(nn.Module):
    def __init__(self, dim, dim_out, time_dim, dropout=0.0):
        super().__init__()
        self.block = ResBlock(dim, dim_out, time_dim, dropout)
        self.down = nn.Conv1d(dim_out, dim_out, 4, stride=2, padding=1)
    def forward(self, x, t):
        x = self.block(x, t)
        return self.down(x), x

class Up(nn.Module):
    def __init__(self, dim, dim_out, time_dim, dropout=0.0):
        super().__init__()
        self.block = ResBlock(dim+dim_out, dim, time_dim, dropout)
        self.up = nn.ConvTranspose1d(dim, dim, 4, stride=2, padding=1)
    def forward(self, x, skip, t):
        x = torch.cat([x, skip], dim=1)
        x = self.block(x, t)
        return self.up(x)

class TemporalUNet(nn.Module):
    def __init__(self, in_channels=2, base_dim=64, dim_mults=(1,2,4), time_dim=256, dropout=0.1):
        super().__init__()
        dims = [base_dim*m for m in dim_mults]
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        self.init = nn.Conv1d(in_channels, dims[0], 3, padding=1)
        self.downs = nn.ModuleList()
        for i in range(len(dims)-1):
            self.downs.append(Down(dims[i], dims[i+1], time_dim, dropout))
        self.mid = ResBlock(dims[-1], dims[-1], time_dim, dropout)
        self.ups = nn.ModuleList()
        for i in reversed(range(len(dims)-1)):
            self.ups.append(Up(dims[i], dims[i+1], time_dim, dropout))
        self.final = nn.Sequential(
            nn.GroupNorm(8, dims[0]), nn.SiLU(),
            nn.Conv1d(dims[0], in_channels, 3, padding=1)
        )

    def forward(self, x, t):
        # x: (B, C=2, T)
        t = sinusoidal_embedding(t, self.time_dim)
        t = self.time_mlp(t)
        x = self.init(x)
        skips = []
        for d in self.downs:
            x, s = d(x, t)
            skips.append(s)
        x = self.mid(x, t)
        for u in self.ups:
            x = u(x, skips.pop(), t)
        return self.final(x)
