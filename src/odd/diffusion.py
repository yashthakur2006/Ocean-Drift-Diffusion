import torch
import torch.nn.functional as F
from dataclasses import dataclass
import math

@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    schedule: str = "cosine"  # or "linear"

def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps
    x = torch.linspace(0, timesteps, steps+1)
    alphas_cumprod = torch.cos(((x/timesteps)+s)/(1+s)*math.pi/2)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

def linear_beta_schedule(timesteps, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

class DDPM:
    def __init__(self, model, timesteps=1000, schedule="cosine", device="cpu"):
        self.model = model
        self.T = timesteps
        self.device = device
        betas = cosine_beta_schedule(timesteps) if schedule=="cosine" else linear_beta_schedule(timesteps)
        self.register(betas.to(device))

    def register(self, betas):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=betas.device), self.alphas_cumprod[:-1]], dim=0)
        self.posterior_variance = self.betas * (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod)

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha = self.sqrt_alphas_cumprod[t][:,None,None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:,None,None]
        return sqrt_alpha * x0 + sqrt_one_minus * noise

    def p_losses(self, x0, t):
        noise = torch.randn_like(x0)
        xt = self.q_sample(x0, t, noise)
        pred_noise = self.model(xt, t)
        return F.mse_loss(pred_noise, noise)

    @torch.no_grad()
    def p_sample(self, xt, t):
        betas_t = self.betas[t][:,None,None]
        sqrt_one_minus = self.sqrt_one_minus_alphas_cumprod[t][:,None,None]
        sqrt_recip_alpha = (1.0 / torch.sqrt(self.alphas[t]))[:,None,None]
        model_mean = sqrt_recip_alpha*(xt - betas_t/sqrt_one_minus * self.model(xt,t))
        if (t == 0).all():
            return model_mean
        noise = torch.randn_like(xt)
        var = torch.sqrt(self.posterior_variance[t])[:,None,None]
        return model_mean + var*noise

    @torch.no_grad()
    def sample(self, shape, steps=None):
        B, C, T = shape
        xt = torch.randn(shape, device=self.device)
        if steps is None or steps>=self.T:
            ts = torch.arange(self.T-1, -1, -1, device=self.device)
        else:
            # subsample timesteps uniformly
            ts = torch.linspace(self.T-1, 0, steps, device=self.device).long()
        for ti in ts:
            t = torch.full((B,), int(ti), device=self.device, dtype=torch.long)
            xt = self.p_sample(xt, t)
        return xt
