import argparse, os, numpy as np, torch
from .model import TemporalUNet
from .diffusion import DDPM
from .utils import load_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/default.yaml')
    p.add_argument('--checkpoint', default='models/best.pt')
    p.add_argument('--horizon', type=int, default=None)
    p.add_argument('--num-samples', type=int, default=128)
    p.add_argument('--out', default='outputs/samples.npz')
    args = p.parse_args()

    cfg = load_config(args.cfg)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")
    model = TemporalUNet(in_channels=2, base_dim=cfg.dim, dim_mults=tuple(cfg.dim_mults), dropout=cfg.dropout).to(device)
    ddpm = DDPM(model, timesteps=cfg.timesteps, schedule=cfg.beta_schedule, device=str(device))
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    H = args.horizon or cfg.horizon
    B = 8
    with torch.no_grad():
        xs = ddpm.sample((args.num_samples, 2, H), steps=cfg.sample_steps).cpu().numpy()  # (N,2,H)
        xs = np.transpose(xs, (0,2,1))  # (N,H,2)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez_compressed(args.out, traj=xs)
    print(f"Saved {args.num_samples} trajectories to {args.out}")

if __name__ == "__main__":
    main()
