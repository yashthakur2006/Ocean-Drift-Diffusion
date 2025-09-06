import argparse, os, json
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from .data import make_dataloaders
from .model import TemporalUNet
from .diffusion import DDPM
from .utils import load_config, set_seed

def ade(pred, gt):
    # pred, gt: (B,H,2)
    return np.linalg.norm(pred - gt, axis=-1).mean()

def crps_1d(samples, y):
    # samples: (N,), y: scalar
    xs = np.sort(samples)
    N = len(xs)
    # empirical CDF at each unique x_i
    F = np.arange(1, N+1)/N
    # integral approximation using trapezoids
    # CRPS = \int (F(x) - 1{x>=y})^2 dx
    grid = xs
    indicator = (grid >= y).astype(float)
    diff = F - indicator
    # approximate integral with trapezoidal rule over grid deltas
    deltas = np.diff(grid, prepend=grid[0])
    return np.sum(diff**2 * deltas)

def pit_values(samples, y):
    # samples: (N,), y scalar -> PIT=F(y)
    return (samples <= y).mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/default.yaml')
    p.add_argument('--data', default='synthetic')
    p.add_argument('--checkpoint', default='models/best.pt')
    p.add_argument('--num-samples', type=int, default=64)
    p.add_argument('--out', default='outputs')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = load_config(args.cfg)
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")
    train_loader, val_loader = make_dataloaders(args.data, cfg.batch_size, cfg.num_workers, cfg.seq_len, cfg.horizon)

    model = TemporalUNet(in_channels=2, base_dim=cfg.dim, dim_mults=tuple(cfg.dim_mults), dropout=cfg.dropout).to(device)
    ddpm = DDPM(model, timesteps=cfg.timesteps, schedule=cfg.beta_schedule, device=str(device))

    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    # Evaluate on one batch
    batch = next(iter(val_loader))
    src = batch['src'].to(device)   # (B,S,2)
    tgt = batch['tgt'].to(device)   # (B,H,2)
    B, S, _ = src.shape
    H = tgt.shape[1]

    # sample trajectories conditioned only implicitly (model trained on whole window)
    # we generate fresh sequences of length S+H and take last H as forecast
    samples = []
    with torch.no_grad():
        for _ in tqdm(range(args.num-samples if hasattr(args,'num-samples') else args.num_samples)):
            x = ddpm.sample((B,2,S+H), steps=cfg.sample_steps)  # (B,2,T)
            x = x.transpose(1,2)  # (B,T,2)
            samples.append(x[:, -H:, :].cpu().numpy())
    samples = np.stack(samples, axis=0)  # (N,B,H,2)

    # Metrics: ADE (using mean sample), CRPS per dim, PIT
    mean_pred = samples.mean(axis=0)  # (B,H,2)
    ade_val = ade(mean_pred, tgt.cpu().numpy())

    # CRPS + PIT for lat and lon separately
    crps_vals = []
    pits = []
    for b in range(B):
        for h in range(H):
            for d in range(2):
                s = samples[:,b,h,d]
                y = tgt[b,h,d].cpu().item()
                # simple grid-based approx: if degenerate grid, skip
                if np.allclose(s.max(), s.min()):
                    continue
                crps_vals.append(crps_1d(s, y))
                pits.append(pit_values(s, y))
    crps_mean = float(np.mean(crps_vals)) if crps_vals else float("nan")

    # PIT histogram
    pits = np.array(pits)
    plt.figure(figsize=(5,3))
    plt.hist(pits, bins=20, density=True)
    plt.title("PIT Histogram")
    plt.xlabel("PIT")
    plt.ylabel("Density")
    plt.tight_layout()
    pit_path = os.path.join(args.out, "pit_hist.png")
    plt.savefig(pit_path, dpi=150)

    # Qual plot
    plt.figure(figsize=(6,4))
    for b in range(min(B,5)):
        gt = tgt[b].cpu().numpy()
        pr = mean_pred[b]
        plt.plot(gt[:,1], gt[:,0], label=f"gt-{b}", alpha=0.8)
        plt.plot(pr[:,1], pr[:,0], '--', label=f"pred-{b}", alpha=0.8)
    plt.legend(ncol=2, fontsize=8)
    plt.xlabel("lon (norm)"); plt.ylabel("lat (norm)")
    plt.tight_layout()
    qual_path = os.path.join(args.out, "qualitative.png")
    plt.savefig(qual_path, dpi=150)

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump({"ADE": ade_val, "CRPS": crps_mean}, f, indent=2)

    print(f"Saved metrics to {args.out}/metrics.json, plots to {pit_path} and {qual_path}")

if __name__ == "__main__":
    main()
