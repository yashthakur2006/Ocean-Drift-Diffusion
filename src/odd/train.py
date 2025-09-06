import argparse, os, json
import torch
from torch import optim
from tqdm import tqdm
from .data import make_dataloaders
from .model import TemporalUNet
from .diffusion import DDPM
from .utils import set_seed, load_config

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cfg', default='configs/default.yaml')
    p.add_argument('--data', default='synthetic', help='"synthetic" or path to NPZ dir')
    p.add_argument('--epochs', type=int, default=None)
    p.add_argument('--batch-size', type=int, default=None)
    p.add_argument('--device', default=None)
    p.add_argument('--outdir', default='models')
    p.add_argument('--deterministic', action='store_true')
    args = p.parse_args()

    cfg = load_config(args.cfg)
    if args.epochs: cfg.epochs = args.epochs
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.device: cfg.device = args.device

    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() and cfg.device=="cuda" else "cpu")
    if args.deterministic:
        torch.use_deterministic_algorithms(True)

    train_loader, val_loader = make_dataloaders(args.data, cfg.batch_size, cfg.num_workers, cfg.seq_len, cfg.horizon)

    model = TemporalUNet(in_channels=2, base_dim=cfg.dim, dim_mults=tuple(cfg.dim_mults), dropout=cfg.dropout).to(device)
    ddpm = DDPM(model, timesteps=cfg.timesteps, schedule=cfg.beta_schedule, device=str(device))
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type=="cuda")

    os.makedirs(args.outdir, exist_ok=True)
    best_val = 1e9
    for epoch in range(cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"epoch {epoch+1}/{cfg.epochs}")
        total = 0.0
        for batch in pbar:
            src = batch['src'].to(device)   # (B,S,2)
            tgt = batch['tgt'].to(device)   # (B,H,2)
            # concatenate along time to train on S+H window
            x0 = torch.cat([src, tgt], dim=1)   # (B, S+H, 2)
            x0 = x0.transpose(1,2)              # (B,2,T)
            t = torch.randint(0, ddpm.T, (x0.size(0),), device=device).long()

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                loss = ddpm.p_losses(x0, t)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(opt)
            scaler.update()
            total += loss.item()
            pbar.set_postfix(loss=loss.item())

        # simple val pass
        model.eval()
        with torch.no_grad():
            vtotal = 0.0
            for batch in val_loader:
                src = batch['src'].to(device)
                tgt = batch['tgt'].to(device)
                x0 = torch.cat([src, tgt], dim=1).transpose(1,2)
                t = torch.randint(0, ddpm.T, (x0.size(0),), device=device).long()
                vtotal += ddpm.p_losses(x0,t).item()
            vtotal /= max(1,len(val_loader))
        print(f"val loss: {vtotal:.6f}")
        # checkpoint
        ckpt = os.path.join(args.outdir, 'latest.pt')
        torch.save({'model': model.state_dict(), 'cfg': vars(cfg)}, ckpt)
        if vtotal < best_val:
            best_val = vtotal
            torch.save({'model': model.state_dict(), 'cfg': vars(cfg)}, os.path.join(args.outdir, 'best.pt'))

    with open(os.path.join(args.outdir, 'training_summary.json'), 'w') as f:
        json.dump({'best_val_loss': best_val}, f, indent=2)

if __name__ == "__main__":
    main()
