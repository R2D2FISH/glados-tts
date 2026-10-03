"""Train the NSF student vocoder to imitate the HQ HiFi-GAN.

    python -m tools.distill.train --data distill/data.pt --out distill/run1 --preset small

The whole dataset is moved to the GPU and batches are cropped there, so the
CPU only launches kernels (no DataLoader workers needed on older machines).
Every --eval-every steps it writes samples (using predicted f0, i.e. exactly
the inference path) to <out>/samples/ and a checkpoint to <out>/latest.pt.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from scipy.io.wavfile import write

sys.path.insert(0, os.getcwd())
from tools.distill.losses import Discriminators, MultiResMelLoss, d_loss, fm_loss, g_adv_loss  # noqa: E402
from tools.distill.nsf_vocoder import HOP, SR, NSFGenerator  # noqa: E402


class GPUData:
    def __init__(self, items, device):
        self.T = torch.tensor([it['mel'].shape[1] for it in items])
        self.off = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(self.T, 0)[:-1]])
        self.mel = torch.cat([it['mel'] for it in items], 1).to(device)                    # (80, sumT) fp16
        self.audio = torch.cat([it['audio'][:it['mel'].shape[1] * HOP] for it in items]).to(device)  # fp16
        self.f0 = torch.cat([it['f0'] for it in items]).float().to(device)
        self.voiced = torch.cat([it['voiced'] for it in items]).float().to(device)
        self.device = device

    def batch(self, B, seg):
        ok = torch.nonzero(self.T > seg + 1).squeeze(1)
        w = (self.T[ok] - seg).float()
        u = ok[torch.multinomial(w, B, replacement=True)]
        start = self.off[u] + (torch.rand(B) * (self.T[u] - seg)).long()
        fi = (start[:, None] + torch.arange(seg)).to(self.device)
        ai = (start[:, None] * HOP + torch.arange(seg * HOP)).to(self.device)
        mel = self.mel[:, fi].permute(1, 0, 2).float()     # (B,80,seg)
        return mel, self.audio[ai].float()[:, None], self.f0[fi], self.voiced[fi]


def evaluate(G, val, mel_loss, device, out_dir, step, n_save=3):
    G.eval()
    losses = []
    with torch.no_grad():
        for i, it in enumerate(val):
            mel = it['mel'].float()[None].to(device)
            y = it['audio'].float()[None, None, :mel.shape[2] * HOP].to(device)
            y_hat = G(mel)  # predicted f0: the real inference path
            losses.append(mel_loss(y_hat, y).item())
            if i < n_save:
                os.makedirs(f'{out_dir}/samples', exist_ok=True)
                a = (y_hat.squeeze().clamp(-1, 1).cpu().numpy() * 32767).astype(np.int16)
                write(f'{out_dir}/samples/{step:07d}_{i}.wav', SR, a)
                if step == 0:
                    write(f'{out_dir}/samples/teacher_{i}.wav', SR, (y.squeeze().cpu().numpy() * 32767).astype(np.int16))
    G.train()
    return float(np.mean(losses))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', default='distill/data.pt')
    ap.add_argument('--out', default='distill/run')
    ap.add_argument('--preset', default='small', choices=['tiny', 'small', 'base'])
    ap.add_argument('--steps', type=int, default=300000)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--seg-frames', type=int, default=32, help='training crop length in mel frames (x256 samples)')
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--adv-start', type=int, default=5000, help='train on mel loss only until this step')
    ap.add_argument('--pred-f0-start', type=float, default=0.7,
                    help='after this fraction of training, half the batches use predicted f0 so the '
                         'generator adapts to the predictor')
    ap.add_argument('--mel-weight', type=float, default=45.0)
    ap.add_argument('--amp', action='store_true', help='bf16 autocast (faster on Ampere, e.g. a 3090)')
    ap.add_argument('--val', type=int, default=8, help='held-out utterances')
    ap.add_argument('--eval-every', type=int, default=5000)
    ap.add_argument('--log-every', type=int, default=100)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    dev = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    items = torch.load(args.data, weights_only=False)
    val, train = items[-args.val:], items[:-args.val]
    data = GPUData(train, dev)
    del items, train
    print(f'train {len(data.T)} utts ({data.audio.numel() / SR / 3600:.2f} h) on {dev}, val {len(val)}')

    G = NSFGenerator(args.preset).to(dev)
    D = Discriminators().to(dev)
    mel_loss = MultiResMelLoss().to(dev)
    opt_g = torch.optim.AdamW(G.parameters(), args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    opt_d = torch.optim.AdamW(D.parameters(), args.lr, betas=(0.8, 0.99), weight_decay=0.0)
    gamma = 0.25 ** (1 / args.steps)  # lr decays to 1/4 over the run
    sch_g = torch.optim.lr_scheduler.ExponentialLR(opt_g, gamma)
    sch_d = torch.optim.lr_scheduler.ExponentialLR(opt_d, gamma)

    step = 0
    ckpt = f'{args.out}/latest.pt'
    if os.path.exists(ckpt):
        s = torch.load(ckpt, map_location=dev, weights_only=False)
        G.load_state_dict(s['G']); D.load_state_dict(s['D'])
        opt_g.load_state_dict(s['opt_g']); opt_d.load_state_dict(s['opt_d'])
        sch_g.load_state_dict(s['sch_g']); sch_d.load_state_dict(s['sch_d'])
        step = s['step']
        print(f'resumed from step {step}')
    json.dump(vars(args), open(f'{args.out}/args.json', 'w'), indent=1)

    amp = torch.autocast(dev.type, dtype=torch.bfloat16, enabled=args.amp)
    log, t0 = {}, time.time()
    if step == 0:
        print(f'step 0  val mel {evaluate(G, val, mel_loss, dev, args.out, 0):.4f}')
    while step < args.steps:
        mel, y, f0, voiced = data.batch(args.batch, args.seg_frames)
        with amp:
            logf0, uv_logit = G.f0(mel)
        f0_loss = F.l1_loss(logf0.float(), torch.log2(f0)) + F.binary_cross_entropy_with_logits(uv_logit.float(), voiced)

        f0_in = f0 * torch.exp2(torch.randn_like(f0) * 0.01)  # tiny jitter: robustness to f0 error
        uv_in = voiced
        if step >= args.pred_f0_start * args.steps and step % 2:
            f0_in = torch.exp2(logf0.detach().float().clamp(5.0, 10.5))
            uv_in = (uv_logit.detach() > 0).float()
        with amp:
            y_hat = G(mel, f0_in, uv_in, random_phase=True).float()

        adv = step >= args.adv_start
        if adv:
            with amp:
                real, _ = D(y)
                fake, _ = D(y_hat.detach())
            loss_d = d_loss([r.float() for r in real], [f.float() for f in fake])
            opt_d.zero_grad(set_to_none=True)
            loss_d.backward()
            opt_d.step()
            log['d'] = log.get('d', 0) + loss_d.item()

        lm = mel_loss(y_hat, y)
        loss_g = args.mel_weight * lm + f0_loss
        if adv:
            with amp:
                real, fr = D(y)
                fake, ff = D(y_hat)
            ga = g_adv_loss([f.float() for f in fake])
            fm = fm_loss([[a.float() for a in d] for d in fr], [[a.float() for a in d] for d in ff])
            loss_g = loss_g + ga + 2 * fm
            log['adv'] = log.get('adv', 0) + ga.item()
            log['fm'] = log.get('fm', 0) + fm.item()
        opt_g.zero_grad(set_to_none=True)
        loss_g.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), 1000.0)
        opt_g.step()
        sch_g.step()
        if adv:
            sch_d.step()
        log['mel'] = log.get('mel', 0) + lm.item()
        log['f0'] = log.get('f0', 0) + f0_loss.item()
        step += 1

        if step % args.log_every == 0:
            dt = time.time() - t0
            msg = '  '.join(f'{k} {v / args.log_every:.4f}' for k, v in log.items())
            eta = (args.steps - step) * dt / args.log_every / 3600
            print(f'step {step:7d}  {msg}  {args.log_every / dt:.1f} it/s  eta {eta:.1f} h', flush=True)
            log, t0 = {}, time.time()
        if step % args.eval_every == 0 or step == args.steps:
            v = evaluate(G, val, mel_loss, dev, args.out, step)
            torch.save(dict(G=G.state_dict(), D=D.state_dict(), opt_g=opt_g.state_dict(), opt_d=opt_d.state_dict(),
                            sch_g=sch_g.state_dict(), sch_d=sch_d.state_dict(), step=step, preset=args.preset),
                       ckpt + '.tmp')
            os.replace(ckpt + '.tmp', ckpt)
            torch.save(dict(G=G.state_dict(), preset=args.preset, step=step), f'{args.out}/G_{step:07d}.pt')
            print(f'step {step}  val mel {v:.4f}  (checkpoint saved)', flush=True)
            t0 = time.time()


if __name__ == '__main__':
    main()
