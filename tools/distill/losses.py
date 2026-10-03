"""Discriminators and losses (HiFi-GAN MPD + UnivNet multi-resolution spectrogram
discriminator, full-band multi-resolution mel loss)."""
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

from tools.distill.nsf_vocoder import LRELU, SR


class PeriodD(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.p = period
        chs = [1, 32, 128, 512, 1024]
        self.convs = nn.ModuleList([weight_norm(nn.Conv2d(chs[i], chs[i + 1], (5, 1), (3, 1), padding=(2, 0))) for i in range(4)]
                                   + [weight_norm(nn.Conv2d(1024, 1024, (5, 1), 1, padding=(2, 0)))])
        self.post = weight_norm(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        b, c, t = x.shape
        if t % self.p:
            x = F.pad(x, (0, self.p - t % self.p), 'reflect')
        x = x.view(b, c, -1, self.p)
        fmap = []
        for c_ in self.convs:
            x = F.leaky_relu(c_(x), LRELU)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class SpecD(nn.Module):
    """Discriminates log-magnitude spectrograms; good at catching high-band artifacts."""
    def __init__(self, n_fft, hop, win):
        super().__init__()
        self.n_fft, self.hop, self.win = n_fft, hop, win
        self.register_buffer('window', torch.hann_window(win), persistent=False)
        self.convs = nn.ModuleList([
            weight_norm(nn.Conv2d(1, 32, (3, 9), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 9), (1, 2), padding=(1, 4))),
            weight_norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.post = weight_norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def forward(self, x):
        with torch.autocast(x.device.type, enabled=False):  # stft needs fp32
            S = torch.stft(x.float().squeeze(1), self.n_fft, self.hop, self.win, self.window, return_complex=True)
            x = torch.log(S.abs().clamp(min=1e-5))[:, None].transpose(2, 3)  # (B,1,T,F)
        fmap = []
        for c in self.convs:
            x = F.leaky_relu(c(x), LRELU)
            fmap.append(x)
        x = self.post(x)
        fmap.append(x)
        return x.flatten(1), fmap


class Discriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.ds = nn.ModuleList([PeriodD(p) for p in (2, 3, 5, 7, 11)] +
                                [SpecD(*r) for r in ((1024, 120, 600), (2048, 240, 1200), (512, 50, 240))])

    def forward(self, x):
        outs, fmaps = [], []
        for d in self.ds:
            o, f = d(x)
            outs.append(o)
            fmaps.append(f)
        return outs, fmaps


def d_loss(real, fake):
    return sum(torch.mean((1 - r) ** 2) + torch.mean(f ** 2) for r, f in zip(real, fake))


def g_adv_loss(fake):
    return sum(torch.mean((1 - f) ** 2) for f in fake)


def fm_loss(fr, ff):
    return sum(F.l1_loss(b, a.detach()) for da, db in zip(fr, ff) for a, b in zip(da, db))


class MultiResMelLoss(nn.Module):
    """L1 between log-mels at several resolutions, full band (0 to 11025 Hz),
    so the high frequencies are supervised directly."""
    def __init__(self, res=((512, 128, 64), (1024, 256, 100), (2048, 512, 128))):
        super().__init__()
        self.res = res
        for i, (n_fft, hop, n_mels) in enumerate(res):
            fb = librosa.filters.mel(sr=SR, n_fft=n_fft, n_mels=n_mels, fmin=0, fmax=SR / 2)
            self.register_buffer(f'fb{i}', torch.from_numpy(fb).float(), persistent=False)
            self.register_buffer(f'win{i}', torch.hann_window(n_fft), persistent=False)

    def logmel(self, x, i):
        n_fft, hop, _ = self.res[i]
        with torch.autocast(x.device.type, enabled=False):
            S = torch.stft(x.float().squeeze(1), n_fft, hop, n_fft, getattr(self, f'win{i}'), return_complex=True).abs()
            return torch.log(torch.matmul(getattr(self, f'fb{i}'), S).clamp(min=1e-5))

    def forward(self, y_hat, y):
        return sum(F.l1_loss(self.logmel(y_hat, i), self.logmel(y, i)) for i in range(len(self.res))) / len(self.res)
