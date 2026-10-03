"""NSF-HiFiGAN-style vocoder for distilling the GLaDOS HQ HiFi-GAN.

The periodic part of the voice is generated explicitly: a sum of sine harmonics
whose phase is the running integral of f0, so the harmonics cannot drift or
warble. A small HiFi-GAN-like network then only has to shape that excitation
(spectral envelope, breathiness, transients). A tiny conv net predicts f0 and
voicing from the mel, so the exported model is a drop-in mel -> audio vocoder.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import weight_norm

SR = 22050
HOP = 256
LRELU = 0.1

PRESETS = {
    # channels after conv_pre, resblock type, kernels, dilations
    'tiny': dict(ch=64, resblock=2, kernels=[3, 5, 7], dilations=[[1, 2], [2, 6], [3, 12]]),
    'small': dict(ch=128, resblock=2, kernels=[3, 5, 7], dilations=[[1, 2], [2, 6], [3, 12]]),
    'base': dict(ch=192, resblock=1, kernels=[3, 7, 11], dilations=[[1, 3, 5]] * 3),
}
UPSAMPLE = [(8, 16), (8, 16), (4, 8)]  # (rate, kernel): 8*8*4 = 256 = HOP


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.normal_(m.weight, 0.0, 0.01)


class ResBlock1(nn.Module):
    def __init__(self, ch, k, dilations):
        super().__init__()
        pad = lambda d: (k * d - d) // 2  # noqa: E731
        self.c1 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=d, padding=pad(d))) for d in dilations])
        self.c2 = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, padding=pad(1))) for _ in dilations])
        self.apply(init_weights)

    def forward(self, x):
        for c1, c2 in zip(self.c1, self.c2):
            x = x + c2(F.leaky_relu(c1(F.leaky_relu(x, LRELU)), LRELU))
        return x


class ResBlock2(nn.Module):
    def __init__(self, ch, k, dilations):
        super().__init__()
        self.c = nn.ModuleList([weight_norm(nn.Conv1d(ch, ch, k, dilation=d, padding=(k * d - d) // 2)) for d in dilations])
        self.apply(init_weights)

    def forward(self, x):
        for c in self.c:
            x = x + c(F.leaky_relu(x, LRELU))
        return x


class F0Predictor(nn.Module):
    """mel (B,80,T) -> continuous log2(f0) (B,T) and voicing logit (B,T)."""
    def __init__(self, ch=128, layers=3):
        super().__init__()
        convs = [nn.Conv1d(80, ch, 5, padding=2)] + [nn.Conv1d(ch, ch, 5, padding=2) for _ in range(layers - 1)]
        self.convs = nn.ModuleList(convs)
        self.out = nn.Conv1d(ch, 2, 1)
        # start near GLaDOS's mean pitch (~165 Hz) so early training is stable
        nn.init.zeros_(self.out.weight)
        with torch.no_grad():
            self.out.bias.copy_(torch.tensor([math.log2(165.0), 2.0]))

    def forward(self, mel):
        h = mel
        for c in self.convs:
            h = F.leaky_relu(c(h), LRELU)
        o = self.out(h)
        return o[:, 0], o[:, 1]


class SineSource(nn.Module):
    """Harmonic excitation from frame-rate f0 and voicing."""
    def __init__(self, harmonics=8, sine_amp=0.1, noise_std=0.003, noise_table=2 ** 19):
        super().__init__()
        self.K = harmonics
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.merge = nn.Conv1d(harmonics + 1, 1, 1)
        # Fixed noise for export: avoids random ops that onnxruntime-web/WebGPU lacks.
        g = torch.Generator().manual_seed(0)
        self.register_buffer('noise', torch.randn(1, 1, noise_table, generator=g), persistent=False)

    def forward(self, f0, uv, random_phase=False):
        """f0, uv: (B,T) frame rate, f0 in Hz (continuous), uv in {0,1}. -> (B,1,T*HOP)"""
        B, T = f0.shape
        # f0 linearly interpolated within frames; uv held per frame
        f0_s = F.interpolate(f0[:, None], scale_factor=HOP, mode='linear', align_corners=False)  # (B,1,N)
        uv_s = F.interpolate(uv[:, None], scale_factor=HOP, mode='nearest')
        # Phase in cycles, accumulated per frame then within frame, so float32
        # stays precise over long utterances.
        inc = (f0_s / SR).reshape(B, 1, T, HOP)
        within = torch.cumsum(inc, dim=3)
        frame_tot = within[..., -1]
        start = torch.cumsum(frame_tot, dim=2) - frame_tot
        start = start - torch.floor(start)
        cycles = (start[..., None] + within).reshape(B, 1, T * HOP)
        k = torch.arange(1, self.K + 1, device=f0.device, dtype=f0.dtype).view(1, self.K, 1)
        phase = k * cycles
        if random_phase:
            phase = phase + torch.rand(B, self.K, 1, device=f0.device)
        sines = torch.sin(2 * math.pi * phase) * (k * f0_s < SR / 2).to(f0.dtype)
        N = T * HOP
        if self.training:
            noise = torch.randn(B, self.K + 1, N, device=f0.device)
        else:
            reps = N // self.noise.shape[-1] + 1
            base = self.noise.repeat(1, 1, reps)[..., :N]
            noise = torch.cat([torch.roll(base, 997 * i, dims=2) for i in range(self.K + 1)], dim=1)
        namp = uv_s * self.noise_std + (1 - uv_s) * self.sine_amp / 3
        sines = sines * self.sine_amp * uv_s + noise[:, :self.K] * namp
        src = torch.cat([sines, noise[:, self.K:] * self.sine_amp / 3], dim=1)
        return torch.tanh(self.merge(src))


class NSFGenerator(nn.Module):
    def __init__(self, preset='small', harmonics=8):
        super().__init__()
        cfg = PRESETS[preset]
        ch = cfg['ch']
        Res = ResBlock1 if cfg['resblock'] == 1 else ResBlock2
        self.source = SineSource(harmonics)
        self.f0 = F0Predictor()
        self.conv_pre = weight_norm(nn.Conv1d(80, ch, 7, padding=3))
        self.ups, self.noise_convs, self.res = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        rates = [r for r, _ in UPSAMPLE]
        for i, (r, k) in enumerate(UPSAMPLE):
            cin, cout = ch // 2 ** i, ch // 2 ** (i + 1)
            self.ups.append(weight_norm(nn.ConvTranspose1d(cin, cout, k, r, padding=(k - r) // 2)))
            s = math.prod(rates[i + 1:])
            self.noise_convs.append(nn.Conv1d(1, cout, 2 * s, s, padding=s // 2) if s > 1 else nn.Conv1d(1, cout, 1))
            self.res.append(nn.ModuleList([Res(cout, kk, d) for kk, d in zip(cfg['kernels'], cfg['dilations'])]))
        self.conv_post = weight_norm(nn.Conv1d(cout, 1, 7, padding=3))
        self.ups.apply(init_weights)
        self.conv_post.apply(init_weights)

    def predict_f0(self, mel):
        logf0, uv_logit = self.f0(mel)
        f0 = torch.pow(2.0, logf0.clamp(5.0, 10.5))  # 32 Hz .. 1.4 kHz (pow, not exp2: exportable)
        return f0, uv_logit

    def forward(self, mel, f0=None, uv=None, random_phase=False):
        """mel (B,80,T). If f0/uv (B,T) are not given they are predicted from the mel."""
        if f0 is None:
            f0, uv_logit = self.predict_f0(mel)
            uv = (uv_logit > 0).to(mel.dtype)
        src = self.source(f0, uv, random_phase)
        x = self.conv_pre(mel)
        for up, nc, res in zip(self.ups, self.noise_convs, self.res):
            x = up(F.leaky_relu(x, LRELU))
            x = x + nc(src)
            x = sum(r(x) for r in res) / len(res)
        return torch.tanh(self.conv_post(F.leaky_relu(x)))

    def remove_weight_norm(self):
        for m in self.modules():
            if parametrize.is_parametrized(m):
                parametrize.remove_parametrizations(m, 'weight')
        return self


class ExportWrapper(nn.Module):
    """mel (1,80,T) -> audio (1,1,T*256); same interface as the other vocoders."""
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def forward(self, mel):
        return self.gen(mel)
