"""Eager re-implementation of the multi-speaker ForwardTacotron in models/glados-new.pt.

The shipped model is TorchScript, which can't be exported to ONNX directly
(isinstance checks, packed-sequence branches, a Python loop in the length
regulator). This module mirrors its structure and parameter names so the
TorchScript state_dict loads with strict=True, and replaces the length
regulator with a loop-free gather so the whole graph traces cleanly.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchNormConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel, relu=True):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=1, padding=kernel // 2, bias=False)
        self.bnorm = nn.BatchNorm1d(out_ch)
        self.relu = relu

    def forward(self, x):
        x = self.conv(x)
        if self.relu:
            x = F.relu(x)
        return self.bnorm(x)


class HighwayNetwork(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.W1 = nn.Linear(size, size)
        self.W2 = nn.Linear(size, size)

    def forward(self, x):
        g = torch.sigmoid(self.W2(x))
        return g * F.relu(self.W1(x)) + (1. - g) * x


class CBHG(nn.Module):
    def __init__(self, K, in_channels, channels, proj_channels, num_highways):
        super().__init__()
        self.conv1d_bank = nn.ModuleList(
            [BatchNormConv(in_channels, channels, k) for k in range(1, K + 1)])
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        self.conv_project1 = BatchNormConv(K * channels, proj_channels[0], 3)
        self.conv_project2 = BatchNormConv(proj_channels[0], proj_channels[1], 3, relu=False)
        self.pre_highway = nn.Linear(proj_channels[-1], channels, bias=False)
        self.highways = nn.ModuleList([HighwayNetwork(channels) for _ in range(num_highways)])
        self.rnn = nn.GRU(channels, channels, batch_first=True, bidirectional=True)

    def forward(self, x):
        residual = x
        seq_len = x.size(-1)
        x = torch.cat([conv(x)[:, :, :seq_len] for conv in self.conv1d_bank], dim=1)
        x = self.maxpool(x)[:, :, :seq_len]
        x = self.conv_project1(x)
        x = self.conv_project2(x)
        x = (x + residual).transpose(1, 2)
        x = self.pre_highway(x)
        for h in self.highways:
            x = h(x)
        x, _ = self.rnn(x)
        return x


def _repeat_speaker(semb, n):
    return semb[:, None, :].expand(-1, n, -1)


class SeriesPredictor(nn.Module):
    def __init__(self, num_chars, emb_dim, semb_dim, conv_dims, rnn_dims, out_dims, cond=False):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, emb_dim)
        in_dims = emb_dim + semb_dim
        if cond:
            self.pitch_cond_embedding = nn.Embedding(4, 4)
            in_dims += 4
        self.convs = nn.ModuleList([
            BatchNormConv(in_dims, conv_dims, 5),
            BatchNormConv(conv_dims, conv_dims, 5),
            BatchNormConv(conv_dims, conv_dims, 5),
        ])
        self.rnn = nn.GRU(conv_dims, rnn_dims, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(2 * rnn_dims, out_dims)

    def forward(self, x, semb, x_cond=None):
        e = self.embedding(x)
        parts = [e]
        if x_cond is not None:
            parts.append(self.pitch_cond_embedding(x_cond))
        parts.append(_repeat_speaker(semb, e.size(1)))
        h = torch.cat(parts, dim=2).transpose(1, 2)
        for conv in self.convs:
            h = conv(h)
        h, _ = self.rnn(h.transpose(1, 2))
        return self.lin(h)


class ForwardTacotron(nn.Module):
    def __init__(self, num_chars=135):
        super().__init__()
        self.embedding = nn.Embedding(num_chars, 256)
        self.dur_pred = SeriesPredictor(num_chars, 128, 256, 256, 128, 1, cond=True)
        self.pitch_cond_pred = SeriesPredictor(num_chars, 128, 256, 256, 128, 3)
        self.pitch_pred = SeriesPredictor(num_chars, 128, 256, 256, 256, 1, cond=True)
        self.energy_pred = SeriesPredictor(num_chars, 128, 256, 256, 64, 1)
        self.prenet = CBHG(16, 256, 256, [256, 256], 4)
        self.lstm = nn.LSTM(768, 512, batch_first=True, bidirectional=True)
        self.lin = nn.Linear(1024, 80)
        self.postnet = CBHG(8, 80, 256, [256, 80], 4)
        self.post_proj = nn.Linear(512, 80, bias=False)
        self.pitch_proj = nn.Conv1d(1, 768, kernel_size=3, padding=1)
        self.energy_proj = nn.Conv1d(1, 768, kernel_size=3, padding=1)
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.pitch_strength = 1.0
        self.energy_strength = 1.0

    @staticmethod
    def length_regulate(x, dur):
        # Equivalent to repeat_interleave(x[0], round(dur)) but loop-free:
        # each output frame t takes the token i where ends[i-1] <= t < ends[i].
        reps = (dur.clamp(min=0) + 0.5).long()
        ends = torch.cumsum(reps[0], 0)
        frames = torch.arange(ends[-1], device=x.device)
        idx = (frames[:, None] >= ends[None, :]).sum(1)
        return x[:, idx, :]

    def forward(self, x, semb, alpha):
        """x: (1, N) int64 phoneme ids, semb: (1, 256), alpha: (1,) speed factor.
        Returns the post-net mel spectrogram, (1, 80, T)."""
        pitch_cond = self.pitch_cond_pred(x, semb).argmax(-1)
        dur = self.dur_pred(x, semb, pitch_cond).squeeze(-1) / alpha
        # The original fills all durations with 2 when the total is <= 0.
        dur = torch.where(dur.long().sum() <= 0, torch.full_like(dur, 2.), dur)
        pitch = self.pitch_pred(x, semb, pitch_cond).transpose(1, 2)
        energy = self.energy_pred(x, semb).transpose(1, 2)

        h = self.prenet(self.embedding(x).transpose(1, 2))
        h = torch.cat([h, _repeat_speaker(semb, h.size(1))], dim=2)
        h = h + self.pitch_proj(pitch).transpose(1, 2) * self.pitch_strength
        h = h + self.energy_proj(energy).transpose(1, 2) * self.energy_strength
        h = self.length_regulate(h, dur)
        h, _ = self.lstm(h)
        mel = self.lin(h).transpose(1, 2)
        post = self.post_proj(self.postnet(mel))
        return post.transpose(1, 2)


def load_from_torchscript(path='models/glados-new.pt'):
    ts = torch.jit.load(path, map_location='cpu')
    model = ForwardTacotron()
    model.load_state_dict(ts.state_dict(), strict=True)
    model.pitch_strength = float(ts.pitch_strength)
    model.energy_strength = float(ts.energy_strength)
    return model.eval(), ts
