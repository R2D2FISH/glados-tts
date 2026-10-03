"""Recover models/vocoder-cpu-lq.pt as a plain PyTorch module.

That file was saved with torch.utils.mobile_optimizer, which bakes conv weights
into XNNPACK `Conv2dOpContext` objects. Current PyTorch builds can no longer
load those, so we unpickle constants.pkl ourselves, pull out the raw weight
tensors, and rebuild the generator (HiFi-GAN, ResBlock2, upsample 8x8x4).
"""
import pickle
import zipfile

import torch
import torch.nn as nn
import torch.nn.functional as F

LRELU = 0.1


class _Ctx:
    """Stand-in for xnnpack.Conv2dOpContext; keeps its pickled state."""
    def __setstate__(self, state):
        self.state = state


def _read_constants(path):
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split('/')[0]
    dtypes = {'FloatStorage': torch.float32, 'LongStorage': torch.int64, 'DoubleStorage': torch.float64}

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.endswith('xnnpack') and name == 'Conv2dOpContext':
                return _Ctx
            if mod == 'torch' and name in dtypes:
                return dtypes[name]
            if mod == 'torch._utils' and name == '_rebuild_tensor_v2':
                return lambda flat, offset, size, stride, *_: flat.as_strided(size, stride, offset).clone()
            if mod == 'torch.jit._pickle':
                return lambda x: x
            if mod == 'collections' and name == 'OrderedDict':
                from collections import OrderedDict
                return OrderedDict
            raise pickle.UnpicklingError(f'unexpected global {mod}.{name}')

        def persistent_load(self, pid):
            _, dtype, key, _, numel = pid
            raw = bytearray(z.read(f'{root}/constants/{key}'))
            return torch.frombuffer(raw, dtype=dtype, count=numel)

    consts = U(z.open(f'{root}/constants.pkl')).load()
    out = []
    for c in consts:
        if isinstance(c, _Ctx):
            # state: (weight[O,I,1,K], bias, stride, padding, dilation, groups, min, max)
            w, b, _, pad, dil = c.state[:5]
            out.append(('conv', w.squeeze(2), b, pad[1], dil[1]))
        else:
            out.append(c)
    return out


class Conv(nn.Conv1d):
    def __init__(self, w, b, pad, dil):
        super().__init__(w.shape[1], w.shape[0], w.shape[2], padding=pad, dilation=dil)
        self.weight.data.copy_(w)
        self.bias.data.copy_(b)


class GeneratorLQ(nn.Module):
    def __init__(self, consts):
        super().__init__()
        it = iter(consts)
        nxt = lambda: next(it)  # noqa: E731

        def conv():
            _, w, b, pad, dil = nxt()
            return Conv(w, b, pad, dil)

        self.conv_pre = conv()
        self.ups = nn.ModuleList()
        self.firsts = nn.ModuleList()   # [stage][kernel] first conv of each ResBlock2
        self.seconds = nn.ModuleList()  # [stage][kernel] second conv
        strides = [(8, 4), (8, 4), (4, 2)]
        self.strides = strides
        self.num_kernels = 3
        for stage, (s, p) in enumerate(strides):
            w, b = nxt(), nxt()
            up = nn.ConvTranspose1d(w.shape[0], w.shape[1], w.shape[2], stride=s, padding=p)
            up.weight.data.copy_(w)
            up.bias.data.copy_(b)
            self.ups.append(up)
            # The traced code evaluates the three first-convs in reverse order,
            # then interleaves the second-convs in forward order.
            f = [conv() for _ in range(3)][::-1]
            sc = [conv() for _ in range(3)]
            if stage == 0:
                nxt()  # c9: num_kernels constant (3)
            self.firsts.append(nn.ModuleList(f))
            self.seconds.append(nn.ModuleList(sc))
        self.conv_post = conv()

    def forward(self, x):
        x = F.leaky_relu(self.conv_pre(x), LRELU)
        for i, up in enumerate(self.ups):
            x = up(x)
            h = F.leaky_relu(x, LRELU)
            xs = 0
            for c1, c2 in zip(self.firsts[i], self.seconds[i]):
                y = c1(h) + x
                y = c2(F.leaky_relu(y, LRELU)) + y
                xs = xs + y
            x = xs / self.num_kernels
            x = F.leaky_relu(x, LRELU) if i < len(self.ups) - 1 else F.leaky_relu(x)
        return torch.tanh(self.conv_post(x))


def load(path='models/vocoder-cpu-lq.pt'):
    return GeneratorLQ(_read_constants(path)).eval()
