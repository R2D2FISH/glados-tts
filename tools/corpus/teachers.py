"""Teacher adapters: anything that turns a sentence into GLaDOS audio.

A teacher is any object with
    sample_rate: int
    synthesize(text: str, phonemes: str) -> np.ndarray   # float32 mono in [-1, 1]
Use `phonemes` when the teacher accepts IPA (that's how the Hawaiian rules and
lexicon.txt reach it); otherwise ignore it and use `text`.

Select one on the command line:
    --teacher glados                  the current ForwardTacotron + HQ HiFi-GAN (baseline / pipeline test)
    --teacher command --cmd '...'     any external program, run once per sentence
    --teacher mymodule:MyTeacher      your own class (e.g. a fine-tuned StyleTTS2 / F5-TTS wrapper)
"""
import importlib
import os
import shlex
import subprocess
import tempfile

import numpy as np
from scipy.io import wavfile


class GladosTeacher:
    """The existing model, via glados_onnx.py (run tools/export_onnx.py first)."""
    sample_rate = 22050

    def __init__(self, vocoder='hq', voice='p2', **_):
        from glados_onnx import GladosONNX
        self.tts = GladosONNX(vocoder=vocoder, voice=voice)

    def synthesize(self, text, phonemes):
        return self.tts.synthesize(text)


class CommandTeacher:
    """Runs a shell command per sentence. Placeholders: {text}, {phonemes}, {out}
    (a .wav path the command must write). Example:
        --cmd 'python infer.py --ckpt glados.pt --text {text} --out {out}'
    """
    def __init__(self, cmd, sample_rate=None, **_):
        self.cmd = cmd
        self.sample_rate = sample_rate  # taken from the first wav if not given

    def synthesize(self, text, phonemes):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'out.wav')
            cmd = self.cmd.format(text=shlex.quote(text), phonemes=shlex.quote(phonemes), out=shlex.quote(out))
            subprocess.run(cmd, shell=True, check=True, capture_output=True)
            sr, a = wavfile.read(out)
        if a.dtype.kind == 'i':
            a = a.astype(np.float32) / np.iinfo(a.dtype).max
        if a.ndim > 1:
            a = a.mean(1)
        self.sample_rate = self.sample_rate or sr
        if sr != self.sample_rate:
            raise ValueError(f'teacher wrote {sr} Hz, expected {self.sample_rate}')
        return a.astype(np.float32)


def load_teacher(name, **kw):
    if name == 'glados':
        return GladosTeacher(**kw)
    if name == 'command':
        return CommandTeacher(**kw)
    module, cls = name.split(':')
    return getattr(importlib.import_module(module), cls)(**kw)
