"""Fast GLaDOS TTS on ONNX Runtime.

Uses the models exported by tools/export_onnx.py. With the default 'hq' vocoder
it sounds the same as glados.py and is 1.4-2x faster on CPU; it also runs on any ONNX
Runtime provider (CPU, CUDA, CoreML, DirectML…). The 'lq' vocoder is 3-5x faster
again but audibly worse (a fluttery, wet tone). 'nsf' is the distilled student
from tools/distill (see README), once you have trained and exported one.

    python tools/export_onnx.py            # once
    python glados_onnx.py "The cake is a lie."
"""
import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from scipy.io.wavfile import write

from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer

SAMPLE_RATE = 22050


class GladosONNX:
    def __init__(self, model_dir='onnx', vocoder='hq', voice='p2', threads=None, providers=None):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads:
            opts.intra_op_num_threads = threads
        providers = providers or ['CPUExecutionProvider']
        voc_file = {'lq': 'vocoder-lq.onnx', 'hq': 'vocoder.onnx', 'nsf': 'vocoder-nsf.onnx'}[vocoder]
        self.tacotron = ort.InferenceSession(os.path.join(model_dir, 'glados.onnx'), opts, providers=providers)
        self.vocoder = ort.InferenceSession(os.path.join(model_dir, voc_file), opts, providers=providers)
        self.speaker = np.fromfile(os.path.join(model_dir, f'speaker_{voice}.bin'), dtype=np.float32).reshape(1, 256)
        self.cleaner = Cleaner('english_cleaners', True, 'en-us')
        self.tokenizer = Tokenizer()
        self.synthesize('Hello.')  # warm-up

    def tokens(self, text):
        if text[-1] not in '.?!':
            text += '.'
        return np.array([self.tokenizer(self.cleaner(text))], dtype=np.int64)

    def synthesize(self, text, alpha=1.0, log=False):
        """Returns float32 audio in [-1, 1] at 22050 Hz."""
        t0 = time.perf_counter()
        x = self.tokens(text)
        t1 = time.perf_counter()
        mel = self.tacotron.run(None, {'tokens': x, 'speaker': self.speaker,
                                       'alpha': np.array([alpha], dtype=np.float32)})[0]
        t2 = time.perf_counter()
        audio = self.vocoder.run(None, {'mel': mel})[0].squeeze()
        t3 = time.perf_counter()
        if log:
            dur = len(audio) / SAMPLE_RATE
            print(f'front-end {1e3 * (t1 - t0):.0f} ms | tacotron {1e3 * (t2 - t1):.0f} ms | '
                  f'vocoder {1e3 * (t3 - t2):.0f} ms | {dur:.2f} s audio | RTF {(t3 - t0) / dur:.3f}')
        return audio

    @staticmethod
    def to_int16(audio):
        return (np.clip(audio, -1, 1) * 32767).astype(np.int16)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('text', nargs='?', default='Hello, and again, welcome to the Aperture Science computer-aided enrichment center.')
    ap.add_argument('-o', '--output', default='output.wav')
    ap.add_argument('--vocoder', choices=['hq', 'lq', 'nsf'], default='hq')
    ap.add_argument('--voice', choices=['p1', 'p2'], default='p2')
    ap.add_argument('--speed', type=float, default=1.0, help='>1 is faster speech')
    ap.add_argument('--threads', type=int, default=None)
    args = ap.parse_args()
    tts = GladosONNX(vocoder=args.vocoder, voice=args.voice, threads=args.threads)
    audio = tts.synthesize(args.text, alpha=args.speed, log=True)
    write(args.output, SAMPLE_RATE, tts.to_int16(audio))
    print(f'wrote {args.output}')
