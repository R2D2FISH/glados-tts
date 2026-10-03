"""Build a distillation dataset: text -> ForwardTacotron mel -> HQ HiFi-GAN audio,
plus frame-level f0/voicing measured from that audio (WORLD DIO + StoneMask).

Run from the repo root, e.g.
    python -m tools.distill.make_dataset --num 30000 --out distill/data.pt
    python -m tools.distill.make_dataset --text my_sentences.txt --out distill/data.pt

The student learns to reproduce the teacher's audio from the teacher's own mels,
so no recordings are needed, and training sees exactly the mels it will get at
inference time.
"""
import argparse
import os
import random
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from tools.distill.nsf_vocoder import HOP, SR  # noqa: E402

FALLBACK = """Hello, and again, welcome to the Aperture Science computer-aided enrichment center.
We hope your brief detention in the relaxation vault has been a pleasant one.
Oh. It's you. It's been a long time. How have you been?
The Enrichment Center reminds you that the Weighted Companion Cube will never threaten to stab you.
I'm making a note here: huge success. It's hard to overstate my satisfaction.
The cake is a lie. Please proceed to the next test chamber.
This next test involves turrets. You remember them, right?
Science isn't about why. It's about why not.
The weather forecast for today is cloudy with a chance of neurotoxin.
Thank you for helping us help you help us all.""".split('\n')


def load_sentences(path, num, seed):
    sents = []
    if path:
        text = open(path, encoding='utf-8', errors='ignore').read()
        sents = re.findall(r'[^.!?\n]+[.!?]?', text)
    else:
        try:
            import nltk
            for c in ('gutenberg', 'brown', 'webtext'):
                nltk.download(c, quiet=True)
            from nltk.corpus import brown, gutenberg, webtext
            for corpus in (gutenberg, brown, webtext):
                sents += [' '.join(s) for s in corpus.sents()]
        except Exception as e:  # offline: still produce something usable for a smoke test
            print(f'nltk corpora unavailable ({type(e).__name__}); using built-in lines', file=sys.stderr)
    clean = []
    for s in sents:
        s = re.sub(r'\s+([,.;:!?\'])', r'\1', s)   # detokenize "word ," -> "word,"
        s = re.sub(r'\s+', ' ', s).strip(' "\'`-')
        letters = sum(c.isalpha() for c in s)
        if 15 <= len(s) <= 220 and letters > 0.7 * len(s) and not re.search(r'[\[\]{}<>|_*#@~^]', s):
            clean.append(s)
    clean = sorted(set(clean))
    random.Random(seed).shuffle(clean)
    clean = clean[:num]
    while len(clean) < min(num, 64):  # tiny/offline runs
        clean += FALLBACK
    return clean[:num]


def measure_f0(audio):
    """audio float32 (N,) -> f0 Hz continuous (T,), voiced (T,) bool."""
    import pyworld as pw
    from scipy.ndimage import median_filter
    a = audio.astype(np.float64)
    T = len(a) // HOP
    f0, t = pw.dio(a, SR, f0_floor=60, f0_ceil=800, frame_period=HOP / SR * 1000)
    f0 = pw.stonemask(a, f0, t, SR)
    f = np.zeros(T)
    f[:min(T, len(f0))] = f0[:T]
    voiced = median_filter((f > 0).astype(np.uint8), 3).astype(bool) & (f > 0)
    if voiced.sum() == 0:
        return np.full(T, 165.0, np.float32), voiced
    idx = np.arange(T)
    logf = np.interp(idx, idx[voiced], np.log2(f[voiced]))  # fill gaps so f0 is continuous
    return np.exp2(logf).astype(np.float32), voiced


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text', help='text file to draw sentences from (default: nltk gutenberg+brown+webtext)')
    ap.add_argument('--num', type=int, default=30000)
    ap.add_argument('--out', default='distill/data.pt')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    ap.add_argument('--speed-range', type=float, nargs=2, default=(0.85, 1.2))
    ap.add_argument('--pitch-jitter', type=float, default=0.3, help='std of normalized pitch shift (1.0 ~ 30 Hz)')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from tools.forward_tacotron import load_from_torchscript
    from utils.tools import prepare_text

    rng = random.Random(args.seed)
    sents = load_sentences(args.text, args.num, args.seed)
    print(f'{len(sents)} sentences; teacher on {args.device}')
    taco, _ = load_from_torchscript()
    taco = taco.to(args.device)
    voc = torch.jit.load('models/vocoder-gpu.pt', map_location=args.device).eval()
    embs = {v: torch.load(f'models/emb/glados_{v}.pt').to(args.device) for v in ('p1', 'p2')}

    items, t0 = [], time.time()
    with torch.no_grad():
        for i, s in enumerate(sents):
            voice = rng.choice(['p1', 'p2'])
            alpha = rng.uniform(*args.speed_range)
            taco.pitch_shift = max(-1.0, min(1.0, rng.gauss(0, args.pitch_jitter)))
            x = prepare_text(s).to(args.device)
            mel = taco(x, embs[voice], torch.tensor([alpha], device=args.device))
            if mel.shape[2] < 40:
                continue
            audio = voc(mel).squeeze()
            items.append(dict(text=s, voice=voice, alpha=alpha, pitch_shift=taco.pitch_shift,
                              mel=mel[0].half().cpu(), audio=audio.half().cpu()))
            if (i + 1) % 500 == 0:
                hrs = sum(it['audio'].numel() for it in items) / SR / 3600
                print(f'  {i + 1}/{len(sents)}  {hrs:.1f} h audio  {time.time() - t0:.0f} s')

    print(f'measuring f0 on {args.workers} processes...')
    with ProcessPoolExecutor(args.workers) as ex:
        for it, (f0, vuv) in zip(items, ex.map(measure_f0, [it['audio'].float().numpy() for it in items], chunksize=16)):
            it['f0'] = torch.from_numpy(f0)
            it['voiced'] = torch.from_numpy(vuv)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save(items, args.out)
    hrs = sum(it['audio'].numel() for it in items) / SR / 3600
    print(f'wrote {args.out}: {len(items)} utterances, {hrs:.2f} h, {time.time() - t0:.0f} s total')


if __name__ == '__main__':
    main()
