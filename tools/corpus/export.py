"""Export the kept clips as a ForwardTacotron dataset (and optionally NSF vocoder data).

    python -m tools.corpus.export --raw corpus/raw --out corpus/dataset \\
        --real-metadata glados_real/metadata.csv --real-wavs glados_real/wavs \\
        --nsf-data corpus/nsf_data.pt

Writes <out>/wavs/*.wav (trimmed, loudness-matched, resampled to --sr) and
<out>/metadata.csv in the `ljspeech_multi` format (id|speaker|text) that
utils/text/recipes.py reads. Synthetic clips get speaker `glados_synth`; real
lines (if given) get `glados_real`, so ForwardTacotron can learn coverage from
the synthetic data while you synthesize with the real speaker's embedding.

--nsf-data writes the tools/distill training format (mel, audio, f0) computed
from the audio. The mel settings must match your ForwardTacotron config; the
defaults match the current model (22050 Hz, hop 256, 80 bins, 0-8000 Hz).
"""
import argparse
import csv
import os
import sys

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

sys.path.insert(0, os.getcwd())


def load(path):
    sr, a = wavfile.read(path)
    a = a.astype(np.float32) / (32768.0 if a.dtype == np.int16 else 1.0)
    return (a.mean(1) if a.ndim > 1 else a), sr


def prepare(a, sr, target_sr, target_db=-20.0, pad=0.05):
    if sr != target_sr:
        g = np.gcd(sr, target_sr)
        a = resample_poly(a, target_sr // g, sr // g).astype(np.float32)
    hop = target_sr // 100
    frames = a[:len(a) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    loud = np.nonzero(db > db.max() - 40)[0]
    if len(loud):
        p = int(pad * target_sr)
        a = a[max(0, loud[0] * hop - p):min(len(a), (loud[-1] + 1) * hop + p)]
    active = frames[db > db.max() - 40] if len(loud) else frames
    rms = np.sqrt((active ** 2).mean()) + 1e-9
    a = a * 10 ** (target_db / 20) / rms
    peak = np.abs(a).max()
    if peak > 0.95:
        a *= 0.95 / peak
    return a.astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default='corpus/raw')
    ap.add_argument('--out', default='corpus/dataset')
    ap.add_argument('--sr', type=int, default=22050)
    ap.add_argument('--speaker', default='glados_synth')
    ap.add_argument('--real-metadata', help='LJSpeech-style metadata of the real GLaDOS lines (id|text or id|speaker|text)')
    ap.add_argument('--real-wavs', help='folder with the real wavs (<id>.wav)')
    ap.add_argument('--nsf-data', help='also write vocoder training data (tools/distill format) here')
    ap.add_argument('--fmax', type=float, default=8000)
    ap.add_argument('--hop', type=int, default=256)
    ap.add_argument('--n-fft', type=int, default=1024)
    ap.add_argument('--workers', type=int, default=os.cpu_count())
    args = ap.parse_args()

    scores = list(csv.DictReader(open(os.path.join(args.raw, 'scores.tsv'), encoding='utf-8'),
                                 delimiter='\t', quoting=csv.QUOTE_NONE))
    items = [(os.path.join(args.raw, s['file']), s['id'], args.speaker, s['text']) for s in scores if s['keep'] == '1']
    if args.real_metadata:
        for line in open(args.real_metadata, encoding='utf-8'):
            parts = line.rstrip('\n').split('|')
            if len(parts) >= 2:
                items.append((os.path.join(args.real_wavs, parts[0] + '.wav'), 'real_' + parts[0], 'glados_real', parts[-1]))

    os.makedirs(os.path.join(args.out, 'wavs'), exist_ok=True)
    nsf = []
    with open(os.path.join(args.out, 'metadata.csv'), 'w', encoding='utf-8') as meta:
        for path, fid, spk, text in items:
            a = prepare(*load(path), args.sr)
            wavfile.write(os.path.join(args.out, 'wavs', fid + '.wav'), args.sr, (a * 32767).astype(np.int16))
            meta.write(f'{fid}|{spk}|{text.replace("|", " ")}\n')
            if args.nsf_data:
                nsf.append((fid, spk, text, a))
    n_syn = sum(1 for i in items if i[2] == args.speaker)
    print(f'wrote {args.out}: {n_syn} synthetic + {len(items) - n_syn} real clips')

    if args.nsf_data:
        import librosa
        import torch
        from concurrent.futures import ProcessPoolExecutor
        from tools.distill.make_dataset import measure_f0
        if args.sr != 22050 or args.hop != 256:
            sys.exit('the NSF vocoder is built for 22050 Hz / hop 256; adjust tools/distill/nsf_vocoder.py first')
        fb = librosa.filters.mel(sr=args.sr, n_fft=args.n_fft, n_mels=80, fmin=0, fmax=args.fmax, norm='slaney')
        out = []
        for fid, spk, text, a in nsf:
            T = len(a) // args.hop
            S = np.abs(librosa.stft(a, n_fft=args.n_fft, hop_length=args.hop, win_length=args.n_fft))[:, :T]
            mel = np.log(np.clip(fb @ S, 1e-5, None))
            out.append(dict(text=text, voice=spk, mel=torch.from_numpy(mel).half(),
                            audio=torch.from_numpy(a[:T * args.hop]).half()))
        with ProcessPoolExecutor(args.workers) as ex:
            for it, (f0, vuv) in zip(out, ex.map(measure_f0, [it['audio'].float().numpy() for it in out], chunksize=16)):
                it['f0'] = torch.from_numpy(f0)
                it['voiced'] = torch.from_numpy(vuv)
        torch.save(out, args.nsf_data)
        print(f'wrote {args.nsf_data}: {len(out)} clips for tools/distill/train.py')


if __name__ == '__main__':
    main()
