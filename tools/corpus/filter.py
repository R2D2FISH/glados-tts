"""Score teacher clips and keep only the good ones.

    python -m tools.corpus.filter --raw corpus/raw --reference path/to/real_glados_wavs --whisper small.en --utmos

Per clip it measures:
  wer         Whisper transcript vs the text (skipped words, babble, repeats). Hawaiian words
              are excluded: Whisper can't spell them, so they would count as errors.
  spk_sim     cosine similarity of a speaker embedding (Resemblyzer) to the real GLaDOS lines
  f0_shift    median pitch vs the real lines, in semitones
  f0_spread   pitch variability (semitone std) relative to the real lines; GLaDOS's narrow,
              regular pitch is the part implicit-pitch models lose first
  sec_per_ph  speaking rate vs the corpus median (truncation, stalls, rushing)
  max_pause   longest internal silence, clipping
  utmos       predicted naturalness (MOS), optional
Writes <raw>/scores.tsv (every clip, with keep flag and reasons) and prints a summary.
With several takes per sentence, the best passing take is kept.
"""
import argparse
import csv
import glob
import os
import re
import sys

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.getcwd())


def load(path):
    sr, a = wavfile.read(path)
    a = a.astype(np.float32) / (32768.0 if a.dtype == np.int16 else 1.0)
    return (a.mean(1) if a.ndim > 1 else a), sr


def pitch_stats(a, sr):
    import pyworld as pw
    x = a.astype(np.float64)
    f0, t = pw.dio(x, sr, f0_floor=60, f0_ceil=800, frame_period=10)
    f0 = pw.stonemask(x, f0, t, sr)
    v = f0[f0 > 0]
    if len(v) < 10:
        return np.nan, np.nan, len(v) / max(1, len(f0))
    st = 12 * np.log2(v / 440)
    return float(np.median(st)), float(np.std(st)), len(v) / len(f0)


def pauses(a, sr, thresh_db=-40):
    hop = sr // 100
    frames = a[:len(a) // hop * hop].reshape(-1, hop)
    db = 20 * np.log10(np.sqrt((frames ** 2).mean(1)) + 1e-9)
    loud = db > db.max() + thresh_db
    idx = np.nonzero(loud)[0]
    if len(idx) == 0:
        return 99.0
    inner = loud[idx[0]:idx[-1] + 1]
    longest = run = 0
    for v in inner:
        run = 0 if v else run + 1
        longest = max(longest, run)
    return longest / 100


_norm_re = re.compile(r"[^a-z0-9' ]+")


_okina_re = re.compile(r"(?<=\w)[ʻ‘’'`](?=\w)")


def norm_words(t):
    from utils.text.numbers import normalize_numbers
    from unidecode import unidecode
    t = _okina_re.sub('', t)  # Kalanianaʻole / Kalaniana'ole / Kalanianaole -> one word
    t = normalize_numbers(unidecode(t)).lower().replace('-', ' ')
    return _norm_re.sub(' ', t).split()


def wer_excluding(ref_words, hyp_words, skip):
    """WER where reference words in `skip` can't count as errors."""
    import jiwer
    if not ref_words:
        return 0.0
    out = jiwer.process_words(' '.join(ref_words), ' '.join(hyp_words) or '<empty>')
    errors, counted = 0, 0
    for c in out.alignments[0]:
        refs = ref_words[c.ref_start_idx:c.ref_end_idx]
        if c.type == 'insert':
            # ASR often splits a Hawaiian name into several words; forgive insertions next to one
            near = ref_words[max(0, c.ref_start_idx - 1):c.ref_start_idx + 1]
            if not any(w in skip for w in near):
                errors += c.hyp_end_idx - c.hyp_start_idx
            continue
        for w in refs:
            if w in skip:
                continue
            counted += 1
            errors += c.type != 'equal'
    return errors / max(1, counted)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--raw', default='corpus/raw')
    ap.add_argument('--reference', required=True, help='folder of real GLaDOS wavs (speaker + pitch reference)')
    ap.add_argument('--whisper', default=None, help='faster-whisper model, e.g. small.en (omit to skip)')
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--utmos', action='store_true', help='score naturalness with UTMOS (downloads via torch.hub)')
    ap.add_argument('--max-wer', type=float, default=0.1)
    ap.add_argument('--min-spk-sim', type=float, default=None,
                    help='absolute threshold; default: drop outliers (median - 3 robust std) of this teacher\'s clips')
    ap.add_argument('--max-f0-shift', type=float, default=2.0, help='semitones')
    ap.add_argument('--f0-spread', type=float, nargs=2, default=(0.5, 1.8), help='allowed ratio to the real lines')
    ap.add_argument('--rate-range', type=float, nargs=2, default=(0.7, 1.4), help='allowed ratio to median s/phoneme')
    ap.add_argument('--max-pause', type=float, default=0.8, help='seconds')
    ap.add_argument('--min-utmos', type=float, default=3.0)
    args = ap.parse_args()

    from resemblyzer import VoiceEncoder, preprocess_wav
    from utils.text.cleaners import Cleaner
    from utils.text.hawaiian import is_hawaiian

    rows = list(csv.DictReader(open(os.path.join(args.raw, 'manifest.tsv'), encoding='utf-8'),
                               delimiter='\t', quoting=csv.QUOTE_NONE))
    enc = VoiceEncoder(device='cpu')

    # --- reference statistics from the real lines
    refs = sorted(glob.glob(os.path.join(args.reference, '**', '*.wav'), recursive=True))
    if not refs:
        sys.exit(f'no wavs under {args.reference}')
    ref_emb = np.stack([enc.embed_utterance(preprocess_wav(p)) for p in refs])
    centroid = ref_emb.mean(0)
    centroid /= np.linalg.norm(centroid)
    ref_sims = ref_emb @ centroid
    ref_pitch = np.array([pitch_stats(*load(p))[:2] for p in refs], dtype=float)
    ref_med, ref_std = np.nanmedian(ref_pitch[:, 0]), np.nanmedian(ref_pitch[:, 1])
    print(f'reference: {len(refs)} clips, speaker sim to centroid median {np.median(ref_sims):.3f}, '
          f'pitch median {440 * 2 ** (ref_med / 12):.0f} Hz, spread {ref_std:.2f} st')

    whisper = None
    if args.whisper:
        from faster_whisper import WhisperModel
        whisper = WhisperModel(args.whisper, device=args.device,
                               compute_type='float16' if args.device == 'cuda' else 'int8')
    utmos = None
    if args.utmos:
        import torch
        utmos = torch.hub.load('tarepan/SpeechMOS:v1.2.0', 'utmos22_strong', trust_repo=True).eval()

    english = Cleaner('english_cleaners', True, 'en-us').english_dict
    results = []
    for i, r in enumerate(rows):
        a, sr = load(os.path.join(args.raw, r['file']))
        m = {'file': r['file'], 'id': r['id'], 'kind': r['kind'], 'text': r['text'], 'seconds': len(a) / sr}
        n_ph = max(1, len(r['phonemes'].replace(' ', '')))
        m['sec_per_ph'] = m['seconds'] / n_ph
        m['clipped'] = float(np.mean(np.abs(a) > 0.999))
        m['max_pause'] = pauses(a, sr)
        m['f0_med'], m['f0_std'], m['voiced'] = pitch_stats(a, sr)
        m['spk_sim'] = float(enc.embed_utterance(preprocess_wav(a, source_sr=sr)) @ centroid)
        if whisper:
            segs, _ = whisper.transcribe(a if sr == 16000 else _resample(a, sr, 16000), language='en', beam_size=5)
            hyp = ' '.join(s.text for s in segs)
            ref_words = norm_words(r['text'])
            skip = {w for w in ref_words if is_hawaiian(w, english)}
            m['asr'] = hyp.strip()
            m['wer'] = wer_excluding(ref_words, norm_words(hyp), skip)
        if utmos is not None:
            import torch
            with torch.no_grad():
                m['utmos'] = float(utmos(torch.from_numpy(_resample(a, sr, 16000))[None], 16000))
        results.append(m)
        if (i + 1) % 200 == 0:
            print(f'  scored {i + 1}/{len(rows)}', flush=True)

    rate_med = np.median([m['sec_per_ph'] for m in results])
    sims = np.array([m['spk_sim'] for m in results])
    sim_med = float(np.median(sims))
    sim_mad = 1.4826 * float(np.median(np.abs(sims - sim_med)))
    min_sim = args.min_spk_sim if args.min_spk_sim is not None else sim_med - 3 * max(sim_mad, 0.01)
    print(f'teacher: speaker sim median {sim_med:.3f} vs real lines {np.median(ref_sims):.3f} '
          f'(the gap is how far the teacher is from the real voice); dropping below {min_sim:.3f}')
    for m in results:
        why = []
        if m.get('wer', 0) > args.max_wer:
            why.append('wer')
        if m['spk_sim'] < min_sim:
            why.append('speaker')
        if np.isnan(m['f0_med']) or abs(m['f0_med'] - ref_med) > args.max_f0_shift:
            why.append('pitch_level')
        elif not (args.f0_spread[0] <= m['f0_std'] / ref_std <= args.f0_spread[1]):
            why.append('pitch_spread')
        if not (args.rate_range[0] <= m['sec_per_ph'] / rate_med <= args.rate_range[1]):
            why.append('rate')
        if m['max_pause'] > args.max_pause:
            why.append('pause')
        if m['clipped'] > 0.001:
            why.append('clipping')
        if 'utmos' in m and m['utmos'] < args.min_utmos:
            why.append('utmos')
        m['reasons'] = ','.join(why)
        # ranking among takes of one sentence: closer voice + fewer word errors first
        m['rank'] = m['spk_sim'] - m.get('wer', 0) + 0.1 * m.get('utmos', 0)
    best = {}
    for m in results:
        if not m['reasons'] and (m['id'] not in best or m['rank'] > best[m['id']]['rank']):
            best[m['id']] = m
    for m in results:
        m['keep'] = int(best.get(m['id']) is m)

    cols = ['file', 'id', 'kind', 'keep', 'reasons', 'seconds', 'wer', 'spk_sim', 'f0_med', 'f0_std', 'voiced',
            'sec_per_ph', 'max_pause', 'clipped', 'utmos', 'text', 'asr']
    with open(os.path.join(args.raw, 'scores.tsv'), 'w', encoding='utf-8') as f:
        f.write('\t'.join(cols) + '\n')
        for m in results:
            f.write('\t'.join(f'{m[c]:.4f}' if isinstance(m.get(c), float) else str(m.get(c, '')) for c in cols) + '\n')

    kept = [m for m in results if m['keep']]
    sentences = len({m['id'] for m in results})
    print(f'kept {len(kept)} of {sentences} sentences ({sum(m["seconds"] for m in kept) / 3600:.2f} h) '
          f'from {len(results)} clips')
    from collections import Counter
    c = Counter(w for m in results for w in m['reasons'].split(',') if w)
    for k, v in c.most_common():
        print(f'  failed {k}: {v} clips')
    print(f'details: {os.path.join(args.raw, "scores.tsv")}')


def _resample(a, sr, target):
    if sr == target:
        return a.astype(np.float32)
    from scipy.signal import resample_poly
    g = np.gcd(sr, target)
    return resample_poly(a, target // g, sr // g).astype(np.float32)


if __name__ == '__main__':
    main()
