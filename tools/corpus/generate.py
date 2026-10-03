"""Generate teacher audio for every sentence of a text.tsv (from build_text.py).

    python -m tools.corpus.generate --text corpus/text.tsv --out corpus/raw --teacher mymodule:MyTeacher
    python -m tools.corpus.generate --text corpus/text.tsv --out corpus/raw --teacher command \\
        --cmd 'python infer.py --text {text} --out {out}'

Resumable: sentences that already have a wav are skipped. --takes N renders N
versions of each sentence (useful with sampling teachers: the filter then keeps
the best take). Writes <out>/<id>_<take>.wav and appends to <out>/manifest.tsv.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from scipy.io import wavfile

sys.path.insert(0, os.getcwd())
from tools.corpus.teachers import load_teacher  # noqa: E402


def read_tsv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f, delimiter='\t', quoting=csv.QUOTE_NONE))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--text', default='corpus/text.tsv')
    ap.add_argument('--out', default='corpus/raw')
    ap.add_argument('--teacher', default='glados')
    ap.add_argument('--cmd', help='command template for --teacher command')
    ap.add_argument('--takes', type=int, default=1)
    ap.add_argument('--shard', default='0/1', help='i/n: process every n-th sentence starting at i (multi-GPU)')
    args = ap.parse_args()

    kw = {'cmd': args.cmd} if args.cmd else {}
    teacher = load_teacher(args.teacher, **kw)
    rows = read_tsv(args.text)
    si, sn = map(int, args.shard.split('/'))
    rows = rows[si::sn]
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, 'manifest.tsv')
    new = not os.path.exists(manifest)
    t0, done, secs = time.time(), 0, 0.0
    with open(manifest, 'a', encoding='utf-8') as mf:
        if new:
            mf.write('file\tid\ttake\tkind\ttext\tphonemes\n')
        for r in rows:
            for take in range(args.takes):
                name = f"{r['id']}_{take}.wav"
                path = os.path.join(args.out, name)
                if os.path.exists(path):
                    continue
                try:
                    a = teacher.synthesize(r['text'], r['phonemes'])
                except Exception as e:  # one bad sentence shouldn't stop a multi-hour run
                    print(f"  {r['id']}: {type(e).__name__}: {e}", file=sys.stderr)
                    continue
                a = np.clip(np.asarray(a, np.float32), -1, 1)
                wavfile.write(path + '.tmp', teacher.sample_rate, (a * 32767).astype(np.int16))
                os.replace(path + '.tmp', path)
                mf.write(f"{name}\t{r['id']}\t{take}\t{r['kind']}\t{r['text']}\t{r['phonemes']}\n")
                mf.flush()
                done += 1
                secs += len(a) / teacher.sample_rate
                if done % 100 == 0:
                    el = time.time() - t0
                    print(f'  {done} clips, {secs / 3600:.2f} h audio, {secs / el:.1f}x real time', flush=True)
    print(f'done: {done} new clips ({secs / 3600:.2f} h) in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
