"""Render a manifest with the teacher: audio plus everything a student needs.

Per utterance this writes
  wav/<utt>.wav          22.05 kHz int16 from the teacher vocoder
  feat/<utt>.npz         tokens, mel_post (80 x T), dur, pitch, energy
and appends to metadata.csv (utt|phonemes|frames). The per-token durations
mean the student acoustic model trains without an aligner, and (mel, wav)
pairs from the teacher vocoder are ready-made vocoder-student data.

Shard with --shard i/n to spread across processes or machines; finished
utterances are skipped, so it resumes after an interruption.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def render(manifest: Path, out: Path, model: Path, vocoder: Path, emb: Path,
           shard: str = '0/1', device: str = 'auto', limit: int = 0) -> int:
    import torch
    from scipy.io.wavfile import write as write_wav
    from utils.text.tokenizer import Tokenizer

    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    k, n = map(int, shard.split('/'))
    tts = torch.jit.load(str(model), map_location=device)
    voc = torch.jit.load(str(vocoder), map_location=device)
    speaker = torch.load(str(emb), map_location=device)
    tok = Tokenizer()
    (out / 'wav').mkdir(parents=True, exist_ok=True)
    (out / 'feat').mkdir(parents=True, exist_ok=True)
    meta = open(out / f'metadata.{k}.csv', 'a', encoding='utf-8')
    done = 0
    t0 = time.time()
    with open(manifest, encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i % n != k:
                continue
            row = json.loads(line)
            utt = row['utt']
            if (out / 'feat' / f'{utt}.npz').exists():
                continue
            ids = tok(row['phonemes'])
            if not ids:
                continue
            x = torch.as_tensor(ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                o = tts.generate_jit(x, speaker, float(row['alpha']))
                mel = o['mel_post']
                audio = voc(mel.to(device)).squeeze().clamp(-1, 1).cpu().numpy()
            write_wav(out / 'wav' / f'{utt}.wav', 22050, (audio * 32767).astype(np.int16))
            np.savez_compressed(out / 'feat' / f'{utt}.npz', tokens=np.array(ids, dtype=np.int16),
                                **{key: o[key].squeeze(0).cpu().numpy().astype(np.float32)
                                   for key in ('mel_post', 'dur', 'pitch', 'energy') if key in o})
            meta.write(f"{utt}|{row['phonemes']}|{mel.shape[-1]}\n")
            done += 1
            if done % 100 == 0:
                meta.flush()
                print(f'{done} rendered, {done / (time.time() - t0):.2f} utt/s')
            if limit and done >= limit:
                break
    meta.close()
    return done
