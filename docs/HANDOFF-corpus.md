# Handoff: distillation corpus (branch `corpus/distill-corpus`)

This branch was built from `main` without knowing about `tts-retraining`, which
already has `tools/corpus/` (teacher adapters, Whisper/UTMOS filter, export) and
`tools/distill/`. Read `docs/HANDOFF.md` on `tts-retraining` too, including the
3090 clock cap.

## What this branch adds (`corpus/`)
- OSM harvest + Valhalla routing (pyvalhalla, no Docker), per-state tiles, routes
  rendered as Valhalla spoken instructions and terse app-style text.
- Nav-aware normalizer (`St.` street vs saint, I-405, H-1, FM 1960, exits, distances,
  times, dates, ZIPs, grid addresses).
- Hawaiian G2P with two registers: marked spelling (ʻokina/kahakō) -> strict Hawaiian;
  unmarked -> Hawaiʻi English (full vowels, Hawaiian stress, marked form's syllable
  breaks). Hawaiian only if marked or in the OSM-harvested lexicon (Pewaukee stays English).
- Profiles (tiny/small/general + held-out eval) filled by coverage per second of audio.
- Teacher render saving wav + mel + per-token dur/pitch/energy (ForwardTacotron jit API).
- `prose-llm`: in-character prose via `claude -p` (Sonnet 5.5) or the API.
- Untested on real US data: Geofabrik was blocked in the cloud sandbox. Run
  `python -m corpus osm --regions hawaii` first and check how Valhalla words US refs.

## Open decision: the phoneme alphabet
The new teacher is StyleTTS2 (local only, fine-tuned on the full GLaDOS corpus with
Whisper-cross-checked transcripts). Its input alphabet is whatever phonemized the
fine-tuning list. Check it with:

```python
# python check_ft.py Configs/config_ft.yml
import sys, yaml, collections
cfg = yaml.safe_load(open(sys.argv[1])); dp = cfg['data_params']
print({k: dp.get(k) for k in ('train_data', 'val_data', 'OOD_data', 'root_path')})
lines = open(dp['train_data'], encoding='utf-8').read().splitlines()
print(len(lines), 'lines'); print(*lines[:3], sep='\n')
chars = collections.Counter(c for l in lines for c in l.split('|')[1])
print('symbols:', ''.join(sorted(chars)))
print('stress ˈ:', chars['ˈ'], ' ɹ:', chars['ɹ'], ' r:', chars['r'], ' ɾ:', chars['ɾ'], ' ʔ:', chars['ʔ'])
```

- `ˈ` and `ɹ` present -> espeak-ng. Frontend must emit espeak-style IPA: stress mark
  before the vowel (`həlˈoʊ`), `ɹ`, `ɐ`, `ɾ`, `ʔ` (espeak uses it: `bˈʌʔn̩`). Port
  `hawaiian.PHONES` to that style (stress already computed; `to_ipa(show_stress=True)`
  shows placement), and re-express `data/pronunciations.tsv`.
- plain `r`, no `ˈ` -> deep-phonemizer; the current frontend already matches.

Findings that bear on it:
- deep-phonemizer: 15.2M params (65 MB), dictionary of ~124k words, ~46 ms per sentence
  with an unknown word (x86, 1 thread); ~5.6% of corpus words were unknown (street names).
- espeak-ng: 0.2 ms per sentence, but misreads the same things (Waikiki -> wˈeɪkɪki,
  Likelike -> lˈaɪklaɪk, "W 5th St." -> dubya fifth sənt, Schuylkill, Ojai).
- Windows install pain is solved by `pip install espeakng-loader`: the win_amd64 wheel
  bundles `espeak-ng.dll` and the English data (point phonemizer at
  `espeakng_loader.get_library_path()` / `get_data_path()`).
- The user prefers deep-phonemizer for unseen words. Option that keeps that: retrain a
  deep-phonemizer on CMUdict converted to espeak-style IPA (CMUdict stress digits give
  the marks) + harvested street names + Hawaiian G2P output; then a 1-3M param version
  for the Pi. Students can use any alphabet since they train from scratch, but sharing
  the teacher's keeps one frontend.

## Next steps
1. Run the check above; push the StyleTTS2 inference code (or open a session on the
   desktop) with: the inference function, how the style vector is made (GLaDOS
   reference clip or diffusion), and `text_utils.py` if the symbols changed.
2. Consolidate onto `tts-retraining`: move this branch's additions into `tools/corpus/`
   behind its `synthesize(text, phonemes)` teacher interface; keep its filter/export;
   merge the two Hawaiian readers; write a StyleTTS2 teacher that also returns the
   predicted per-phoneme durations and F0 (free targets for explicit-pitch students).
3. Keep the corrected transcripts as plain text next to the phoneme lists (re-phonemize
   if the frontend changes) and hold some real lines out for blind tests and the
   filter's speaker-similarity reference.
4. Optional: feed the corpus prose/nav pool to StyleTTS2's `OOD_data` for the SLM
   adversarial stage, phonemized with the same frontend.
