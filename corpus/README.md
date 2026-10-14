# Distillation corpus

Builds text corpora for distilling the GLaDOS teacher into small students. A
nav-heavy mix suits a Pi Zero-class voice; a prose-heavy mix suits a general
voice. It also renders that text with the teacher, so each student gets the
teacher's audio, mels and per-token durations, pitch and energy.

```
OSM extracts ──► harvest ──► names, Hawaiian lexicon, route endpoints
                    │
                    └──► Valhalla tiles ──► sampled routes ──► terse + spoken instructions
curated text, place names, numbers, prose (books / Claude-written) ─┐
                                                                    ▼
                         pool ──► frontend (normalize + Hawaiian G2P + phonemizer)
                                    ▼
                         eval set (held out) + per-profile manifests
                                    ▼
                         teacher render: wav + mel + dur/pitch/energy
```

## Setup

```bash
pip install -r corpus/requirements.txt    # pyosmium, pyvalhalla, pyyaml, scipy (+ repo requirements)
git lfs pull                              # phonemizer + teacher weights
```

pyvalhalla ships the Valhalla binaries, so no Docker or system install is
needed. Building tiles for a mid-size state (Washington, Texas) takes a few
GB of RAM and some minutes. Hawaii takes seconds.

## Run it

```bash
python -m corpus osm --regions hawaii                 # download, harvest, tiles, routes
python -m corpus osm --regions all                    # every region in config/regions.yaml
python -m corpus prose-llm --batches 200              # optional, needs Claude API credentials
python -m corpus build --profiles eval,tiny,small,general --extra-dir ~/books
python -m corpus render --manifest corpus_work/manifests/tiny.jsonl \
       --model path/to/your/epoch80.pt --shard 0/2    # run 1/2 in another process
python -m corpus say "In 0.3 mi, turn L onto Kalanianaʻole Hwy" --nav
python -m pytest corpus/tests
```

Everything is written under `corpus_work/` (or the path given to `--work`):

| path | contents |
|---|---|
| `osm/*.osm.pbf` | Geofabrik extracts (drop your own there to skip downloads) |
| `tiles/<state>/` | Valhalla tiles, shared by regions cut from the same state |
| `regions/<r>/names.tsv`, `refs.tsv` | every road name and ref with counts |
| `regions/<r>/hawaiian_lexicon.tsv` | Hawaiian words: marked spelling, bare spelling, variants |
| `regions/<r>/routes.jsonl` | raw Valhalla maneuvers, so text can be re-rendered without routing again |
| `prose/*.txt` | Claude-written prose |
| `phonemes.<hash>.jsonl` | frontend cache; a new hash appears when the frontend or lexicons change |
| `manifests/{eval,tiny,small,general}.jsonl` | `utt, category, text, spoken, phonemes, alpha, est_sec, ...` |
| `manifests/report.json` | hours and items per category, for the pool and for each profile |
| `render/<manifest>/` | `wav/`, `feat/*.npz` (tokens, mel_post, dur, pitch, energy), `metadata.<shard>.csv` |

## Profiles

`config/profiles.yaml` sets the hours and per-category weights for each
profile. Weights are shares of audio time, not item counts.

- **tiny** is about 60% navigation.
- **general** is about 60% prose and balanced text.
- **eval** is drawn first and excluded from every training profile. Render it
  with every checkpoint and vocoder you want to compare.

Within each category, items are chosen greedily by new diphones, words and
street names gained per second of audio. Near-duplicate templates are capped,
so "Turn right onto X" can't fill the nav budget. If a category runs short,
the build prints a note and gives the leftover time to the other categories.
Prose will run short until you add books (`--extra-dir`) or run `prose-llm`.

## Text frontend

`corpus/frontend.py` is the student's text frontend too. Keep it identical
between teacher render and student inference.

- **Normalizer** (`normalize.py`, nav mode): handles street-type suffixes
  versus titles ("Main St." is Street, "St. Charles" is Saint), compass
  letters, route refs (I-405 is "I four oh five", H-1 is "H one", FM 1960 is
  "F M nineteen sixty"), exits, distances, times, dates, ZIP codes, and
  house/grid numbers ("500 South 700 East").
- **Hawaiian G2P** (`hawaiian.py`): stress uses Pukui–Elbert feet.
  - Spellings with an ʻokina or kahakō are read strictly: a word break for
    the ʻokina, long vowels, and reduced unstressed *a*.
  - Unmarked spellings are read as Hawaiʻi English: full vowels and Hawaiian
    stress, with no glottal stops or vowel length. They still keep the
    syllable breaks of the known marked form, so "Kalanianaole" doesn't grow
    an *ao* diphthong.
  - A word counts as Hawaiian only if it is marked or appears in the Hawaii
    lexicon, so mainland names like Pewaukee and Kalamazoo stay English.
- **Symbol set**: the GLaDOS phonemizer emits no stress marks and no `ʔ`, so
  the G2P doesn't either. Stress is carried by vowel length and quality, the
  way the dictionary does it (`honolulu → hɑːnəluːluː`). Vowel choices are in
  `hawaiian.PHONES`, which you can tune by ear.
- **Overrides** (`data/pronunciations.tsv`): known place-name fixes such as
  Ojai, Puyallup, Worcester and Schuylkill, in the phonemizer's alphabet.

## Notes

- `vocoder-cpu-hq.pt` doesn't load on current PyTorch (an xnnpack op), so
  `render` defaults to `vocoder-gpu.pt`, which also runs on CPU.
- Generated route text derives from OpenStreetMap (ODbL). If you publish the
  corpus or audio, credit "© OpenStreetMap contributors".
