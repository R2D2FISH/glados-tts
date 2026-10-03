# GLaDOS Text-to-speech (TTS) Voice Generator
Neural network based TTS Engine.

If you want to just play around with the TTS, this works as stand-alone.
```console
python3 glados-tts/glados.py
```

the TTS Engine can also be used remotely on a machine more powerful then the Pi to process in house TTS: (executed from glados-tts directory
```console
python3 engine-remote.py
```

Default port is 8124
Be sure to update settings.env variable in your main Glados-voice-assistant directory:
```
TTS_ENGINE_API			= http://192.168.1.3:8124/synthesize/
```


## Fast inference (ONNX) and in-browser / WebGPU
The models can be exported to ONNX and run with ONNX Runtime, both in Python and
fully on-device in a browser (WebGPU, with a WebAssembly fallback for phones without it).
```console
pip install -r requirements.txt -r requirements-onnx.txt
python tools/export_onnx.py          # writes onnx/ (~13 s)
python glados_onnx.py "The cake is a lie."
```

Recommended config: ONNX Runtime + the `hq` HiFi-GAN (the default). The small `lq`
vocoder (recovered from `vocoder-cpu-lq.pt`, 1.5M params instead of 14M) is much
faster but has a noticeably fluttery, wet tone. Same sentence, 4.7 s of audio, CPU:

| Config | 1 thread | 4 threads |
|---|---|---|
| stock `glados.py` | 2814 ms (RTF 0.59) | 1222 ms (RTF 0.26) |
| + phonemizer loaded once (now the default) | 2277 ms | 677 ms |
| **ONNX Runtime, `hq` vocoder (default)** | **2030 ms (RTF 0.43)** | **624 ms (RTF 0.13)** |
| ONNX Runtime, `lq` vocoder (`--vocoder lq`) | 387 ms (RTF 0.08) | 182 ms (RTF 0.04) |

### Browser demo
`web/` is a static page that runs the whole pipeline (text normalization,
phonemizer, ForwardTacotron, HiFi-GAN) with onnxruntime-web. To try it locally,
serve the repo root and open `/web/`:
```console
python -m http.server 8000           # then visit http://localhost:8000/web/
```
- It downloads ~110 MB with the `hq` vocoder (fp16 ForwardTacotron, int8 phonemizer, vocoder, dictionary),
  cached in the browser after the first visit. `?models=<url>/` loads them from
  another host (e.g. a Hugging Face repo). GitHub Pages can't serve Git LFS files.
- The vocoder runs on WebGPU when available. fp16 weights are used only if the GPU
  supports `shader-f16`; otherwise fp32. ForwardTacotron runs on WASM because
  its GRU/LSTM layers have no WebGPU kernels.
- On CPU-only (WASM) phones the `hq` vocoder is likely slower than real time;
  the page offers `lq` as a fallback there.
- WASM runs single-threaded unless the page is cross-origin isolated
  (`Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp`).

### Hawaiian names and custom pronunciations
Hawaiian words (street and place names) are read with Hawaiian rules instead of English
spelling guesses (`utils/text/hawaiian.py`): ʻokina as a glottal stop, kahakō as long
vowels, Hawaiian stress (ka-ˌme-ha-ˈme-ha), w as v after i/e, and glides between vowels.
A word counts as Hawaiian if it has an ʻokina or kahakō, or if it is spelled only with
Hawaiian letters in (C)V syllables and isn't an English dictionary word, so
"Likelike" is Hawaiian but "like" and "home" stay English. Spellings with or without
diacritics both work (`Kapiʻolani`, `Kapi'olani`, `Kapiolani`). Turn it off with
`Cleaner(..., hawaiian_names=False)`.

For anything the rules get wrong, add `word<TAB>phonemes` lines to `lexicon.txt`
(phoneme symbols from `utils/text/symbols.py`). Entries override both the
Hawaiian rules and the English phonemizer.

### Training a faster vocoder (distillation)
`tools/distill/` trains a small NSF-style vocoder to imitate the HQ HiFi-GAN. A sine
source with phase-continuous harmonics produces the pitch explicitly, so the student can't warble
the way `lq` does. A small network shapes the rest. Its pitch comes from a tiny
predictor reading the mel, so it is a drop-in mel to audio vocoder. No recordings are needed:
ForwardTacotron + the HQ vocoder generate the training data.
```console
pip install -r requirements.txt -r requirements-distill.txt
python -m tools.distill.make_dataset --num 30000 --out distill/data.pt   # teacher on GPU, ~6 GB
python -m tools.distill.train --data distill/data.pt --out distill/run --preset small --amp
python -m tools.distill.export distill/run/latest.pt                    # -> onnx/vocoder-nsf.onnx
python glados_onnx.py "The cake is a lie." --vocoder nsf
```
Listen to `distill/run/samples/` as it trains (`teacher_*.wav` is the target).
Training resumes from `latest.pt` if restarted. Speed of the presets (untrained
weights; speed doesn't depend on training), 4.7 s of audio:

| Vocoder | ONNX Runtime CPU, 1 thread | Browser WASM, 1 thread |
|---|---|---|
| `hq` | RTF 0.40 | RTF 1.61 |
| `lq` | RTF 0.047 | RTF 0.135 |
| NSF `base` (2.0M) | RTF 0.088 | RTF 0.278 |
| NSF `small` (0.4M) | RTF 0.020 | RTF 0.045 |
| NSF `tiny` (0.1M) | RTF 0.009 | RTF 0.018 |


### Building a synthetic GLaDOS corpus (teacher → ForwardTacotron)
`tools/corpus/` turns a big, GLaDOS-fine-tuned teacher model into a large clean
dataset for training ForwardTacotron (explicit pitch) on GLaDOS alone:
```console
pip install -r requirements.txt -r requirements-corpus.txt
python -m tools.corpus.build_text --out corpus/text.tsv --hours 60 --streets my_streets.txt
python -m tools.corpus.generate --text corpus/text.tsv --out corpus/raw --teacher mymodule:MyTeacher
python -m tools.corpus.filter --raw corpus/raw --reference glados_real/wavs --whisper small.en --utmos
python -m tools.corpus.export --raw corpus/raw --out corpus/dataset \
    --real-metadata glados_real/metadata.csv --real-wavs glados_real/wavs --nsf-data corpus/nsf_data.pt
```
- **build_text**: general sentences + navigation prompts (your street names), numbers and
  times, phonemized with this repo's front-end (Hawaiian rules, `lexicon.txt`) and picked
  greedily to cover as many phoneme pairs as possible.
- **generate**: any teacher via `tools/corpus/teachers.py`: a Python class with
  `synthesize(text, phonemes)`, or `--teacher command --cmd '... {text} ... {out}'`.
  Phoneme-input teachers receive the Hawaiian pronunciations. Resumable; `--takes N` renders
  several versions so the filter can keep the best.
- **filter**: per clip Whisper word errors (Hawaiian words excluded), speaker similarity
  to the real lines, GLaDOS pitch level/spread, speaking rate, long pauses, clipping and
  optional UTMOS naturalness. It also reports how far the teacher sits from the real voice
  overall. Results go to `scores.tsv`.
- **export**: `metadata.csv` (`id|speaker|text`, the `ljspeech_multi` format) + `wavs/` for
  ForwardTacotron, with synthetic clips as `glados_synth` and real lines as `glados_real`,
  plus `--nsf-data` for the distilled vocoder.


## Training (New Model)
The Tacotron and ForwardTacotron models were trained as multispeaker models on two datasets separated into three speakers. LJSpeech (13,100 lines), and then on the heavily modified version of the Ellen McClain dataset, separated into Portal 1 and 2 voices (with punctuation and corrections added manually). The lines from the end of Portal 1 after the cores get knocked off were counted as Portal 2 lines.


## Training (Old Model)
The initial, regular Tacotron model was trained first on LJSpeech, and then on a heavily modified version of the Ellen McClain dataset (all non-Portal 2 voice lines removed, punctuation added).

* The Forward Tacotron model was only trained on about 600 voice lines.
* The HiFiGAN model was generated through transfer learning from the sample.
* All models have been optimized and quantized.



## Installation Instruction
If you want to install the TTS Engine on your machine, please follow the steps
below.

1. Download the model files from [`Google Drive`](https://drive.google.com/file/d/1TRJtctjETgVVD5p7frSVPmgw8z8FFtjD/view?usp=sharing) and unzip into the repo folder
2. Install the required Python packages, e.g., by running `pip install -r
   requirements.txt`
