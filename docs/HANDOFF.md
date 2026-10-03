# Handoff: state of the GLaDOS TTS work

Context for picking this up in a new session (e.g. on the laptop).

## Done (all on this branch)
- **ONNX + browser**: `tools/export_onnx.py`, `glados_onnx.py`, `web/` (onnxruntime-web,
  WebGPU with WASM fallback). Default vocoder is `hq`; `lq` was judged too warbly.
- **Text front-end**: phonemizer loaded once; torch>=2.6 fix; Hawaiian place-name rules
  (`utils/text/hawaiian.py`, mirrored in `web/text.js`) and `lexicon.txt` overrides.
- **Distilled vocoder** (`tools/distill/`): NSF student (explicit sine source + small
  filter net + mel->f0 predictor). Speeds measured; not yet trained on a GPU.
- **Synthetic corpus pipeline** (`tools/corpus/`): build_text -> generate (pluggable
  teacher) -> filter (Whisper WER, speaker sim, GLaDOS pitch profile, rate, pauses,
  UTMOS) -> export (ljspeech_multi metadata.csv + wavs, `glados_synth` / `glados_real`
  speakers, NSF data). Tested end to end on CPU with the current model as teacher;
  Whisper/UTMOS paths untested (downloads were blocked in the sandbox).

## Key findings
- ForwardTacotron pitch -> Hz is linear: Hz ~ 30.3*p + 163 (P2), 30.1*p + 167 (P1).
- GLaDOS f0 is narrow (130-210 Hz), smooth, mildly snapped to semitones.
- Model mel: 22050 Hz, hop 256, 80 bins, 0-8000 Hz, slaney, natural log; nothing
  above 8 kHz, so vocoders invent that band (likely part of the harshness at volume).
- The user prefers explicit-pitch models; plan is teacher -> big synthetic corpus ->
  ForwardTacotron trained on GLaDOS only.

## Next steps (laptop: RX 6800M 12 GB, Linux recommended)
1. Ubuntu 24.04 + ROCm + PyTorch ROCm wheels; RDNA2 gfx1031 needs
   `HSA_OVERRIDE_GFX_VERSION=10.3.0`. Check `torch.cuda.is_available()`.
   Train in fp32 (no `--amp`: RDNA2 has no fast bf16). faster-whisper runs on CPU on AMD.
2. `git lfs pull`, `python tools/export_onnx.py`, short `tools/distill` run as a GPU smoke test.
3. Teacher candidate: StyleTTS2 (explicit duration/F0/energy, phoneme input, 24 kHz,
   MIT code; check checkpoint terms). Zero-shot baseline, then short fine-tune on the real
   GLaDOS lines, skipping the SLM adversarial stage on 12 GB. Optional A/B: F5-TTS.
4. Write a `tools/corpus/teachers.py` adapter for it (map our IPA to its espeak symbols),
   generate a few hundred sentences, run the filter, listen, compare teacher-vs-real speaker sim.
- Open idea: address-style number reading ("3392 Liliha" -> "thirty-three ninety-two").
