# Handoff: state of the GLaDOS TTS work

Context for picking this up in a new session on the user's desktop, the machine
where the original ForwardTacotron training run was done.

## ⚠️ Before any GPU work: cap the 3090 at 1600 MHz
The desktop's 750 W PSU is too weak for the RTX 3090's power spikes. At stock clocks the
machine can lose power at random under load. **Before running anything on the GPU**
(training, dataset generation, even benchmarks), lock the core clock to at most 1600 MHz:
```console
sudo nvidia-smi -lgc 0,1600          # Windows: same command in an Administrator prompt
nvidia-smi -q -d CLOCK               # verify the cap took effect
```
The lock does **not** survive a reboot, so check it again at the start of every session
and after any crash. If you find it unset, set it before continuing. Remove it with
`sudo nvidia-smi -rgc`. If shutdowns still happen, tell the user rather than retrying.

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

## The desktop
- RTX 3090 (24 GB) **capped at 1600 MHz, see above**; 750 W PSU.
- Xeon E5-1680 v2 (8 cores, Ivy Bridge, **no AVX2**), 128 GB RAM. Keep the CPU out of
  training loops (tools/distill already keeps data on the GPU). Use multiprocessing for
  CPU stages (pitch extraction, filtering).
- The original training setup is next to this repo: the as-ideas ForwardTacotron
  checkout in `../ForwardTacotron` (it also has `utils/files.py`, which
  `utils/text/recipes.py` imports).
- **The real GLaDOS datasets (audio + transcriptions) are in `../ForwardTacotron/datasets`.**
  Look at the layout and metadata format there before converting anything. Treat the
  originals as read-only and write derived data to `corpus/` or `distill/` in this repo
  (both gitignored).
- The laptop (RX 6800M) was dropped: it crashes too often. Ignore AMD/ROCm advice.
  With NVIDIA, `--amp` (bf16) and faster-whisper on CUDA both work.

## Next steps
1. Set the 1600 MHz cap. Check `torch.cuda.is_available()` and versions; note whatever
   Python/torch environment the original ForwardTacotron run used before installing anything new.
2. `git lfs pull`, `python tools/export_onnx.py`, then a short `tools/distill` run as a GPU smoke test.
3. Inspect `../ForwardTacotron/datasets`: how many real GLaDOS lines/hours, P1 vs P2 split,
   sample rate, transcription quality. This is the fine-tuning set for the teacher and the
   `--reference` set for `tools/corpus/filter.py`. Hold some lines out for blind listening tests.
4. Teacher candidate: StyleTTS2 (explicit duration/F0/energy, phoneme input, 24 kHz,
   MIT code; check checkpoint terms). Zero-shot baseline, then fine-tune on the real
   GLaDOS lines; with 24 GB the SLM adversarial stage may fit (it is optional).
   Optional A/B: F5-TTS (text input, non-commercial weights).
5. Write a `tools/corpus/teachers.py` adapter for it (map our IPA to its espeak symbols),
   generate a few hundred sentences, run the filter, listen, compare teacher-vs-real speaker sim.
6. Then the full corpus (~60 h) -> train ForwardTacotron in `../ForwardTacotron` on
   `glados_synth` + `glados_real` speakers.
- Open idea: address-style number reading ("3392 Liliha" -> "thirty-three ninety-two").
