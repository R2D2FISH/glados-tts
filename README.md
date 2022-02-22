# GLaDOS Text-to-speech (TTS) Voice Generator
Neural network based TTS Engine modified to work with the [GLaDOS Voice Assistant](https://github.com/nerdaxic/glados-voice-assistant/).

Install to working directory of the voice assistant with:
```console
cd ~
cd glados-voice-assistant
git submodule add https://github.com/nerdaxic/glados-tts.git glados_tts
```

If you want to just play around with the TTS, this works as stand-alone.
```console
python3 glados-tts/glados.py
```

## Description
The initial, regular Tacotron model was trained first on LJSpeech, and then on a heavily modified version of the Ellen McClain dataset (all non-Portal 2 voice lines removed, punctuation added).

* The Forward Tacotron model was only trained on about 600 voice lines.
* The HiFiGAN model was generated through transfer learning from the sample.
* All models have been optimized and quantized.
