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

### Linux (bash)

```bash
# 1. Clone the project (Ignore errors if any): 
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/R2D2FISH/glados-tts && cd glados-tts
# 2. Download the model files:
python3 download.py
# 3. Create a venv (Optional)
python3 -m venv .venv && source .venv/bin/activate
# 4. Install requirements
pip install -r requirements.txt
# 5. Run the program
python3 glados-tts/glados.py
# If you installed a venv dont forget to activate it using source .venv/bin/activate before running it next time
```


### Windows (cmd)

```CMD
:: 1. Set the GIT_LFS_SKIP_SMUDGE environment variable
set GIT_LFS_SKIP_SMUDGE=1

:: 2. Clone the project (Ignore errors if any): 
git clone https://github.com/R2D2FISH/glados-tts
cd glados-tts

:: 3. Download the model files:
python download.py

:: 4. Create a venv (Optional)
python -m venv .venv
call .venv\Scripts\activate

:: 5. Install requirements
pip install -r requirements.txt

:: 6. Run the program
python glados-tts\glados.py
```