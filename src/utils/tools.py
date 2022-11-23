import torch
import os

from utils.cleaners import Cleaner
from utils.tokenizer import Tokenizer


def prepare_text(text: str) -> str:
    if not ((text[-1] == '.') or (text[-1] == '?') or (text[-1] == '!')):
        text = text + '.'
    cleaner = Cleaner('english_cleaners', True, 'en-us')
    tokenizer = Tokenizer()
    return torch.as_tensor(tokenizer(cleaner(text)), dtype=torch.int, device='cpu').unsqueeze(0)


def playAudio(path: str):
    # Return if audio-enabled is disabled or not set
    audioEnabled = str(os.environ.get('AUDIO_ENABLED'))
    if audioEnabled == 'false':
        print('Audio is disabled')
        return
    else:
        print('Audio is enabled')

    if os.name == 'nt':
        import winsound
        winsound.PlaySound(path, winsound.SND_FILENAME)
    else:
        from subprocess import call
        try:
            try:
                call(["aplay", path])
            except FileNotFoundError:
                call(["pw-play", path])
        except FileNotFoundError:
            print("Could not play audio file. Please play it manually.")


def configureEspeak():
    if os.environ.get('PHONEMIZER_ESPEAK_LIBRARY') is not None:
        return

    if os.name == 'nt':
        os.environ['PHONEMIZER_ESPEAK_LIBRARY'] = 'C:\Program Files\eSpeak NG\libespeak-ng.dll'
        os.environ['PHONEMIZER_ESPEAK_PATH'] = 'C:\Program Files\eSpeak NG\espeak-ng.exe'


def warmupTorch(glados, device, vocoder, count = 1):
    for i in range(count):
        init = glados.generate_jit(prepare_text(str(1)))
        init_mel = init['mel_post'].to(device)
        init_vo = vocoder(init_mel)


def getOutputFile():
    parent = ''
    if os.environ.get('OUTPUT_DIRECTORY') is not None:
        parent = os.environ.get('OUTPUT_DIRECTORY') + '/'
    else:
        parent = 'audio/'

    # os.path.mkdir(parent, exist_ok=True)
    os.makedirs(parent, exist_ok=True)

    i = 0
    while os.path.exists(parent + "output%s.wav" % i):
        i += 1

    output = parent + "output%s.wav" % i

    return (parent + "output.wav", output)

def getDevice():
    if torch.is_vulkan_available():
        return 'vulkan'
    if torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'

def loadModels(device):
    glados = loadJit('models/glados.pt', 'glados_tts/models/glados.pt')
    vocoder = loadJit('models/vocoder-gpu.pt', 'glados_tts/models/vocoder-gpu.pt', device)
    return (glados, vocoder)

def loadJit(path, alternatePath, device = None):
    if os.path.exists(path):
        return torch.jit.load(path, map_location=device)
    elif os.path.exists(alternatePath):
        return torch.jit.load(alternatePath, map_location=device)
    else:
        print("Could not find model.")




    # def load(self):
    #     with open(self.path, 'r') as f:
    #         for line in f:
    #             if line.startswith('#'):
    #                 continue
    #             if '=' not in line:
    #                 continue
    #             key, value = line.split('=', 1)
    #             self.config[key.strip()] = value.strip()

    # def get(self, key):
    #     return self.config[key]

    # def set(self, key, value):
    #     self.config[key] = value

    # def save(self):
    #     with open(self.path, 'w') as f:
    #         for key, value in self.config.items():
    #             f.write(f'{key}={value}

def getInputText(args):
    if args.input:
        with open(args.input, 'r') as f:
            generateText = f.read()
    else:
        generateText = ' '.join(args.text)
    return generateText