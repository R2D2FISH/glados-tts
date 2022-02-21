# importing sys
import sys
  
# adding Folder_2/subfolder to the system path
sys.path.insert(0, '/home/nerdaxic/glados-voice-assistant/glados_tts')

import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
from playsound import playsound as ps
import time



glados = torch.jit.load('glados_tts/models/glados.pt')

if torch.is_vulkan_available():
    device = 'vulkan'
    vocoder = torch.jit.load('glados_tts/models/vocoder-gpu.pt')
    print("Initializing TTS Engine on GPU (Vulkan)")
if torch.cuda.is_available():
    device = 'cuda'
    vocoder = torch.jit.load('glados_tts/models/vocoder-gpu.pt')
    print("Initializing TTS Engine on GPU (CUDA)")
else:
    device = 'cpu'
    vocoder = torch.jit.load('glados_tts/models/vocoder-cpu-hq.pt')
    print("Initializing TTS Engine on CPU")

glados.cpu()
vocoder.to(device)

def speak(text):
    x = prepare_text(text).to('cpu')

    with torch.no_grad():
        old_time = time.time()
        tts_output = glados.generate_jit(x)
        print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")
        old_time = time.time()
        mel = tts_output['mel_post'].cpu()
        audio = vocoder(mel)
        print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")
        audio = audio.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype('int16')
        output_file = ('output.wav')
        write(output_file, 22050, audio)
        ps('output.wav')