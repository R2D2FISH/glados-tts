# importing sys
import sys
import os

sys.path.insert(0, os.getcwd()+'/glados_tts')

import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time

sys.path.insert(0, os.getcwd()+'/glados_tts')

print("Initializing TTS Engine...")

# Select the device
if torch.is_vulkan_available():
    device = 'vulkan'
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

# Load models
glados = torch.jit.load('glados_tts/models/glados.pt')
vocoder = torch.jit.load('glados_tts/models/vocoder-gpu.pt', map_location=device)

# Prepare models in RAM
for i in range(4):
    init = glados.generate_jit(prepare_text(str(i)))
    init_mel = init['mel_post'].to(device)
    init_vo = vocoder(init_mel)


def glados_tts(text):

    # Tokenize, clean and phonemize input text
    x = prepare_text(text).to('cpu')

    with torch.no_grad():

        # Generate generic TTS-output
        old_time = time.time()
        tts_output = glados.generate_jit(x)
        print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")

        # Use HiFiGAN as vocoder to make output sound like GLaDOS
        old_time = time.time()
        mel = tts_output['mel_post'].to(device)
        audio = vocoder(mel)
        print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")

        # Normalize audio to fit in wav-file
        audio = audio.squeeze()
        audio = audio * 32768.0
        audio = audio.cpu().numpy().astype('int16')
        output_file = ('output.wav')

        # Write audio file to disk
        # 22,05 kHz sample rate 
        write(output_file, 22050, audio)

    return True