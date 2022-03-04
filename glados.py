import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time
from subprocess import call

print("Initializing TTS Engine...")

glados = torch.jit.load('models/glados.pt')

if torch.is_vulkan_available():
    device = 'vulkan'
    vocoder = torch.jit.load('models/vocoder-gpu.pt')
if torch.cuda.is_available():
    device = 'cuda'
    vocoder = torch.jit.load('models/vocoder-gpu.pt')
else:
    device = 'cpu'
    vocoder = torch.jit.load('models/vocoder-cpu-hq.pt')

glados.cpu()
vocoder.to(device)

while(1):
    text = input("Input: ")

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
        call(["aplay", "./output.wav"])
