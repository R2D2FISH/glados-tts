import torch
from utils.tools import getDevice, loadModels, prepare_text, configureEspeak, playAudio, warmupTorch, getOutputFile
from scipy.io.wavfile import write
import time
from sys import modules as mod
import os

configureEspeak()

print("Initializing TTS Engine...")
device = getDevice()

# Load models
(glados, vocoder) = loadModels(device)

print("Warming up...")
warmupTorch(glados, device, vocoder)
print("Warmup done...")

while(1):
    text = input("Input: ")

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
        (output_file, output_file_old) = getOutputFile()
        if os.path.exists(output_file):
            os.rename(output_file, output_file_old)
        
        # Write audio file to disk
        # 22,05 kHz sample rate
        write(output_file, 22050, audio)
        playAudio(output_file)
