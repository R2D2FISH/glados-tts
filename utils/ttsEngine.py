import utils.tools as tools

import sys
import os
sys.path.insert(0, os.getcwd()+'/glados_tts')

import torch
from scipy.io.wavfile import write
import time
import hashlib
import io

import pyaudio
import wave
import sys

class TTSEngine:
    def __init__(self):
        tools.configureEspeak()
        self.device = tools.getDevice()
        (self.glados, self.vocoder) = tools.loadModels(self.device)
    
    def warmup(self, count = 1):
        tools.warmupTorch(self.glados, self.device, self.vocoder, count)
    
    def generate(self, text):
        x = tools.prepare_text(text).to('cpu')
        with torch.no_grad():
            # Generate generic TTS-output
            start_time = time.time()
            tts_output = self.glados.generate_jit(x)
            tacotron_time_ms = (time.time() - start_time) * 1000

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            mel = tts_output['mel_post'].to(self.device)
            audio = self.vocoder(mel)
            hifi_time_ms = (time.time() - start_time) * 1000 - tacotron_time_ms

            # Normalize audio to fit in wav-file
            audio = audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')
            finalize_time_ms = (time.time() - start_time) * 1000 - tacotron_time_ms - hifi_time_ms

            return TTSResult(audio, tacotron_time_ms, hifi_time_ms, finalize_time_ms)

class TTSResult:
    def __init__(self, audio, tacotron_time_ms, hifi_time_ms, finalize_time_ms):
        self.audio = audio
        self.tacotron_time_ms = tacotron_time_ms
        self.hifi_time_ms = hifi_time_ms
        self.finalize_time_ms = finalize_time_ms

    def getAudio(self):
        return self.audio

    def getTacotronTime(self):
        return self.tacotron_time_ms

    def getHifiTime(self):
        return self.hifi_time_ms

    def getFinalizeTime(self):
        return self.finalize_time_ms

    def save(self, filename):
        write(filename, 22050, self.audio)
    
    def asMemoryFile(self):
        file_format = "WAV"
        memory_file = io.BytesIO( )
        memory_file.name = "audio.wav"
        self.save(memory_file)
        memory_file.seek(0)
        return memory_file

    def play(self):
        CHUNK = 1024
        wf = wave.open(self.asMemoryFile(), 'rb')

        p = pyaudio.PyAudio()

        stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                        channels=wf.getnchannels(),
                        rate=wf.getframerate(),
                        output=True)

        data = wf.readframes(CHUNK)

        while len(data):
            stream.write(data)
            data = wf.readframes(CHUNK)

        stream.stop_stream()
        stream.close()

        p.terminate()

    def __str__(self):
        return  "Tacotron: " + str(self.tacotron_time_ms) + "ms, " +\
                "HiFiGAN: " + str(self.hifi_time_ms) + "ms, " +\
                "Finalize: " + str(self.finalize_time_ms) + "ms"

