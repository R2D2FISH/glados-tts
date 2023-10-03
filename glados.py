import torch
import numpy as np
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time
import tempfile
import subprocess
from pydub import AudioSegment
from pydub.playback import play
from nltk import download
from nltk.tokenize import sent_tokenize
from sys import modules as mod
try:
    import winsound
except ImportError:
    from subprocess import call
print("Initializing TTS Engine...")

kwargs = {
    'stdout':subprocess.PIPE,
    'stderr':subprocess.PIPE,
    'stdin':subprocess.PIPE
}

class tts_runner:
    def __init__(self, use_p1: bool=False, log: bool=False):
        self.log = log
        if use_p1:
            self.emb = torch.load('models/emb/glados_p1.pt')
        else:
            self.emb = torch.load('models/emb/glados_p2.pt')
        # Select the device
        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.is_vulkan_available():
            self.device = 'vulkan'
        else:
            self.device = 'cpu'

        # Load models
        self.glados = torch.jit.load('models/glados-new.pt')
        self.vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=self.device)
        for i in range(2):
            init = self.glados.generate_jit(prepare_text(str(i)), self.emb, 1.0)
            init_mel = init['mel_post'].to(self.device)
            init_vo = self.vocoder(init_mel)

    def run_tts(self, text, alpha: float=1.0) -> AudioSegment:
        x = prepare_text(text)

        with torch.no_grad():

            # Generate generic TTS-output
            old_time = time.time()
            tts_output = self.glados.generate_jit(x, self.emb, alpha)
            if self.log:
                print("Forward Tacotron took " + str((time.time() - old_time) * 1000) + "ms")

            # Use HiFiGAN as vocoder to make output sound like GLaDOS
            old_time = time.time()
            mel = tts_output['mel_post'].to(self.device)
            audio = self.vocoder(mel)
            if self.log:
                print("HiFiGAN took " + str((time.time() - old_time) * 1000) + "ms")

            # Normalize audio to fit in wav-file
            audio = audio.squeeze()
            audio = audio * 32768.0
            audio = audio.cpu().numpy().astype('int16')
            output_file = tempfile.TemporaryFile()
            write(output_file, 22050, audio)
            sound = AudioSegment.from_wav(output_file)
            output_file.close()
            return sound

    def speak_one_line(self, audio, name: str):
        audio.export(name, format = "wav")
        if 'winsound' in mod:
            winsound.PlaySound(name, winsound.SND_FILENAME | winsound.SND_ASYNC)
        else:
            try:
                subprocess.Popen(["play", name], **kwargs)
            except FileNotFoundError:
                try:
                    subprocess.Popen(["aplay", name], **kwargs)
                except FileNotFoundError:
                    subprocess.Popen(["pw-play", name], **kwargs)


    def speak(self, text, alpha: float=1.0, save: bool=False, delay: float=0.1):
        download('punkt',quiet=self.log)
        sentences = sent_tokenize(text)
        audio = self.run_tts(sentences[0])
        pause = AudioSegment.silent(duration=delay)
        old_line = AudioSegment.silent(duration=1.0) + audio
        self.speak_one_line(old_line, "old_line.wav")
        old_time = time.time()
        old_dur = old_line.duration_seconds
        new_dur = old_dur
        if len(sentences) > 1:
            for idx in range(1, len(sentences)):
                if idx % 2 == 1:
                    new_line = self.run_tts(sentences[idx])
                    audio = audio + pause + new_line
                    new_dur = new_line.duration_seconds
                else:
                    old_line = self.run_tts(sentences[idx])
                    audio = audio + pause + old_line
                    new_dur = old_line.duration_seconds
                time_left = old_dur - time.time() + old_time
                if time_left <= 0 and self.log:
                    print("Processing is slower than realtime!")
                else:
                    time.sleep(time_left + delay)
                if idx % 2 == 1:
                    self.speak_one_line(new_line, "new_line.wav")
                else:
                    self.speak_one_line(old_line, "old_line.wav")
                old_time = time.time()
                old_dur = new_dur
        else:
            time.sleep(old_dur + 0.1)

        audio.export("output.wav", format = "wav")
        time_left = old_dur - time.time() + old_time
        if time_left >= 0:
            time.sleep(time_left + delay)

if __name__ == "__main__":
    glados = tts_runner(False, True)
    while True:
        text = input("Input: ")
        if len(text) > 0:
            glados.speak(text, True)
