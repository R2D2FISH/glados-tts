
from flask import Flask, request, send_file
import torch
from utils.tools import prepare_text
from scipy.io.wavfile import write
import time
import os
import urllib.parse

app = Flask(__name__)

@app.route('/synthesize/', defaults={'text': ''})
@app.route('/synthesize/<path:text>')
def synthesize(text):
	text = request.url[request.url.find('synthesize/')+11:]
	if(text == ''): return 'No input'
	if(glados_tts(urllib.parse.unquote(text))):
		return send_file(os.getcwd()+'/output.wav')
	else:
		return 'TTS Engine Failed'
		
print("Initializing TTS Engine...")

# Select the device
if torch.is_vulkan_available():
	device = 'vulkan'
if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'

# Load models
glados = torch.jit.load('models/glados.pt')
vocoder = torch.jit.load('models/vocoder-gpu.pt', map_location=device)

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

print("Initializing TTS Server...")
if __name__ == "__main__":
	app.run(host="0.0.0.0", port=8124)