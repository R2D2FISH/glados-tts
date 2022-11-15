import sys
import os
sys.path.insert(0, os.getcwd()+'/glados_tts')

import torch
from utils.tools import getDevice, loadModels, prepare_text, configureEspeak, playAudio, warmupTorch, getOutputFile
from scipy.io.wavfile import write
import time
import hashlib

print("\033[1;94mINFO:\033[;97m Initializing TTS Engine...")
configureEspeak()
device = getDevice()

# Load models
(glados, vocoder) = loadModels(device)

print("Warming up...")
warmupTorch(glados, device, vocoder, 4)
print("Warmup done...")


def glados_tts(text, output_file):

	# Tokenize, clean and phonemize input text
	x = prepare_text(text).to('cpu')

	with torch.no_grad():

		# Generate generic TTS-output
		old_time = time.time()
		tts_output = glados.generate_jit(x)

		# Use HiFiGAN as vocoder to make output sound like GLaDOS
		mel = tts_output['mel_post'].to(device)
		audio = vocoder(mel)
		print("\033[1;94mINFO:\033[;97m The audio sample took " + str(round((time.time() - old_time) * 1000)) + " ms to generate.")

		# Normalize audio to fit in wav-file
		audio = audio.squeeze()
		audio = audio * 32768.0
		audio = audio.cpu().numpy().astype('int16')
		# if(key):
		# 	output_file = ('audio/GLaDOS-tts-temp-output-'+key+'.wav')
		# else:
		# 	output_file = ('audio/GLaDOS-tts-temp-output.wav')

		print("1")
		# Write audio file to disk
		# 22,05 kHz sample rate 
		write(output_file, 22050, audio)
		print("2")


# If the script is run directly, assume remote engine
if __name__ == "__main__":
	
	# Remote Engine Veritables
	PORT = 8124
	CACHE = True

	from flask import Flask, request, send_file
	import urllib.parse
	import shutil
	
	print("\033[1;94mINFO:\033[;97m Initializing TTS Server...")
	
	app = Flask(__name__)
	@app.post('/synthesize', defaults={'text': ''})
	@app.route('/synthesize/<path:text>')
	def synthesize(text):
		# when post then parse the text
		if request.method == 'POST':
			text = request.get_json()['text']
		
		if(text == ''): return 'No input'

		if request.method == 'POST':
			line = text
		else:
			line = urllib.parse.unquote(request.url[request.url.find('synthesize/')+11:])

		fileKey = hashlib.md5(line.upper().encode('utf-8')).hexdigest()
		filename = "GLaDOS-tts-" + fileKey + ".wav"
		file = os.getcwd() + '/audio/' + filename
		
		if(os.path.exists(file)):
			print("\033[1;94mINFO:\033[;97m Using cached file.")
			return send_file(file, mimetype='audio/wav')
		
		glados_tts(line, file)
		return send_file(file)
	
	cli = sys.modules['flask.cli']
	cli.show_server_banner = lambda *x: None
	app.run(host="0.0.0.0", port=PORT)