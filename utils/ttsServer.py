import sys
import os

import torch
from scipy.io.wavfile import write
import time
import hashlib

from flask import Flask, request, send_file
import urllib.parse
import shutil
import io
from scipy.io.wavfile import write

class TTSServer:
    def __init__(self, engine, port = 5000):
        self.engine = engine
        self.port = port

    def start(self):
        from flask import Flask, request
        app = Flask(__name__)

        @app.route('/generate', methods=['POST'])
        def generate():
            text = request.form['text']
            result = self.engine.generate(text)
            return send_file(result.asMemoryFile(), mimetype='audio/wav', as_attachment=True, download_name='audio.wav')

        
        cli = sys.modules['flask.cli']
        cli.show_server_banner = lambda *x: None
        app.run(host="0.0.0.0", port=self.port)



# engine = TTSEngine.TTSEngine()
# engine.warmup()

# while(1):
#     result = engine.generate("Hello World you are a nice person")
#     result.save("test.wav")
#     tools.playAudio("test.wav")
#     os.unlink("test.wav")

#     result = engine.generate("You are also quite intelligent")
#     result.save("test.wav")
#     tools.playAudio("test.wav")
#     os.unlink("test.wav")