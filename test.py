import os

audioEnabled = str(os.environ.get('AUDIO_ENABLED'))
print("Audio: " + audioEnabled)

if audioEnabled == 'None' or audioEnabled == 'false':
    print('Audio is disabled')
else:
    print('Audio is enabled')