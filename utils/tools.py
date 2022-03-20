import torch

from utils.cleaners import Cleaner
from utils.tokenizer import Tokenizer

def prepare_text(text: str)->str:
    if not ((text[-1] == '.') or (text[-1] == '?') or (text[-1] == '!')):
        text = text + '.'
    cleaner = Cleaner('english_cleaners', True, 'en-us')
    tokenizer = Tokenizer()
    return torch.as_tensor(tokenizer(cleaner(text)), dtype=torch.int, device='cpu').unsqueeze(0)
	
# Check Local Cache
def cleanTTSFile(line):
	filename = "GLaDOS-tts-"+line.replace(" ", "-")
	filename = filename.replace("!", "")
	filename = filename.replace("°c", "degrees celcius")
	filename = filename.replace(",", "")+".wav"

	return filename