import torch

from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer

_cleaner = None
_tokenizer = None


def prepare_text(text: str)->str:
    # Build the cleaner once: it loads the 65 MB phonemizer checkpoint, which
    # used to happen on every sentence.
    global _cleaner, _tokenizer
    if _cleaner is None:
        _cleaner = Cleaner('english_cleaners', True, 'en-us')
        _tokenizer = Tokenizer()
    if not ((text[-1] == '.') or (text[-1] == '?') or (text[-1] == '!')):
        text = text + '.'
    return torch.as_tensor(_tokenizer(_cleaner(text)), dtype=torch.long, device='cpu').unsqueeze(0)
