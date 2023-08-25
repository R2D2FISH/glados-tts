import torch
from utils.cleaners import Cleaner
from utils.tokenizer import Tokenizer

def prepare_text(text: str) -> torch.Tensor:
    cleaner = Cleaner('english_cleaners', True, 'en-us') # Applying text cleaning
    cleaned_text = cleaner(text)

    tokenizer = Tokenizer() # Tokenization
    tokenized_text = tokenizer(cleaned_text)
    tensor_text = torch.tensor(tokenized_text, dtype=torch.int, device='cpu').unsqueeze(0)
    return tensor_text