import os
import re
from typing import Dict, Any

from unidecode import unidecode

from utils.text import hawaiian
from utils.text.numbers import normalize_numbers
from utils.text.symbols import phonemes_set

from dp.phonemizer import Phonemizer
from dp.preprocessing.text import LanguageTokenizer, Preprocessor, SequenceTokenizer
import torch

# torch>=2.6 defaults torch.load to weights_only=True; allow DeepPhonemizer's
# tokenizer classes so its checkpoint still loads.
if hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([Preprocessor, SequenceTokenizer, LanguageTokenizer, set])

# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')
# Candidate words for the Hawaiian rules / lexicon: letters, kahakō and ʻokina variants.
_word_re = re.compile(r"[A-Za-zĀĒĪŌŪāēīōūʻ‘’'`]+")
_quotes = "‘’'`"

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort')
]]


def load_lexicon(path: str) -> Dict[str, str]:
    """Optional pronunciation overrides: one `word<TAB>phonemes` per line, # comments."""
    lex = {}
    if path and os.path.exists(path):
        for line in open(path, encoding='utf-8'):
            line = line.split('#', 1)[0].strip()
            if '\t' in line:
                word, ph = line.split('\t', 1)
                lex[hawaiian.normalize(word.strip())] = ph.strip()
    return lex


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def no_cleaners(text):
    return text


def english_cleaners(text):
    text = unidecode(text)
    text = normalize_numbers(text)
    text = expand_abbreviations(text)
    return text


class Cleaner:

    def __init__(self,
                 cleaner_name: str,
                 use_phonemes: bool,
                 lang: str,
                 hawaiian_names: bool = True,
                 lexicon_path: str = 'lexicon.txt') -> None:
        if cleaner_name == 'english_cleaners':
            self.clean_func = english_cleaners
        elif cleaner_name == 'no_cleaners':
            self.clean_func = no_cleaners
        else:
            raise ValueError(f'Cleaner not supported: {cleaner_name}! '
                             f'Currently supported: [\'english_cleaners\', \'no_cleaners\']')
        self.use_phonemes = use_phonemes
        self.lang = lang
        self.hawaiian_names = hawaiian_names
        self.lexicon = load_lexicon(lexicon_path)
        if use_phonemes:
            self.phonemize = Phonemizer.from_checkpoint('models/en_us_cmudict_ipa_forward.pt')
            self.english_dict = self.phonemize.lang_phoneme_dict['en_us']

    def _special(self, word: str):
        """Phonemes for a word from the lexicon or the Hawaiian rules, else None."""
        key = hawaiian.normalize(word)
        if key in self.lexicon:
            return self.lexicon[key]
        if self.hawaiian_names and hawaiian.is_hawaiian(word, self.english_dict):
            return hawaiian.to_phonemes(word)
        return None

    def __call__(self, text: str) -> str:
        if not self.use_phonemes:
            return collapse_whitespace(self.clean_func(text)).strip()
        # Split out words with a special pronunciation; the rest goes through the
        # English cleaners and phonemizer, which work word by word anyway.
        parts, last = [], 0
        for m in _word_re.finditer(text):
            word = m.group(0)
            lead = len(word) - len(word.lstrip(_quotes))
            core = word.strip(_quotes)
            ph = self._special(core) if core else None
            if ph is None:
                continue
            start = m.start() + lead
            parts.append(self._english(text[last:start]))
            parts.append(ph)
            last = start + len(core)
        parts.append(self._english(text[last:]))
        text = ''.join(parts)
        text = ''.join([p for p in text if p in phonemes_set])
        text = collapse_whitespace(text)
        text = text.strip()
        return text

    def _english(self, text: str) -> str:
        if not text:
            return ''
        return self.phonemize(self.clean_func(text), lang='en_us')

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'Cleaner':
        return Cleaner(
            cleaner_name=config['preprocessing']['cleaner_name'],
            use_phonemes=config['preprocessing']['use_phonemes'],
            lang=config['preprocessing']['language']
        )
