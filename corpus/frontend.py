"""Text -> phoneme frontend shared by the teacher render and the students.

normalize (nav-aware) -> split into Hawaiian / English runs -> Hawaiian G2P for
the former, the repo's DeepPhonemizer for the latter -> filter to the model's
symbol set. The output is exactly what utils.text.tokenizer.Tokenizer eats, so
the teacher renders from it directly and the students train on it verbatim.
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from unidecode import unidecode

from corpus.hawaiian import HawaiianClassifier, OKINA, load_lexicon, normalize_okina
from corpus.normalize import normalize

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from utils.text.symbols import phonemes_set  # noqa: E402

DATA = Path(__file__).parent / 'data'
_TOKEN = re.compile(r"[A-Za-zÀ-ɏ" + OKINA + r"]+")


def load_phonemizer(checkpoint: Path):
    import torch
    import dp.preprocessing.text as dpt
    from dp.phonemizer import Phonemizer
    # torch>=2.6 defaults to weights_only=True; allowlist just the classes this
    # checkpoint pickles instead of turning safety off.
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([dpt.Preprocessor, dpt.LanguageTokenizer, dpt.SequenceTokenizer])
    return Phonemizer.from_checkpoint(str(checkpoint))


def load_overrides(path: Path) -> Dict[str, str]:
    """word<TAB>ipa, in the phonemizer's own alphabet (see corpus.hawaiian)."""
    out = {}
    if path.exists():
        for line in open(path, encoding='utf-8'):
            line = line.split('#', 1)[0].rstrip()
            if '\t' in line:
                word, ipa = line.split('\t', 1)
                out[word.strip().lower()] = ipa.strip()
    return out


class Frontend:
    def __init__(self, phonemizer_ckpt: Path = REPO / 'models' / 'en_us_cmudict_ipa_forward.pt',
                 lexicons: Sequence[Path] = (), phonemizer=None):
        """lexicons: Hawaiian word lists (e.g. each Hawaiian region's
        hawaiian_lexicon.tsv); the bundled seed list is always included."""
        self.phonemizer = phonemizer or load_phonemizer(phonemizer_ckpt)
        english = self.phonemizer.lang_phoneme_dict.get('en_us', {}).keys()
        seed = load_lexicon(DATA / 'hawaiian_seed.txt')
        words: List[str] = list(seed)
        for path in lexicons:
            if Path(path).exists():
                words += load_lexicon(path)
        self.hawaiian = HawaiianClassifier(words, english, trusted=seed)
        self.overrides = load_overrides(DATA / 'pronunciations.tsv')

    def segments(self, text: str):
        """Yield (is_hawaiian, chunk) runs over normalized text."""
        pos, buf = 0, []
        for m in _TOKEN.finditer(text):
            ipa = self.overrides.get(m.group(0).lower()) or self.hawaiian.ipa(m.group(0))
            if ipa is None:
                continue
            buf.append((False, text[pos:m.start()]))
            buf.append((True, ipa))
            pos = m.end()
        buf.append((False, text[pos:]))
        return [(h, c) for h, c in buf if c]

    def __call__(self, text: str, nav: bool = False) -> str:
        return self.batch([text], nav=nav)[0]

    def batch(self, texts: Sequence[str], nav: bool = False) -> List[str]:
        spoken = [normalize(normalize_okina(t), nav=nav) for t in texts]
        segs = [self.segments(t) for t in spoken]
        english = [unidecode(c).strip() for s in segs for h, c in s if not h]
        phon = iter(self.phonemizer(english, lang='en_us', batch_size=64) if english else [])
        out = []
        for s in segs:
            parts = []
            for is_haw, chunk in s:
                if is_haw:
                    parts.append(chunk)
                else:
                    # the phonemizer trims edges; put the word breaks back
                    ipa = next(phon)
                    lead = ' ' if chunk[:1].isspace() else ''
                    trail = ' ' if chunk[-1:].isspace() else ''
                    parts.append(lead + ipa + trail)
            ipa = ''.join(c for c in ''.join(parts) if c in phonemes_set)
            ipa = re.sub(r'\s+', ' ', ipa).strip()
            ipa = re.sub(r'\s+([,.;:!?])', r'\1', ipa)
            if ipa and ipa[-1] not in '.?!':
                ipa += '.'
            out.append(ipa)
        return out

    def spoken(self, text: str, nav: bool = False) -> str:
        return normalize(normalize_okina(text), nav=nav)
