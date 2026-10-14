"""Rule-based Hawaiian G2P that emits the same IPA alphabet the GLaDOS
phonemizer (en_us_cmudict_ipa_forward) produces, so student and teacher models
need no new symbols.

Two registers, chosen per word from the spelling:

* strict  - the word carries an ʻokina or kahakō, so the writer cared about
            Hawaiian orthography: glottal stops, long vowels, native stress.
* local   - unmarked spelling ("Kalakaua"): Hawaiʻi English. Pure vowels and
            Hawaiian stress (never mainland "KAL-a-kow-a"), but no glottal
            stops and no vowel length, since the spelling doesn't say where
            they go.

Stress follows Pukui & Elbert / Schütz: long vowels and diphthongs are heavy
syllables; feet are built right to left as (H) or (L L), the head of each foot
is stressed, the rightmost foot carries primary stress, and a stray light
syllable stays unstressed. That gets Kalākaua, Kalanianaʻole, Waikīkī,
Kāneʻohe and friends right without a dictionary.

The vowel/consonant realisations live in PHONES so they can be tuned by ear.
"""
import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, List, Optional

OKINA = 'ʻ'
# Characters people type instead of the ʻokina. Only converted when they sit
# where an ʻokina can (word-initial or between vowels, before a vowel).
OKINA_LOOKALIKES = "'‘’`ʼʽ´"
MACRONS = {'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u'}
CONSONANTS = set('hklmnpw') | {OKINA}
VOWELS = set('aeiou')
DIPHTHONGS = {'ai', 'ae', 'ao', 'au', 'ei', 'eu', 'oi', 'ou', 'iu'}

# Realisations, restricted to symbols the CMUdict phonemizer actually emits:
#   ɪ ə ː n t l s r k d i m ɝ z ɛ ʊ ɑ æ b p o a g f ʃ e ɔ v ŋ u h w ʌ ʒ j θ ð
# It emits no stress marks and no ʔ, so the GLaDOS model never learned them.
# Stress is therefore carried the way the dictionary carries it, by vowel
# length and quality (honolulu -> hɑːnəluːluː), and the ʻokina the way the
# dictionary writes a glottal onset, as a word break (uh-oh -> ʌ oʊ).
# Keys: vowel -> (primary stress, other stressed, unstressed); long vowels;
# diphthongs; okina.
PHONES = {
    'strict': {
        'vowel': {'a': ('ɑː', 'ɑ', 'ə'), 'e': ('ɛ', 'ɛ', 'ɛ'), 'i': ('iː', 'i', 'i'),
                  'o': ('oʊ', 'oʊ', 'oʊ'), 'u': ('uː', 'u', 'u')},
        'long': {'a': 'ɑː', 'e': 'eɪ', 'i': 'iː', 'o': 'oʊ', 'u': 'uː'},
        'diphthong': {'ai': 'aɪ', 'ae': 'aɪ', 'ao': 'aʊ', 'au': 'aʊ', 'ei': 'eɪ',
                      'eu': 'ɛu', 'oi': 'ɔɪ', 'ou': 'oʊ', 'iu': 'ju'},
        'okina': ' ',
    },
    # Hawaiʻi English: full vowels everywhere (no mainland schwa), Hawaiian
    # stress, no glottal stops or vowel length.
    'local': {
        'vowel': {'a': ('ɑː', 'ɑ', 'ɑ'), 'e': ('ɛ', 'ɛ', 'ɛ'), 'i': ('iː', 'i', 'i'),
                  'o': ('oʊ', 'oʊ', 'oʊ'), 'u': ('uː', 'u', 'u')},
        'long': {'a': 'ɑ', 'e': 'ɛ', 'i': 'i', 'o': 'oʊ', 'u': 'u'},
        'diphthong': {'ai': 'aɪ', 'ae': 'aɪ', 'ao': 'aʊ', 'au': 'aʊ', 'ei': 'eɪ',
                      'eu': 'ɛu', 'oi': 'ɔɪ', 'ou': 'oʊ', 'iu': 'ju'},
        'okina': '',
    },
}
PRIMARY, SECONDARY = 'ˈ', 'ˌ'          # internal only; never emitted


@dataclass
class Syllable:
    onset: str          # '', a consonant, or OKINA
    nucleus: str        # one vowel or a diphthong, macrons stripped
    long: bool          # long vowel (kahakō) or long-first diphthong
    stress: str = ''    # '', PRIMARY or SECONDARY

    @property
    def heavy(self) -> bool:
        return self.long or len(self.nucleus) == 2


def normalize_okina(text: str) -> str:
    """Canonicalise ʻokina lookalikes and decomposed macrons in running text."""
    text = unicodedata.normalize('NFC', text)
    pattern = r"(?:(?<=^)|(?<=[\s\-(\"])|(?<=[aeiouāēīōūAEIOUĀĒĪŌŪ]))[" + re.escape(OKINA_LOOKALIKES) + r"](?=[aeiouāēīōūAEIOUĀĒĪŌŪ])"
    return re.sub(pattern, OKINA, text)


_UNMACRON = {**MACRONS, **{k.upper(): v.upper() for k, v in MACRONS.items()}}


def strip_marks(word: str) -> str:
    return ''.join(_UNMACRON.get(c, c) for c in word).replace(OKINA, '')


def is_marked(word: str) -> bool:
    return OKINA in word or any(c.lower() in MACRONS for c in word)


def syllabify(word: str) -> Optional[List[Syllable]]:
    """Split a Hawaiian word into (C)V syllables with diphthongs merged.
    Returns None if the word isn't a possible Hawaiian word."""
    w = unicodedata.normalize('NFC', word.lower())
    sylls: List[Syllable] = []
    i, onset = 0, ''
    while i < len(w):
        c = w[i]
        if c in CONSONANTS:
            if onset:                     # two consonants in a row
                return None
            onset = c
            i += 1
            continue
        base = MACRONS.get(c, c)
        if base not in VOWELS:
            return None
        long = c in MACRONS
        nxt = w[i + 1] if i + 1 < len(w) else ''
        # Short diphthong: two vowels, no consonant between, second one short.
        # A long first vowel also makes a (long) diphthong: āi, āu, ēi, ōu.
        if nxt in VOWELS and base + nxt in DIPHTHONGS:
            sylls.append(Syllable(onset, base + nxt, long))
            i += 2
        else:
            sylls.append(Syllable(onset, base, long))
            i += 1
        onset = ''
    if onset or not sylls:                # word-final consonant
        return None
    return sylls


def assign_stress(sylls: List[Syllable]) -> None:
    feet = []
    i = len(sylls) - 1
    while i >= 0:
        if sylls[i].heavy:
            feet.append(i)
            i -= 1
        elif i >= 1 and not sylls[i - 1].heavy:
            feet.append(i - 1)
            i -= 2
        else:
            i -= 1                         # stray light syllable
    for n, head in enumerate(feet):
        sylls[head].stress = PRIMARY if n == 0 else SECONDARY


def to_ipa(word: str, register: Optional[str] = None, show_stress: bool = False) -> Optional[str]:
    """IPA for one Hawaiian word in the phonemizer's alphabet. None if the word
    isn't Hawaiian-shaped. show_stress=True adds ˈ/ˌ for debugging only; the
    model has never seen those symbols."""
    word = normalize_okina(word)
    if register is None:
        register = 'strict' if is_marked(word) else 'local'
    sylls = syllabify(word)
    if sylls is None:
        return None
    assign_stress(sylls)
    ph = PHONES[register]
    out = []
    prev_vowel = ''
    for n, s in enumerate(sylls):
        if s.onset == OKINA:
            if n:                          # word-initial ʻokina needs no break
                out.append(ph['okina'])
        elif s.onset == 'w':
            # strict: [v] after i/e, [w] after u/o, either after a (we pick w)
            out.append('v' if register == 'strict' and prev_vowel in ('i', 'e') else 'w')
        else:
            out.append(s.onset)
        if show_stress:
            out.append(s.stress)
        if len(s.nucleus) == 2:
            out.append(ph['diphthong'][s.nucleus])
        elif s.long:
            out.append(ph['long'][s.nucleus])
        else:
            primary, other, unstressed = ph['vowel'][s.nucleus]
            out.append(primary if s.stress == PRIMARY else other if s.stress else unstressed)
        prev_vowel = s.nucleus[-1]
    return ''.join(out).strip()


class HawaiianClassifier:
    """Decides, per token, whether to use the Hawaiian G2P and in which register.

    marked spelling                                  -> strict
    in the lexicon, and either not an English word
      or known from a marked spelling / the seed list -> local
    otherwise                                        -> English (phonemizer)

    Shape alone is not enough: Pewaukee, Kalamazoo and Dutch "Nieuwe" are all
    CV-shaped. The lexicon (harvested from Hawaiʻi street names) is what says a
    word is Hawaiian. min_unknown_syllables re-enables a shape-only fallback
    for text known to be about Hawaiʻi.

    `lexicon` entries may carry marks (Kalanianaʻole). An unmarked token whose
    marked spelling is known is still read in the local register, but with the
    syllable breaks and stress of the marked form: a local speaker who sees
    "Kalanianaole" doesn't turn the a-o into a diphthong, they just soften the
    glottal stop.
    """

    def __init__(self, lexicon: Iterable[str] = (), english_vocab: Iterable[str] = (),
                 min_unknown_syllables: Optional[int] = None, trusted: Iterable[str] = ()):
        self.marked_forms = {}
        self.trusted = {strip_marks(normalize_okina(w)).lower() for w in trusted}
        for w in lexicon:
            w = normalize_okina(w)
            bare = strip_marks(w).lower()
            if is_marked(w):
                self.trusted.add(bare)
            if is_marked(w) or bare not in self.marked_forms:
                self.marked_forms[bare] = w.lower()
        self.english = {w.lower() for w in english_vocab}
        self.min_unknown_syllables = min_unknown_syllables

    def register(self, token: str) -> Optional[str]:
        token = normalize_okina(token)
        if syllabify(token) is None:
            return None
        if is_marked(token):
            return 'strict'
        bare = token.lower()
        if bare in self.marked_forms and (bare in self.trusted or bare not in self.english):
            return 'local'
        if bare in self.english or not self.min_unknown_syllables:
            return None
        return 'local' if len(syllabify(token)) >= self.min_unknown_syllables else None

    def ipa(self, token: str) -> Optional[str]:
        reg = self.register(token)
        if reg is None:
            return None
        token = normalize_okina(token)
        if reg == 'local':
            token = self.marked_forms.get(token.lower(), token)
        return to_ipa(token, reg)


def load_lexicon(path) -> List[str]:
    words = []
    with open(path, encoding='utf-8') as f:
        for line in f:
            line = line.split('#', 1)[0].strip()
            if line:
                words.append(line.split('\t')[0])
    return words
