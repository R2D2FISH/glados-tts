"""Rule-based pronunciation for Hawaiian words (street and place names).

Hawaiian spelling is almost perfectly phonemic, so a few rules beat the English
phonemizer, which guesses with English spelling habits (Likelike -> "like-like")
and drops the ʻokina. Output uses the same IPA symbols as the English phonemizer;
the TTS model was trained without stress marks, so stress is expressed through
vowel quality and length instead (stressed a -> ɑ, unstressed a -> ə).

Rules:
  - syllables are (C)V, with long vowels (kahakō) and the diphthongs
    ai ae ao au ei eu oi ou iu as single heavy (two-mora) nuclei
  - stress: feet of two morae built from the end of the word; the rightmost
    foot carries the main stress (Kamehameha -> ka.ˌme.ha.ˈme.ha)
  - w is pronounced v after i and e (ʻEwa -> "eva")
  - a glide (y / w) is inserted between vowels in separate syllables
  - the ʻokina is a glottal stop
"""
import re
import unicodedata

OKINA = 'ʔ'
_OKINA_CHARS = "ʻ‘’'`"  # U+02BB is correct; maps often use quotes or apostrophes
_MACRON = {'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u'}
_VOWELS = 'aeiou'
_CONS = 'hklmnpw' + OKINA
_DIPHTHONGS = ('ai', 'ae', 'ao', 'au', 'ei', 'eu', 'oi', 'ou', 'iu')
_WORD_RE = re.compile(r'^(?:[hklmnpwʔ]?(?:[aeiou]|[āēīōū]))+$')

# vowel -> (stressed, unstressed) phones
_SHORT = {'a': ('ɑ', 'ə'), 'e': ('ɛ', 'ɛ'), 'i': ('iː', 'i'), 'o': ('oʊ', 'oʊ'), 'u': ('uː', 'u')}
_LONG = {'a': 'ɑː', 'e': 'eɪ', 'i': 'iː', 'o': 'oʊ', 'u': 'uː'}
_DIPH = {'ai': 'aɪ', 'ae': 'aɪ', 'ao': 'aʊ', 'au': 'aʊ', 'ei': 'eɪ', 'eu': 'ɛu', 'oi': 'ɔɪ', 'ou': 'oʊ', 'iu': 'iu'}

# Common Hawaiian words that the English dictionary already lists (with
# anglicized pronunciations); always read these as Hawaiian.
FORCE_HAWAIIAN = {
    'hawaii', 'kamehameha', 'waikiki', 'kailua', 'honolulu', 'kona', 'maui', 'kauai', 'oahu', 'molokai',
    'hilo', 'lahaina', 'kihei', 'kaneohe', 'manoa', 'makaha', 'waianae', 'wahiawa', 'haleiwa', 'kapalua',
    'kaanapali', 'kapaa', 'lihue', 'hana', 'kahului', 'wailuku', 'aloha', 'mahalo', 'pali', 'punahou',
    'nuuanu', 'moana', 'kalakaua', 'kuhio', 'kapiolani', 'liliuokalani', 'kaimuki', 'kahala', 'aina',
    'ewa', 'kakaako', 'kalihi', 'moiliili', 'makiki', 'hauula', 'laie', 'kahuku', 'pupukea', 'ala',
}


def normalize(word):
    """Lowercase, unify ʻokina variants, keep kahakō."""
    w = unicodedata.normalize('NFC', word).lower()
    return ''.join(OKINA if c in _OKINA_CHARS else c for c in w)


def is_hawaiian(word, english_dict=None):
    """True if `word` should be read with Hawaiian rules."""
    w = normalize(word)
    if not _WORD_RE.match(w):
        return False
    if any(c in _MACRON for c in w) or (OKINA in w and not w.startswith(OKINA) and not w.endswith(OKINA)):
        return True
    plain = w.replace(OKINA, '')
    if plain in FORCE_HAWAIIAN:
        return True
    if len(plain) < 3:
        return False  # a, i, no, he, me, ...
    return english_dict is not None and plain not in english_dict


def _syllables(w):
    """-> list of (onset, nucleus, long?) with nucleus as plain letters."""
    syls, i = [], 0
    while i < len(w):
        onset = ''
        if w[i] in _CONS:
            onset = w[i]
            i += 1
        c = w[i]
        if c in _MACRON:
            syls.append((onset, _MACRON[c], True))
            i += 1
        elif i + 1 < len(w) and w[i:i + 2] in _DIPHTHONGS:
            syls.append((onset, w[i:i + 2], True))
            i += 2
        else:
            syls.append((onset, c, False))
            i += 1
    return syls


def _stress(syls):
    """-> list of 0 (unstressed), 1 (secondary), 2 (primary)."""
    s = [0] * len(syls)
    i = len(syls) - 1
    first = True
    while i >= 0:
        heavy = syls[i][2]
        if heavy:
            s[i] = 2 if first else 1
            i -= 1
        elif i > 0 and not syls[i - 1][2]:
            s[i - 1] = 2 if first else 1
            i -= 2
        else:
            i -= 1  # stray light syllable: unstressed
            continue
        first = False
    return s


def to_phonemes(word):
    w = normalize(word)
    syls = _syllables(w)
    stress = _stress(syls)
    out = []
    prev_nucleus = ''
    for k, ((onset, nuc, heavy), st) in enumerate(zip(syls, stress)):
        if onset == OKINA:
            if k > 0:
                out.append(OKINA)  # word-initial ʻokina is inaudible after a pause
        elif onset == 'w':
            out.append('v' if prev_nucleus[-1:] in ('i', 'e') else 'w')
        elif onset:
            out.append(onset)
        elif prev_nucleus:
            # vowel-vowel across syllables: glide, as in natural Hawaiian speech
            if prev_nucleus[-1] == 'i' and nuc[0] != 'i':
                out.append('j')
            elif prev_nucleus[-1] == 'u' and nuc[0] != 'u':
                out.append('w')
        if len(nuc) == 2:
            out.append(_DIPH[nuc])
        elif heavy:
            out.append(_LONG[nuc])
        else:
            out.append(_SHORT[nuc][0 if st else 1])
        prev_nucleus = nuc
    return ''.join(out)
