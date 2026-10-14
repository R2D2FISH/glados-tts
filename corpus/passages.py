"""Non-route item sources: curated files, generated numbers, Hawaiian street
names (exhaustive), carrier sentences for hard place names, and plain-text
prose ingest (Gutenberg books, your own writing, LLM output)."""
import random
import re
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from corpus.hawaiian import HawaiianClassifier, is_marked, normalize_okina, strip_marks, syllabify

DATA = Path(__file__).parent / 'data'


def read_curated(path: Path) -> Iterator[Dict]:
    """Plain text, one item per line. Header comments set defaults:
    '# category: prose' and '# tags: a, b'. Other '#' lines are comments."""
    category, tags = path.stem, []
    with open(path, encoding='utf-8') as f:
        for n, line in enumerate(f):
            line = line.rstrip('\n')
            if line.startswith('#'):
                m = re.match(r'#\s*(category|tags):\s*(.*)', line)
                if m and m.group(1) == 'category':
                    category = m.group(2).strip()
                elif m:
                    tags = [t.strip() for t in m.group(2).split(',') if t.strip()]
                continue
            if line.strip():
                yield {'text': line.strip(), 'category': category, 'tags': list(tags),
                       'source': f'curated:{path.name}:{n + 1}'}


def curated_items(extra_dirs: Iterable[Path] = ()) -> List[Dict]:
    items = []
    for d in [DATA / 'curated', *extra_dirs]:
        for p in sorted(Path(d).glob('*.txt')):
            if p.name == 'place_names.txt':
                continue
            items.extend(read_curated(p))
    return items


# ---------------------------------------------------------------------------
# hard US place names in carrier sentences

CARRIERS = {
    'city': ['Continue toward {x}.', 'Take the exit for {x}.', 'Welcome to {x}.', 'Next stop, {x}.',
             '{x} is twelve miles ahead.', 'Your destination is in {x}.', 'Have you ever been to {x}?'],
    'street': ['Turn right onto {x}.', 'Turn left onto {x}.', 'Continue on {x}.', 'Keep left to stay on {x}.',
               'Your destination is on {x}.'],
    'road': ['Merge onto the {x}.', 'Take the {x} toward downtown.', 'Traffic is heavy on the {x}.'],
    'water': ['Cross the {x}.', 'The {x} is on your left.', 'Continue along the {x}.'],
    'county': ['You are now entering {x}.', 'Welcome to {x}.'],
    'landmark': ['{x} is on your right.', 'Head toward {x}.'],
}


def place_name_items(path: Path = DATA / 'curated' / 'place_names.txt', per_name: int = 2, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    out = []
    for line in open(path, encoding='utf-8'):
        if line.startswith('#') or not line.strip():
            continue
        parts = line.rstrip('\n').split('\t')
        name, kind = parts[0], (parts[1] if len(parts) > 1 else 'city')
        for tmpl in rng.sample(CARRIERS[kind], min(per_name, len(CARRIERS[kind]))):
            out.append({'text': tmpl.format(x=name), 'category': 'place_names', 'tags': ['placename', kind],
                        'source': f'place_names:{name}'})
    return out


# ---------------------------------------------------------------------------
# numbers: raw text, normalized later by the nav normalizer

def number_items(n: int = 3000, seed: int = 0) -> List[Dict]:
    rng = random.Random(seed)
    streets = ['Main St', 'Oak Ave', 'Elm Dr', 'Park Blvd', 'Lake Rd', 'Hill Ln', 'Pine Ct', 'Cedar Pl',
               'Kalakaua Ave', 'Kapiolani Blvd', 'Broadway', 'Market St', 'Mission St', 'Colfax Ave']
    dirs = ['N', 'S', 'E', 'W', 'NE', 'NW', 'SE', 'SW']
    gens = [
        lambda: f'{rng.randint(1, 12)}:{rng.randint(0, 59):02d} {rng.choice(["AM", "PM", "a.m.", "p.m.", "am", "pm"])}',
        lambda: f'Arriving at {rng.randint(1, 12)}:{rng.randint(0, 59):02d}.',
        lambda: f'ETA {rng.randint(1, 12)}:{rng.choice(["00", "05", "15", "30", "45", "52"])}, {rng.randint(2, 95)} min remaining.',
        lambda: f'{rng.randint(1, 99999)} {rng.choice(streets)}',
        lambda: f'{rng.randint(1, 9999)} {rng.choice(dirs)} {rng.choice(streets)}',
        lambda: f'{rng.randint(1, 20) * 100} {rng.choice(["South", "North", "East", "West"])} {rng.randint(1, 20) * 100} {rng.choice(["East", "West", "North", "South"])}',
        lambda: f'{rng.randint(10, 250)}-{rng.randint(1, 99):02d} {rng.randint(1, 220)}th St',
        lambda: f'Take exit {rng.randint(1, 450)}{rng.choice(["", "", "A", "B", "C"])}.',
        lambda: f'In {rng.choice(["0.1", "0.2", "0.3", "0.5", "0.8", "1.2", "2.5", "1/2", "1/4", "3/4"])} mi, turn {rng.choice(["left", "right"])}.',
        lambda: f'In {rng.choice([100, 200, 300, 400, 500, 750, 1000, 1500])} ft, keep {rng.choice(["left", "right"])}.',
        lambda: f'Continue for {rng.randint(2, 280)} miles.',
        lambda: f'Speed limit {rng.choice([15, 25, 30, 35, 45, 55, 65, 70, 75])} mph.',
        lambda: f'Take I-{rng.choice([5, 10, 15, 25, 35, 40, 45, 70, 75, 80, 90, 95, 405, 280, 680, 287, 495])} {rng.choice(dirs[:4])}.',
        lambda: f'Merge onto US-{rng.randint(1, 101)} {rng.choice(dirs[:4])}.',
        lambda: f'Take {rng.choice(["SR", "CR", "FM", "HI", "CA", "TX", "NY"])} {rng.randint(1, 3000)} for {rng.randint(1, 40)} mi.',
        lambda: f'Merge onto H-{rng.choice([1, 2, 3, 201])} {rng.choice(dirs[:4])}.',
        lambda: f'Honolulu, HI 968{rng.randint(10, 99)}',
        lambda: f'The total is ${rng.randint(1, 999)}.{rng.randint(0, 99):02d}.',
        lambda: f'{rng.randint(1, 100)}% complete.',
        lambda: f'Call {rng.randint(200, 999)}-555-{rng.randint(100, 199):04d}.',
        lambda: f'It is {rng.randint(-20, 110)} degrees outside.',
        lambda: f'The year {rng.randint(1700, 2099)}.',
        lambda: f'{rng.randint(1, 12)}/{rng.randint(1, 28)}/{rng.randint(1990, 2040)}',
        lambda: f'Room {rng.randint(1, 999)}, floor {rng.randint(1, 60)}.',
        lambda: f'{rng.randint(1, 9)}.{rng.randint(0, 99)} km to go.',
    ]
    out = []
    for i in range(n):
        text = rng.choice(gens)()
        out.append({'text': text, 'category': 'numbers', 'tags': ['numbers'], 'source': 'numbers', 'nav': True})
    return out


# ---------------------------------------------------------------------------
# Hawaiian street names, exhaustive

HAW_TEMPLATES_TERSE = ['Turn left onto {x}', 'Turn R onto {x}', 'Continue on {x}', 'In 0.4 mi, turn right onto {x}',
                       'Keep left to stay on {x}', 'Destination is on {x}', 'Head mauka on {x}', 'Head makai on {x}']
HAW_TEMPLATES_VERBAL = ['Turn left onto {x}.', 'Turn right onto {x}.', 'Continue on {x} for a half mile.',
                        'Bear right onto {x}.', 'Your destination is on {x}, on the right.',
                        'Make a U-turn at {x}.', 'Head ʻEwa on {x}.', 'Head toward Diamond Head on {x}.']

_SUFFIX_FULL_TO_ABBR = {'Street': 'St', 'Avenue': 'Ave', 'Boulevard': 'Blvd', 'Road': 'Rd', 'Drive': 'Dr',
                        'Place': 'Pl', 'Highway': 'Hwy', 'Lane': 'Ln', 'Loop': 'Lp', 'Way': 'Way'}


def hawaiian_street_items(names: List[Dict], classifier: HawaiianClassifier, seed: int = 0,
                          per_name: int = 2) -> List[Dict]:
    """Every street name containing a Hawaiian word, in nav carriers, in both
    registers when possible: as harvested, and with marks restored (strict) or
    removed (local) from the lexicon's marked spelling."""
    rng = random.Random(seed)
    out, seen = [], set()
    for row in names:
        name = normalize_okina(row['name'])
        toks = name.split()
        if not any(classifier.register(t) for t in toks):
            continue
        variants = {name}
        restored = ' '.join(classifier.marked_forms.get(t.lower(), t) if not is_marked(t) and classifier.register(t)
                            else t for t in toks)
        restored = ' '.join(w[:1].upper() + w[1:] if w[:1].islower() or (w[:1] == 'ʻ') else w
                            for w in restored.split())
        if restored != name:
            variants.add(_titlecase_haw(restored))
        stripped = strip_marks(name)
        if stripped != name:
            variants.add(stripped)
        for v in variants:
            if v in seen:
                continue
            seen.add(v)
            register = 'strict' if is_marked(v) else 'local'
            for k in range(per_name):
                if rng.random() < 0.5:
                    w = v.split()
                    if len(w) > 1 and w[-1] in _SUFFIX_FULL_TO_ABBR and rng.random() < 0.7:
                        w[-1] = _SUFFIX_FULL_TO_ABBR[w[-1]]
                    text = rng.choice(HAW_TEMPLATES_TERSE).format(x=' '.join(w))
                else:
                    text = rng.choice(HAW_TEMPLATES_VERBAL).format(x=v)
                out.append({'text': text, 'category': 'hawaiian', 'tags': ['hawaiian', register, 'street'],
                            'source': f'osm-hawaii:{row["name"]}', 'nav': True})
    return out


def _titlecase_haw(s: str) -> str:
    def tc(w):
        if w.startswith('ʻ') and len(w) > 1:
            return w[0] + w[1].upper() + w[2:]
        return w[:1].upper() + w[1:]
    return ' '.join(tc(w) for w in s.split())


# ---------------------------------------------------------------------------
# prose ingest

_SENT = re.compile(r'(?<=[.!?])["”\']?\s+(?=["“\']?[A-Z])')


def prose_from_text(path: Path, category: str = 'prose', min_words: int = 4, max_words: int = 45,
                    max_items: Optional[int] = None, seed: int = 0, tags=('prose',)) -> List[Dict]:
    """Split a plain-text file (e.g. a Project Gutenberg book) into sentence
    items. Strips Gutenberg boilerplate, joins wrapped lines, drops lines with
    odd characters. Pairs short neighbouring sentences so passages carry some
    inter-sentence prosody."""
    raw = open(path, encoding='utf-8', errors='ignore').read()
    s = raw.find('*** START OF')
    e = raw.find('*** END OF')
    if s != -1:
        raw = raw[raw.find('\n', s) + 1: e if e != -1 else None]
    paras = [re.sub(r'\s+', ' ', p).strip() for p in re.split(r'\n\s*\n', raw)]
    out = []
    for para in paras:
        if not para or para.isupper() or re.search(r'[_\[\]{}<>|#*=\\@]|https?:', para):
            continue
        sents = [x.strip() for x in _SENT.split(para) if x.strip()]
        buf = ''
        for sent in sents:
            cand = (buf + ' ' + sent).strip() if buf else sent
            nw = len(cand.split())
            if nw < min_words * 2 and len(buf.split()) < max_words // 2:
                buf = cand
                continue
            if min_words <= nw <= max_words:
                out.append(cand)
            buf = ''
        if buf and min_words <= len(buf.split()) <= max_words:
            out.append(buf)
    rng = random.Random(seed)
    if max_items and len(out) > max_items:
        out = rng.sample(out, max_items)
    return [{'text': t, 'category': category, 'tags': list(tags), 'source': f'text:{path.name}'} for t in out]
