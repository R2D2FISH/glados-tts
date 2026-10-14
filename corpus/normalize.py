"""Spoken-form normalizer for US English, with a navigation mode.

Turns app-style text ("In 0.3 mi, turn L onto W 5th St / I-405 N") into what a
person would say ("In point three miles, turn left onto West Fifth Street, I
four oh five North"). The output has no digits or abbreviations left, so the
repo's english_cleaners can't misread "St." as "saint" or "Dr." as "doctor".

nav=True enables the rules that are only safe on navigation text (street-type
suffixes, compass letters, turn letters, house/grid numbers). Prose should use
nav=False so "vitamin E" stays "vitamin E".

Hawaiian words pass through untouched; corpus.frontend sends them to the
Hawaiian G2P.
"""
import re

_ONES = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
         'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
         'seventeen', 'eighteen', 'nineteen']
_TENS = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
_ORD_IRREGULAR = {'one': 'first', 'two': 'second', 'three': 'third', 'five': 'fifth',
                  'eight': 'eighth', 'nine': 'ninth', 'twelve': 'twelfth'}


def cardinal(n: int) -> str:
    if n < 0:
        return 'minus ' + cardinal(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        t, o = divmod(n, 10)
        return _TENS[t] + (' ' + _ONES[o] if o else '')
    if n < 1000:
        h, r = divmod(n, 100)
        return _ONES[h] + ' hundred' + (' ' + cardinal(r) if r else '')
    for div, name in ((10**9, 'billion'), (10**6, 'million'), (1000, 'thousand')):
        if n >= div:
            q, r = divmod(n, div)
            return cardinal(q) + ' ' + name + (' ' + cardinal(r) if r else '')
    raise ValueError(n)


def ordinal(n: int) -> str:
    head, _, last = cardinal(n).rpartition(' ')
    if last in _ORD_IRREGULAR:
        last = _ORD_IRREGULAR[last]
    elif last.endswith('y'):
        last = last[:-1] + 'ieth'
    else:
        last += 'th'
    return (head + ' ' + last).strip()


def digits(s: str, zero: str = 'oh') -> str:
    return ' '.join(_ONES[int(c)] if c != '0' else zero for c in s)


def paired(s: str) -> str:
    """Read a number the way route numbers, house numbers and years are read:
    405 -> four oh five, 280 -> two eighty, 1960 -> nineteen sixty,
    101 -> one oh one, 500 -> five hundred, 2000 -> two thousand,
    2005 -> two thousand five, 12345 -> one twenty three forty five."""
    s = s.lstrip('0') or '0'
    n = int(s)
    if n < 100:
        return cardinal(n)
    if len(s) == 3:
        r = int(s[1:])
        if r == 0:
            return cardinal(n)
        return _ONES[int(s[0])] + ' ' + ('oh ' + _ONES[r] if r < 10 else cardinal(r))
    if len(s) == 4:
        hi, lo = int(s[:2]), int(s[2:])
        if n % 1000 == 0 or (hi % 10 == 0 and lo < 10):
            return cardinal(n)
        if lo == 0:
            return cardinal(hi) + ' hundred'
        return cardinal(hi) + ' ' + ('oh ' + _ONES[lo] if lo < 10 else cardinal(lo))
    if len(s) == 5:
        return _ONES[int(s[0])] + ' ' + paired(s[1:])
    return digits(s)


# ---------------------------------------------------------------------------
# Lexical tables

STREET_SUFFIX = {
    'st': 'Street', 'str': 'Street', 'ave': 'Avenue', 'av': 'Avenue', 'blvd': 'Boulevard',
    'rd': 'Road', 'dr': 'Drive', 'ln': 'Lane', 'ct': 'Court', 'pl': 'Place',
    'pkwy': 'Parkway', 'pky': 'Parkway', 'hwy': 'Highway', 'fwy': 'Freeway',
    'expy': 'Expressway', 'expwy': 'Expressway', 'tpke': 'Turnpike', 'cir': 'Circle',
    'ter': 'Terrace', 'terr': 'Terrace', 'trl': 'Trail', 'sq': 'Square',
    'hts': 'Heights', 'ctr': 'Center', 'cres': 'Crescent', 'aly': 'Alley',
    'xing': 'Crossing', 'byp': 'Bypass', 'cswy': 'Causeway', 'plz': 'Plaza',
    'pt': 'Point', 'lp': 'Loop', 'wy': 'Way', 'rte': 'Route', 'mtwy': 'Motorway',
    'brg': 'Bridge', 'tunl': 'Tunnel', 'hbr': 'Harbor', 'jct': 'Junction',
    'ext': 'Extension', 'svc': 'Service', 'frtg': 'Frontage', 'jr': 'Junior',
    'sr': 'Senior', 'hl': 'Hill', 'mnr': 'Manor', 'vly': 'Valley', 'cv': 'Cove',
    'grv': 'Grove', 'lk': 'Lake', 'pk': 'Park', 'sta': 'Station', 'hwys': 'Highways',
}
PREFIX_TITLE = {'st': 'Saint', 'ste': 'Sainte', 'mt': 'Mount', 'ft': 'Fort', 'dr': 'Doctor',
                'pt': 'Point'}
DIRECTIONS = {'n': 'North', 's': 'South', 'e': 'East', 'w': 'West',
              'ne': 'Northeast', 'nw': 'Northwest', 'se': 'Southeast', 'sw': 'Southwest',
              'nb': 'northbound', 'sb': 'southbound', 'eb': 'eastbound', 'wb': 'westbound'}
STATES = {
    'AL': 'Alabama', 'AK': 'Alaska', 'AZ': 'Arizona', 'AR': 'Arkansas', 'CA': 'California',
    'CO': 'Colorado', 'CT': 'Connecticut', 'DE': 'Delaware', 'FL': 'Florida', 'GA': 'Georgia',
    'HI': 'Hawaii', 'ID': 'Idaho', 'IL': 'Illinois', 'IN': 'Indiana', 'IA': 'Iowa',
    'KS': 'Kansas', 'KY': 'Kentucky', 'LA': 'Louisiana', 'ME': 'Maine', 'MD': 'Maryland',
    'MA': 'Massachusetts', 'MI': 'Michigan', 'MN': 'Minnesota', 'MS': 'Mississippi',
    'MO': 'Missouri', 'MT': 'Montana', 'NE': 'Nebraska', 'NV': 'Nevada', 'NH': 'New Hampshire',
    'NJ': 'New Jersey', 'NM': 'New Mexico', 'NY': 'New York', 'NC': 'North Carolina',
    'ND': 'North Dakota', 'OH': 'Ohio', 'OK': 'Oklahoma', 'OR': 'Oregon', 'PA': 'Pennsylvania',
    'RI': 'Rhode Island', 'SC': 'South Carolina', 'SD': 'South Dakota', 'TN': 'Tennessee',
    'TX': 'Texas', 'UT': 'Utah', 'VT': 'Vermont', 'VA': 'Virginia', 'WA': 'Washington',
    'WV': 'West Virginia', 'WI': 'Wisconsin', 'WY': 'Wyoming', 'DC': 'D\uE000C',
}
# Route-network prefixes -> spoken prefix. State codes read as the state name.
ROUTE_PREFIX = dict(STATES)
ROUTE_PREFIX.update({
    'I': 'I', 'IH': 'I', 'US': 'U\uE000S', 'SR': 'State Route', 'CR': 'County Road',
    'FM': 'F\uE000M', 'RM': 'R\uE000M', 'RR': 'Ranch Road', 'SH': 'State Highway', 'PR': 'Park Road',
    'HWY': 'Highway', 'Hwy': 'Highway', 'RTE': 'Route', 'Rte': 'Route', 'Route': 'Route',
    'ROUTE': 'Route', 'LOOP': 'Loop', 'Loop': 'Loop', 'SPUR': 'Spur', 'Spur': 'Spur',
    'BW': 'Beltway', 'CTH': 'County Highway', 'CH': 'County Highway', 'H': 'H',
})
UNITS = {'mi': ('mile', 'miles'), 'mile': ('mile', 'miles'), 'miles': ('mile', 'miles'),
         'ft': ('foot', 'feet'), 'feet': ('foot', 'feet'), 'yd': ('yard', 'yards'),
         'yds': ('yard', 'yards'), 'km': ('kilometer', 'kilometers'), 'm': ('meter', 'meters'),
         'mph': ('mile per hour', 'miles per hour'), 'min': ('minute', 'minutes'),
         'mins': ('minute', 'minutes'), 'hr': ('hour', 'hours'), 'hrs': ('hour', 'hours'),
         'sec': ('second', 'seconds'), 'secs': ('second', 'seconds')}
FRACTIONS = {'1/2': 'a half', '1/4': 'a quarter', '3/4': 'three quarters', '1/3': 'a third',
             '2/3': 'two thirds', '1/8': 'an eighth'}

_WORD = r"[A-Za-zÀ-ɏʻ'’]+"


# ---------------------------------------------------------------------------
# Rewrite rules

def _route(m):
    pre, num = m.group('pre'), m.group('num')
    letter, card = (m.group('let') or '').upper(), m.group('card') or ''
    if m.group('h') or pre == 'H':                     # Hawaii: H-1, I H-1, I-H1
        spoken = 'H ' + paired(num)
    else:
        spoken = ROUTE_PREFIX[pre] + ' ' + paired(num)
    if letter in ('N', 'S', 'E', 'W') and not card:
        card = letter
    elif letter:
        spoken += ' ' + letter
    if card:
        spoken += ' ' + DIRECTIONS[card.lower()]
    return spoken


_ROUTE_RE = re.compile(
    r"\b(?P<pre>" + '|'.join(sorted(ROUTE_PREFIX, key=len, reverse=True)) + r")"
    r"(?:[-‐‑ ]?(?P<h>H)-?|[-‐‑ ]?)"
    r"(?P<num>\d{1,4})(?P<let>[A-Z])?\b"
    r"(?:\s+(?P<card>NB|SB|EB|WB|N|S|E|W)\b)?")


def _time(m):
    h, mi, ap = int(m.group(1)), m.group(2), (m.group(3) or '').lower().replace('.', '')
    out = cardinal(h)
    if mi == '00':
        out += '' if ap else " o'clock"
    elif mi[0] == '0':
        out += ' oh ' + _ONES[int(mi[1])]
    else:
        out += ' ' + cardinal(int(mi))
    return out + (' ' + '\uE000'.join(ap.upper()) if ap else '')


def _decimal(num: str) -> str:
    whole, frac = num.split('.')
    head = cardinal(int(whole)) + ' ' if whole and whole != '0' else ''
    return head + 'point ' + digits(frac, zero='zero')


def _distance(m):
    num, (sing, plur) = m.group(1), UNITS[m.group(2).lower()]
    if num in FRACTIONS:
        return FRACTIONS[num] + ' ' + sing
    if '.' in num:
        return _decimal(num) + ' ' + plur
    n = int(num.replace(',', ''))
    return cardinal(n) + ' ' + (sing if n == 1 else plur)


_ORDINAL_WORDS = {ordinal(i).split()[-1] for i in range(1, 100)}
_ABBREV_KEYS = set(STREET_SUFFIX) | set(PREFIX_TITLE)


def _prev_is_name(s: str, start: int) -> bool:
    """True if the token before `start` is a capitalized word or a number, i.e.
    an abbreviation at `start` is a street-type suffix rather than a title."""
    before = s[:start].rstrip()
    m = re.search(r"(\S+)$", before)
    if not m:
        return False
    tok = m.group(1)
    if re.search(r"[!?:]$", tok) or (tok.endswith('.') and tok[:-1].lower() not in _ABBREV_KEYS):
        return False                                    # sentence boundary
    return tok[0].isupper() or tok[0].isdigit() or tok.lower() in _ORDINAL_WORDS


def _keep_period(s: str, end: int) -> str:
    return '.' if re.match(r"\s*$|\s+[A-Z]", s[end:]) else ''


def _abbrev(m):
    s, abbr, dot = m.string, m.group(1), m.group(2) or ''
    key = abbr.lower()
    nxt = s[m.end():]
    if _prev_is_name(s, m.start()):
        if key in STREET_SUFFIX:
            return STREET_SUFFIX[key] + (_keep_period(s, m.end()) if dot and key not in ('jr', 'sr') else '')
        return m.group(0)
    if key in PREFIX_TITLE and re.match(r"\s+[A-Z][a-z]", nxt):
        return PREFIX_TITLE[key]
    return m.group(0)


_ABBREV_RE = re.compile(r"\b(" + '|'.join(sorted(set(STREET_SUFFIX) | set(PREFIX_TITLE), key=len, reverse=True))
                        + r")(\.)?(?![A-Za-zʻ'])", re.I)


def _direction(m):
    s = m.string
    tok = m.group(2)
    # a lone letter is a compass direction only next to a name or number
    nxt = s[m.end():]
    nxt_word = re.match(r"\s+(\S+)", nxt)
    if len(tok) == 1 and not ((nxt_word and (nxt_word.group(1)[0].isupper() or nxt_word.group(1)[0].isdigit()
                                             or nxt_word.group(1).lower() in _ORDINAL_WORDS))
                              or _prev_is_name(s, m.start(2))):
        return m.group(0)
    return m.group(1) + DIRECTIONS[tok.lower()]


_MONTHS = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September',
           'October', 'November', 'December']


def _date(m):
    mo, d, y = int(m.group(1)), int(m.group(2)), m.group(3)
    if not (1 <= mo <= 12 and 1 <= d <= 31):
        return m.group(0)
    if len(y) == 2:
        y = ('20' if int(y) < 50 else '19') + y
    return f'{_MONTHS[mo - 1]} {ordinal(d)}, {paired(y)}'


_COMMON_PASSES = [
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4}|\d{2})\b"), _date),
    (re.compile(r"\b(\d{1,2}):(\d{2})\s*([AaPp]\.?[Mm](?:\.(?!\s*$))?)?(?!\d)"), _time),
    (re.compile(r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?|\.\d+|\d/\d)\s*(miles?|mi|ft|feet|yds?|km|mph|mins?|hrs?|secs?|m)\b"), _distance),
    (re.compile(r"\b(\d+)(?:st|nd|rd|th)\b", re.I), lambda m: ordinal(int(m.group(1)))),
    (re.compile(r"\$(\d+)(?:\.(\d\d))?"), lambda m: cardinal(int(m.group(1))) + (' dollar' if m.group(1) == '1' else ' dollars')
     + (' and ' + cardinal(int(m.group(2))) + ' cents' if m.group(2) and m.group(2) != '00' else '')),
    (re.compile(r"(\d+(?:\.\d+)?)\s*%"), lambda m: (_decimal(m.group(1)) if '.' in m.group(1) else cardinal(int(m.group(1)))) + ' percent'),
]
_NAV_PASSES = [
    (re.compile(r"\b([Ee]xits?)\s+(\d{1,4})\s*([A-Za-z])?\b"),
     lambda m: m.group(1) + ' ' + paired(m.group(2)) + (' ' + m.group(3).upper() if m.group(3) else '')),
    # state + ZIP: "Honolulu, HI 96813"
    (re.compile(r"\b(" + '|'.join(STATES) + r")\s+(\d{5})(?:-\d{4})?\b"),
     lambda m: STATES[m.group(1)] + ' ' + digits(m.group(2), zero='zero')),
    (_ROUTE_RE, _route),
    # Queens-style hyphenated house numbers: 37-15 76th St
    (re.compile(r"\b(\d{1,3})-(\d{2,3})(?=\s+(?:\d|[A-Z]))"),
     lambda m: paired(m.group(1)) + ' ' + ('oh ' + paired(m.group(2)) if m.group(2)[0] == '0' else paired(m.group(2)))),
    # terse turn letters: "turn L onto", "keep R"
    (re.compile(r"\b([Tt]urn|[Bb]ear|[Kk]eep|[Ss]light|[Ss]harp)\s+(L|R)\b"),
     lambda m: m.group(1) + ' ' + {'L': 'left', 'R': 'right'}[m.group(2)]),
    (re.compile(r"\bU-?turn\b", re.I), lambda m: 'U-turn'),
    (_ABBREV_RE, _abbrev),
    (re.compile(r"(^|[\s(/,])(NE|NW|SE|SW|NB|SB|EB|WB|N|S|E|W)\.?(?=$|[\s,;:/)])"), _direction),
    # house numbers and grid coordinates: a number followed by a name or direction
    (re.compile(r"\b(\d{2,5})(?=\s+(?:North|South|East|West|[A-Z][a-z]))"), lambda m: paired(m.group(1))),
]


def normalize(text: str, nav: bool = False) -> str:
    text = text.replace('–', '-').replace('—', ', ').replace('…', '...')
    text = re.sub(r'\s*&\s*', ' and ', text)
    if nav:
        text = re.sub(r'(?<=[A-Za-z])\s*/\s*(?=[A-Za-z])', ', ', text)     # "Kamehameha Hwy/HI-83"
    passes = (_NAV_PASSES[:4] + _COMMON_PASSES + _NAV_PASSES[4:]) if nav else _COMMON_PASSES
    for regex, fn in passes:
        text = regex.sub(fn, text)
    text = re.sub(r'\b(\d+)\.(\d+)\b', lambda m: _decimal(m.group(0)), text)
    text = re.sub(r'(?<![\d,])\b(1[1-9]\d\d|20\d\d)\b(?![\d,])', lambda m: paired(m.group(1)), text)   # years
    text = re.sub(r'\d{1,3}(?:,\d{3})+|\d+', lambda m: cardinal(int(m.group(0).replace(',', ''))), text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\s+([,.;:!?])', r'\1', text)
    text = re.sub(r'([,;:])(?=\S)', r'\1 ', text)
    text = text.replace('\uE000', ' ')
    return text[:1].upper() + text[1:] if text else text
