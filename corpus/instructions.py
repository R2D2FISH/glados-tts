"""Turn Valhalla maneuvers into corpus text in two styles.

verbal  Valhalla's own spoken instructions (verbal_pre_transition etc.), which
        are full sentences with expanded names.
terse   what a phone app puts on screen and a TTS frontend actually receives:
        "In 0.3 mi, turn R onto W 5th St", "Take exit 23B toward I-5 N".
        Abbreviations are applied at random so the normalizer is exercised.
"""
import random
import re
from typing import Dict, Iterator, List, Optional

# Valhalla maneuver types (valhalla/proto/directions.proto)
START, START_R, START_L = 1, 2, 3
DEST, DEST_R, DEST_L = 4, 5, 6
BECOMES, CONTINUE = 7, 8
SLIGHT_R, RIGHT, SHARP_R, UTURN_R, UTURN_L, SHARP_L, LEFT, SLIGHT_L = 9, 10, 11, 12, 13, 14, 15, 16
RAMP_STRAIGHT, RAMP_R, RAMP_L, EXIT_R, EXIT_L = 17, 18, 19, 20, 21
STAY_STRAIGHT, STAY_R, STAY_L, MERGE = 22, 23, 24, 25
ROUNDABOUT_ENTER, ROUNDABOUT_EXIT, FERRY_ENTER, FERRY_EXIT = 26, 27, 28, 29
MERGE_R, MERGE_L = 37, 38

SUFFIX_ABBR = {
    'Street': 'St', 'Avenue': 'Ave', 'Boulevard': 'Blvd', 'Road': 'Rd', 'Drive': 'Dr',
    'Lane': 'Ln', 'Court': 'Ct', 'Place': 'Pl', 'Parkway': 'Pkwy', 'Highway': 'Hwy',
    'Freeway': 'Fwy', 'Expressway': 'Expy', 'Turnpike': 'Tpke', 'Circle': 'Cir',
    'Terrace': 'Ter', 'Trail': 'Trl', 'Square': 'Sq', 'Heights': 'Hts', 'Center': 'Ctr',
}
DIR_ABBR = {'North': 'N', 'South': 'S', 'East': 'E', 'West': 'W', 'Northeast': 'NE',
            'Northwest': 'NW', 'Southeast': 'SE', 'Southwest': 'SW'}
ORD = {1: '1st', 2: '2nd', 3: '3rd'}


def _ord(n: int) -> str:
    return ORD.get(n, f'{n}th')


class Terse:
    def __init__(self, rng: random.Random, abbreviate: float = 0.7, metric: float = 0.05):
        self.rng, self.p_abbr, self.p_metric = rng, abbreviate, metric

    def street(self, name: str) -> str:
        if self.rng.random() > self.p_abbr:
            return name
        words = name.split()
        if len(words) > 1 and words[-1] in SUFFIX_ABBR:
            words[-1] = SUFFIX_ABBR[words[-1]]
        if len(words) > 1 and words[0] in DIR_ABBR:
            words[0] = DIR_ABBR[words[0]]
        if len(words) > 1 and words[-1] in DIR_ABBR:
            words[-1] = DIR_ABBR[words[-1]]
        out = ' '.join(words)
        # route refs: "I 405" -> "I-405", "US 101" -> "US-101"
        if self.rng.random() < 0.6:
            out = re.sub(r'\b(I|US|SR|CA|HI|TX|FM|H)\s+(\d)', r'\1-\2', out)
        return out

    def streets(self, m: Dict, key: str = 'street_names') -> str:
        names = m.get(key) or m.get('street_names') or []
        if not names:
            return ''
        if len(names) > 1 and self.rng.random() < 0.4:
            return '/'.join(self.street(n) for n in names[:2])
        return self.street(names[0])

    def distance(self, miles: float, units: str = 'miles') -> str:
        r = self.rng
        if units == 'kilometers' or r.random() < self.p_metric:
            km = miles * (1.609344 if units != 'kilometers' else 1)
            if km < 1:
                return f'{int(round(km * 1000, -1))} m'
            return f'{km:.1f} km'.replace('.0 ', ' ')
        if miles < 0.19:
            ft = max(50, int(round(miles * 5280 / 50.0)) * 50)
            return f"{ft:,} {r.choice(['ft', 'ft', 'feet'])}"
        for frac, label in ((0.25, '1/4'), (0.5, '1/2'), (0.75, '3/4')):
            if abs(miles - frac) < 0.02 and r.random() < 0.5:
                return f'{label} mi'
        unit = r.choice(['mi', 'mi', 'mi', 'miles'])
        if miles < 10:
            s = f'{miles:.1f}'
            return f"{s[:-2] if s.endswith('.0') else s} {unit}"
        return f'{round(miles)} {unit}'

    def sign(self, m: Dict, kind: str) -> List[str]:
        return [e['text'] for e in m.get('sign', {}).get(f'exit_{kind}_elements', [])]

    def toward(self, m: Dict) -> str:
        t = self.sign(m, 'toward') or self.sign(m, 'branch')
        if not t:
            return ''
        return ' toward ' + '/'.join(self.street(x) for x in t[:2])

    def maneuver(self, m: Dict, units: str) -> Optional[str]:
        r, t = self.rng, m['type']
        onto = self.streets(m)
        L = lambda words: r.choice(words)
        side = 'right' if t in (SLIGHT_R, RIGHT, SHARP_R, UTURN_R, RAMP_R, EXIT_R, STAY_R, MERGE_R, START_R, DEST_R) else 'left'
        short = side[0].upper()
        if t in (START, START_R, START_L):
            d = re.search(r'\b(north|south|east|west|northeast|northwest|southeast|southwest)\b', m.get('instruction', ''))
            head = d.group(1) if d else ''
            if head and r.random() < 0.5:
                head = DIR_ABBR[head.capitalize()]
            return f"{L(['Head', 'Head', 'Drive'])} {head} on {onto}".replace('  ', ' ') if onto else f'Head {head}'
        if t in (DEST, DEST_R, DEST_L):
            where = '' if t == DEST else f' on the {side}'
            return L([f'Arrive at destination{where}', f'Your destination is{where or " ahead"}',
                      f'Destination{where or " ahead"}', f'You have arrived{where and ". Destination" + where}'])
        if t == BECOMES:
            prev = self.streets(m, 'begin_street_names')
            return f'{prev} becomes {onto}' if prev and onto else None
        if t in (CONTINUE, STAY_STRAIGHT):
            dist = self.distance(m.get('length', 0), units)
            return L([f'Continue on {onto}', f'Continue on {onto} for {dist}', f'Stay on {onto} for {dist}',
                      f'Continue straight to stay on {onto}']) if onto else f'Continue straight for {dist}'
        if t in (RIGHT, LEFT, SLIGHT_R, SLIGHT_L, SHARP_R, SHARP_L):
            adj = {SLIGHT_R: L(['Slight', 'Bear']), SLIGHT_L: L(['Slight', 'Bear']),
                   SHARP_R: 'Sharp', SHARP_L: 'Sharp'}.get(t, 'Turn')
            word = L([side, side, short]) if adj in ('Turn', 'Bear') else side
            prep = L([' onto ', ' onto ', ' on ', ' at '])
            return f'{adj} {word}{prep + onto if onto else ""}'
        if t in (UTURN_R, UTURN_L):
            return L(['Make a U-turn', f'Make a U-turn at {onto}' if onto else 'U-turn', 'Make a U-turn when possible'])
        if t in (RAMP_STRAIGHT, RAMP_R, RAMP_L):
            tw = self.toward(m)
            return f"Take the ramp{' onto ' + onto if onto and r.random() < 0.5 else ''}{tw}" if (onto or tw) else 'Take the ramp'
        if t in (EXIT_R, EXIT_L):
            num = self.sign(m, 'number')
            tw = self.toward(m)
            ex = f"exit {num[0]}" if num else 'the exit'
            return L([f'Take {ex}{tw}', f'Take {ex} onto {onto}' if onto else f'Take {ex}{tw}',
                      f'Use the {side} lane to take {ex}{tw}'])
        if t in (STAY_R, STAY_L):
            tw = self.toward(m)
            return L([f'Keep {side} at the fork{tw}', f'Keep {short}{tw or " to stay on " + onto}',
                      f'Keep {side} to stay on {onto}' if onto else f'Keep {side}{tw}'])
        if t in (MERGE, MERGE_R, MERGE_L):
            return f'Merge onto {onto}' if onto else 'Merge'
        if t == ROUNDABOUT_ENTER:
            n = m.get('roundabout_exit_count')
            if not n:
                return 'Enter the roundabout'
            return L([f'At the roundabout, take the {_ord(n)} exit' + (f' onto {onto}' if onto else ''),
                      f'Enter roundabout, {_ord(n)} exit'])
        if t == ROUNDABOUT_EXIT:
            return f'Exit the roundabout onto {onto}' if onto else 'Exit the roundabout'
        if t == FERRY_ENTER:
            return f'Board the {onto or "ferry"}'
        if t == FERRY_EXIT:
            return f'Drive off the ferry onto {onto}' if onto else 'Leave the ferry'
        return m.get('instruction')


_VERBS = r'(?:Turn|Bear|Keep|Take|Make|Enter|Exit|Continue|Merge|Stay|Drive|Head|Go|Board|Use)'


def _fix_verbal(s: str) -> str:
    # Valhalla capitalizes the verb after "Then" / "Then, in 200 feet,"
    return re.sub(r'(\bThen,?(?: in [^,]+,)? )(' + _VERBS + r')\b', lambda m: m.group(1) + m.group(2).lower(), s)


def render_route(route: Dict, rng: random.Random, region: str, route_idx: int,
                 p_preannounce: float = 0.45, p_chain: float = 0.5) -> Iterator[Dict]:
    """Yield nav_terse and nav_verbal items for one route."""
    terse = Terse(rng)
    units = route.get('units', 'miles')
    mans = route['maneuvers']
    for i, m in enumerate(mans):
        base = {'source': 'valhalla', 'region': region, 'route': route_idx, 'maneuver': i, 'mtype': m['type']}
        # verbal (pre and alert are often identical; keep one)
        seen = set()
        for key in ('verbal_pre_transition_instruction', 'verbal_post_transition_instruction',
                    'verbal_transition_alert_instruction'):
            v = m.get(key)
            if v and v not in seen and (key == 'verbal_pre_transition_instruction' or rng.random() < 0.3):
                seen.add(v)
                yield {**base, 'text': _fix_verbal(v), 'category': 'nav_verbal', 'tags': ['nav', key.split('_')[1]]}
        # terse
        s = terse.maneuver(m, units)
        if not s:
            continue
        s = s[0].upper() + s[1:]
        if i > 0 and m['type'] not in (DEST, DEST_R, DEST_L) and rng.random() < p_preannounce:
            s = f'In {terse.distance(mans[i - 1].get("length", 0.1), units)}, {s[0].lower() + s[1:]}'
        if i + 1 < len(mans) and m.get('length', 1) < 0.1 and rng.random() < p_chain:
            nxt = terse.maneuver(mans[i + 1], units)
            if nxt:
                s = f'{s}, then {nxt[0].lower() + nxt[1:]}'
        yield {**base, 'text': s, 'category': 'nav_terse', 'tags': ['nav', 'terse']}


def signature(text: str) -> str:
    """Template signature used to cap near-duplicates: names and numbers out."""
    s = re.sub(r"\d[\d,./]*", '#', text)
    s = re.sub(r"\b[A-Zʻ][\wʻ'\-]*(?:\s+[A-Zʻ][\wʻ'\-]*)*", 'N', s[1:])
    return text[0] + s
