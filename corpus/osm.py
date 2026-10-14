"""OSM extract handling: download, harvest street names, sample route endpoints.

Everything streams through pyosmium so memory stays flat even on large states:
pass 1 reads ways (names, refs, a reservoir of road node ids) plus place and
address nodes; pass 2 looks up coordinates for the sampled road nodes.
"""
import json
import random
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import osmium

from corpus.hawaiian import is_marked, normalize_okina, strip_marks, syllabify

# Roads a car is routed along, and that people say out loud.
ROAD_CLASSES = {
    'motorway', 'trunk', 'primary', 'secondary', 'tertiary', 'unclassified', 'residential',
    'motorway_link', 'trunk_link', 'primary_link', 'secondary_link', 'tertiary_link',
    'living_street', 'service',
}
ENDPOINT_CLASSES = {'primary', 'secondary', 'tertiary', 'unclassified', 'residential', 'living_street'}
NAME_KEYS = ('name', 'alt_name', 'old_name', 'name:haw', 'official_name', 'short_name')


def download(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    tmp = dest.with_suffix(dest.suffix + '.part')
    print(f'downloading {url}')
    with urllib.request.urlopen(url) as r, open(tmp, 'wb') as f:
        shutil.copyfileobj(r, f, length=1 << 20)
    tmp.rename(dest)
    return dest


def _in_bbox(lon, lat, bbox):
    return bbox is None or (bbox[0] <= lon <= bbox[2] and bbox[1] <= lat <= bbox[3])


def harvest(pbf: Path, out_dir: Path, bbox: Optional[Sequence[float]] = None,
            road_samples: int = 20000, address_samples: int = 20000, seed: int = 0,
            hawaiian: bool = False) -> Dict:
    """Write names.tsv, refs.tsv, endpoints.json and, for Hawaiian regions,
    hawaiian_lexicon.tsv to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    names: Counter = Counter()
    name_class: Dict[str, Counter] = defaultdict(Counter)
    refs: Counter = Counter()
    places, addresses, road_nodes = [], [], []
    seen_ways = seen_addr = 0

    fp = osmium.FileProcessor(str(pbf), osmium.osm.NODE | osmium.osm.WAY)
    for obj in fp:
        if obj.is_node():
            tags = obj.tags
            if not obj.location.valid():
                continue
            lon, lat = obj.location.lon, obj.location.lat
            if not _in_bbox(lon, lat, bbox):
                continue
            if 'place' in tags and 'name' in tags and tags['place'] in (
                    'city', 'town', 'village', 'suburb', 'neighbourhood', 'hamlet', 'quarter'):
                pop = tags.get('population', '')
                pop = int(pop) if pop.isdigit() else {'city': 50000, 'town': 5000, 'suburb': 5000}.get(tags['place'], 500)
                places.append({'name': tags['name'], 'lat': lat, 'lon': lon, 'pop': pop, 'place': tags['place']})
            elif 'addr:street' in tags and 'addr:housenumber' in tags:
                seen_addr += 1
                rec = {'lat': lat, 'lon': lon, 'street': tags['addr:street'],
                       'number': tags['addr:housenumber'], 'city': tags.get('addr:city', '')}
                if len(addresses) < address_samples:
                    addresses.append(rec)
                else:
                    j = rng.randrange(seen_addr)
                    if j < address_samples:
                        addresses[j] = rec
        elif obj.is_way():
            tags = obj.tags
            hw = tags.get('highway')
            if hw not in ROAD_CLASSES:
                continue
            for key in NAME_KEYS:
                if key in tags:
                    for n in tags[key].split(';'):
                        n = normalize_okina(n.strip())
                        if n:
                            names[n] += 1
                            name_class[n][hw] += 1
            if 'ref' in tags:
                for r in tags['ref'].split(';'):
                    refs[r.strip()] += 1
            if hw in ENDPOINT_CLASSES and len(obj.nodes):
                seen_ways += 1
                node_id = obj.nodes[rng.randrange(len(obj.nodes))].ref
                if len(road_nodes) < road_samples:
                    road_nodes.append(node_id)
                else:
                    j = rng.randrange(seen_ways)
                    if j < road_samples:
                        road_nodes[j] = node_id

    # pass 2: coordinates for the sampled road nodes
    wanted = set(road_nodes)
    road_points = []
    for obj in osmium.FileProcessor(str(pbf), osmium.osm.NODE).with_filter(osmium.filter.IdFilter(wanted)):
        if obj.location.valid() and _in_bbox(obj.location.lon, obj.location.lat, bbox):
            road_points.append({'lat': obj.location.lat, 'lon': obj.location.lon})

    with open(out_dir / 'names.tsv', 'w', encoding='utf-8') as f:
        f.write('name\tcount\tclasses\n')
        for n, c in names.most_common():
            f.write(f"{n}\t{c}\t{','.join(k for k, _ in name_class[n].most_common(3))}\n")
    with open(out_dir / 'refs.tsv', 'w', encoding='utf-8') as f:
        for r, c in refs.most_common():
            f.write(f'{r}\t{c}\n')
    with open(out_dir / 'endpoints.json', 'w', encoding='utf-8') as f:
        json.dump({'places': places, 'addresses': addresses, 'roads': road_points}, f)
    lex = hawaiian_lexicon(names) if hawaiian else {}
    if lex:
        write_lexicon(lex, out_dir / 'hawaiian_lexicon.tsv')
    stats = {'names': len(names), 'refs': len(refs), 'places': len(places),
             'addresses': len(addresses), 'road_points': len(road_points), 'hawaiian_words': len(lex)}
    print(f'{pbf.name}: ' + ', '.join(f'{k}={v}' for k, v in stats.items()))
    return stats


# ---------------------------------------------------------------------------
# Hawaiian lexicon from street names

ENGLISH_STREET_WORDS = {
    'street', 'avenue', 'road', 'drive', 'lane', 'place', 'way', 'loop', 'highway', 'boulevard',
    'circle', 'court', 'parkway', 'freeway', 'terrace', 'trail', 'alley', 'mall', 'walk', 'path',
    'bridge', 'access', 'service', 'north', 'south', 'east', 'west', 'old', 'new', 'the', 'of',
    'and', 'park', 'pine', 'home', 'lake', 'make', 'wake', 'mile', 'pole', 'hole', 'one', 'lone',
    'name', 'line', 'lime', 'nine', 'maple', 'pali', 'kane',
}


def hawaiian_lexicon(names: Counter, min_syllables: int = 2) -> Dict[str, Dict]:
    """Group Hawaiian-shaped tokens of street names by their unmarked spelling.

    Returns {bare: {'marked': best marked spelling or '', 'variants': Counter,
    'count': n}}. A bare form whose marked spelling never appears is still
    included (read in the local register)."""
    lex: Dict[str, Dict] = {}
    for name, count in names.items():
        for tok in name.replace('-', ' ').split():
            tok = tok.strip('.,()"')
            if not tok or not tok[0].isalpha() and tok[0] != 'ʻ':
                continue
            sylls = syllabify(tok)
            if sylls is None or len(sylls) < min_syllables:
                continue
            bare = strip_marks(tok).lower()
            if bare in ENGLISH_STREET_WORDS and not is_marked(tok):
                continue
            e = lex.setdefault(bare, {'variants': Counter(), 'count': 0})
            e['variants'][tok] += count
            e['count'] += count
    for bare, e in lex.items():
        marked = [(c, v) for v, c in e['variants'].items() if is_marked(v)]
        e['marked'] = max(marked)[1] if marked else ''
    return lex


def write_lexicon(lex: Dict[str, Dict], path: Path) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write('# word\tbare\tcount\tvariants  (word = marked spelling when one was seen)\n')
        for bare, e in sorted(lex.items(), key=lambda kv: -kv[1]['count']):
            word = e['marked'] or max(e['variants'].items(), key=lambda kv: kv[1])[0]
            variants = ','.join(v for v, _ in e['variants'].most_common(5))
            f.write(f"{word}\t{bare}\t{e['count']}\t{variants}\n")


def read_names(path: Path) -> List[Dict]:
    rows = []
    with open(path, encoding='utf-8') as f:
        next(f)
        for line in f:
            name, count, classes = line.rstrip('\n').split('\t')
            rows.append({'name': name, 'count': int(count), 'classes': classes.split(',')})
    return rows
