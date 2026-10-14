"""Assemble the item pool and select per-profile manifests.

pool:   routes (rendered terse + verbal), Hawaiian street names, curated files,
        place names, numbers, prose files -> deduplicated items
phones: every item goes through corpus.frontend once (cached on disk)
select: the eval set is drawn first and held out of everything else; then each
        profile fills a per-category audio budget by lazy-greedy coverage:
        gain = new diphones + new words (+ new street names), per second of
        audio, with a cap on near-duplicate templates so "Turn right onto X"
        can't eat the nav budget.
"""
import hashlib
import heapq
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from corpus import passages
from corpus.hawaiian import HawaiianClassifier, load_lexicon
from corpus.instructions import render_route, signature
from corpus.routes import read_routes

CONFIG = Path(__file__).parent / 'config'
NAV_CATEGORIES = {'nav_terse', 'nav_verbal', 'hawaiian', 'numbers', 'place_names'}
SEC_PER_PHONE = 0.065          # teacher at alpha 1.0: ~204 frames for 37 symbols
SIGNATURE_CAP = {'nav_terse': 40, 'nav_verbal': 40, 'hawaiian': 400, 'numbers': 60}


def item_id(item: Dict) -> str:
    return hashlib.sha1(f"{item['category']}|{item['text']}".encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# pool

def region_items(work: Path, regions: Dict, seed: int = 0) -> List[Dict]:
    items = []
    for name, cfg in regions.items():
        rdir = work / 'regions' / name
        routes = rdir / 'routes.jsonl'
        if routes.exists():
            rng = random.Random(f'{seed}:{name}')
            for i, route in enumerate(read_routes(routes)):
                for it in render_route(route, rng, name, i):
                    it['nav'] = True
                    items.append(it)
        names = rdir / 'names.tsv'
        if cfg.get('hawaiian') and names.exists():
            from corpus.osm import read_names
            lex = rdir / 'hawaiian_lexicon.tsv'
            seed_words = load_lexicon(passages.DATA / 'hawaiian_seed.txt')
            words = (load_lexicon(lex) if lex.exists() else []) + seed_words
            clf = HawaiianClassifier(words, trusted=seed_words)
            for it in passages.hawaiian_street_items(read_names(names), clf, seed=seed):
                it['region'] = name
                items.append(it)
    return items


def prose_items(dirs: Iterable[Path], max_per_book: int = 4000, seed: int = 0) -> List[Dict]:
    """Text files in prose dirs: curated format if they start with '# category',
    otherwise treated as books and split into sentences."""
    items = []
    for d in dirs:
        for p in sorted(Path(d).glob('*.txt')):
            head = open(p, encoding='utf-8', errors='ignore').read(200)
            if head.startswith('# category'):
                items.extend(passages.read_curated(p))
            else:
                items.extend(passages.prose_from_text(p, max_items=max_per_book, seed=seed))
    return items


def build_pool(work: Path, regions: Dict, extra_dirs: Iterable[Path] = (), seed: int = 0,
               numbers: int = 3000) -> List[Dict]:
    items = region_items(work, regions, seed)
    items += passages.curated_items()
    items += passages.place_name_items(seed=seed)
    items += passages.number_items(numbers, seed=seed)
    items += prose_items([work / 'prose', *extra_dirs], seed=seed)
    pool, seen = [], set()
    for it in items:
        key = it['text']
        if key in seen:
            continue
        seen.add(key)
        it.setdefault('tags', [])
        it.setdefault('nav', it['category'] in NAV_CATEGORIES)
        it['id'] = item_id(it)
        pool.append(it)
    return pool


# ---------------------------------------------------------------------------
# phonemes (cached)

def phonemize_pool(pool: List[Dict], work: Path, lexicons: List[Path]) -> None:
    from corpus.frontend import Frontend

    h = hashlib.sha1()
    for p in sorted(lexicons) + [passages.DATA / 'hawaiian_seed.txt']:
        if p.exists():
            h.update(p.read_bytes())
    h.update((passages.DATA / 'pronunciations.tsv').read_bytes())
    for mod in ('normalize.py', 'hawaiian.py', 'frontend.py'):
        h.update((Path(__file__).parent / mod).read_bytes())
    version = h.hexdigest()[:10]
    cache_path = work / f'phonemes.{version}.jsonl'
    cache = {}
    if cache_path.exists():
        for line in open(cache_path, encoding='utf-8'):
            r = json.loads(line)
            cache[(r['text'], r['nav'])] = r
    todo = [it for it in pool if (it['text'], it['nav']) not in cache]
    if todo:
        fe = Frontend(lexicons=lexicons)
        print(f'phonemizing {len(todo)} items ({len(cache)} cached)')
        with open(cache_path, 'a', encoding='utf-8') as f:
            for nav in (True, False):
                batch = [it for it in todo if it['nav'] == nav]
                for i in range(0, len(batch), 512):
                    chunk = batch[i:i + 512]
                    spoken = [fe.spoken(it['text'], nav) for it in chunk]
                    phon = fe.batch([it['text'] for it in chunk], nav=nav)
                    for it, s, p in zip(chunk, spoken, phon):
                        r = {'text': it['text'], 'nav': nav, 'spoken': s, 'phonemes': p,
                             'hawaiian': any(hw for hw, _ in fe.segments(s))}
                        cache[(it['text'], nav)] = r
                        f.write(json.dumps(r, ensure_ascii=False) + '\n')
    for it in pool:
        r = cache[(it['text'], it['nav'])]
        it['spoken'], it['phonemes'] = r['spoken'], r['phonemes']
        it['est_sec'] = round(len(r['phonemes']) * SEC_PER_PHONE, 2)
        if r['hawaiian'] and 'hawaiian' not in it['tags']:
            it['tags'].append('hawaiian')
        # route instructions that mention Hawaiian names count toward the Hawaiian budget
        if r['hawaiian'] and it['category'] in ('nav_terse', 'nav_verbal'):
            it['category'] = 'hawaiian'


# ---------------------------------------------------------------------------
# selection

def _units(it: Dict):
    words = {w.lower() for w in it['spoken'].replace(',', ' ').replace('.', ' ').split()}
    ph = it['phonemes'].replace(' ', '_')
    diphones = {ph[i:i + 2] for i in range(len(ph) - 1)}
    return diphones, words


class Selector:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.diphones, self.words = Counter(), Counter()
        self.signatures = Counter()

    def gain(self, it, d, w) -> float:
        g = sum(1 for x in d if not self.diphones[x]) + 2 * sum(1 for x in w if not self.words[x])
        g += 0.25 * sum(1.0 / (1 + self.words[x]) for x in w)      # soft preference for rarer words
        return g / (it['est_sec'] + 1.0)

    def take(self, it, d, w):
        self.diphones.update(d)
        self.words.update(w)
        self.signatures[signature(it['text'])] += 1

    def select(self, cands: List[Dict], budget_sec: float, cap: Optional[int]) -> List[Dict]:
        units = {it['id']: _units(it) for it in cands}
        heap = [(-self.gain(it, *units[it['id']]), self.rng.random(), i) for i, it in enumerate(cands)]
        heapq.heapify(heap)
        chosen, used = [], 0.0
        while heap and used < budget_sec:
            neg, tie, i = heapq.heappop(heap)
            it = cands[i]
            if cap and self.signatures[signature(it['text'])] >= cap:
                continue
            g = self.gain(it, *units[it['id']])
            if heap and -g > heap[0][0] + 1e-9:          # stale score; re-queue
                heapq.heappush(heap, (-g, tie, i))
                continue
            self.take(it, *units[it['id']])
            chosen.append(it)
            used += it['est_sec']
        return chosen


def select_eval(pool: List[Dict], n: int, seed: int, weights: Dict[str, float]) -> List[Dict]:
    rng = random.Random(f'eval:{seed}')
    by_cat = defaultdict(list)
    for it in pool:
        by_cat[it['category']].append(it)
    total = sum(weights.values())
    out = []
    for cat, w in weights.items():
        cands = by_cat.get(cat, [])
        # never take more than a tenth of a category, so small ones still train
        k = min(max(1, len(cands) // 10), max(1, round(n * w / total)))
        # stratify by source so every region/file is represented
        by_src = defaultdict(list)
        for it in cands:
            by_src[it.get('region') or it.get('source', '').split(':')[0]].append(it)
        groups = [rng.sample(v, len(v)) for v in by_src.values()]
        picked = []
        while len(picked) < k and any(groups):
            for g in groups:
                if g and len(picked) < k:
                    picked.append(g.pop())
        out.extend(picked)
    return out


def select_profile(pool: List[Dict], profile: Dict, exclude: set, seed: int) -> List[Dict]:
    by_cat = defaultdict(list)
    for it in pool:
        if it['id'] not in exclude:
            by_cat[it['category']].append(it)
    weights = {c: w for c, w in profile['weights'].items() if w > 0}
    # `hours` is the rendered total, and every item is rendered once per alpha
    total_sec = profile['hours'] * 3600 / len(profile.get('alphas', [1.0]))
    budgets = {c: total_sec * w / sum(weights.values()) for c, w in weights.items()}
    # categories that can't fill their budget hand the rest to the others
    for _ in range(3):
        avail = {c: sum(it['est_sec'] for it in by_cat.get(c, [])) for c in budgets}
        short = {c: budgets[c] - avail[c] for c in budgets if avail[c] < budgets[c]}
        if not short:
            break
        spare = sum(short.values())
        open_cats = {c: weights[c] for c in budgets if c not in short}
        for c in short:
            budgets[c] = avail[c]
        if not open_cats:
            break
        for c, w in open_cats.items():
            budgets[c] += spare * w / sum(open_cats.values())
    sel = Selector(seed)
    chosen = []
    for cat in sorted(budgets, key=lambda c: -weights[c]):
        got = sel.select(by_cat.get(cat, []), budgets[cat], SIGNATURE_CAP.get(cat))
        want = total_sec * weights[cat] / sum(weights.values())
        have = sum(it['est_sec'] for it in got)
        if have < 0.8 * want:
            print(f'  note: {cat} filled {have / 3600:.2f} of {want / 3600:.2f} h requested; '
                  f'add more {cat} material (the rest went to other categories)')
        chosen += got
    return chosen


def write_manifest(items: List[Dict], alphas: List[float], path: Path) -> Dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: {'items': 0, 'sec': 0.0})
    with open(path, 'w', encoding='utf-8') as f:
        for it in items:
            for a in alphas:
                row = {k: it.get(k) for k in ('id', 'category', 'text', 'spoken', 'phonemes', 'tags',
                                              'source', 'region', 'est_sec')}
                row['alpha'] = a
                row['utt'] = f"{it['id']}_a{int(round(a * 100)):03d}"
                f.write(json.dumps(row, ensure_ascii=False) + '\n')
                stats[it['category']]['items'] += 1
                stats[it['category']]['sec'] += it['est_sec'] / a
    return {c: {'items': s['items'], 'hours': round(s['sec'] / 3600, 2)} for c, s in sorted(stats.items())}


def build(work: Path, profiles: List[str], regions: Dict, extra_dirs: Iterable[Path] = (), seed: int = 0,
          profiles_cfg: Path = CONFIG / 'profiles.yaml') -> Dict:
    cfg = yaml.safe_load(open(profiles_cfg))['profiles']
    pool = build_pool(work, regions, extra_dirs, seed)
    lexicons = [work / 'regions' / r / 'hawaiian_lexicon.tsv' for r, c in regions.items() if c.get('hawaiian')]
    phonemize_pool(pool, work, lexicons)
    pool = [it for it in pool if it['phonemes'] and len(it['phonemes']) > 1]
    cats = Counter(it['category'] for it in pool)
    report = {'pool': {c: {'items': n, 'hours': round(sum(i['est_sec'] for i in pool if i['category'] == c) / 3600, 2)}
                       for c, n in sorted(cats.items())}}
    out = work / 'manifests'
    eval_cfg = cfg.get('eval')
    eval_items = select_eval(pool, eval_cfg['items'], seed, eval_cfg['weights']) if eval_cfg else []
    exclude = {it['id'] for it in eval_items}
    if eval_cfg:
        report['eval'] = write_manifest(eval_items, eval_cfg.get('alphas', [1.0]), out / 'eval.jsonl')
    for name in profiles:
        if name == 'eval':
            continue
        chosen = select_profile(pool, cfg[name], exclude, seed)
        report[name] = write_manifest(chosen, cfg[name].get('alphas', [1.0]), out / f'{name}.jsonl')
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    return report
