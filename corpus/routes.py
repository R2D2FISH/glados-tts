"""Valhalla tile building and route sampling via pyvalhalla (no Docker needed).

Each region gets its own tile directory, so routing a big state only costs
that state's RAM. Routes are stored raw (maneuvers JSONL) so instruction
rendering can change without re-routing.
"""
import json
import math
import random
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional

LOG_FIELDS = ('type', 'instruction', 'verbal_transition_alert_instruction',
              'verbal_succinct_transition_instruction', 'verbal_pre_transition_instruction',
              'verbal_post_transition_instruction', 'verbal_multi_cue', 'street_names',
              'begin_street_names', 'sign', 'roundabout_exit_count', 'length', 'time',
              'travel_mode', 'toll', 'highway', 'ferry')


def _bin(name: str) -> str:
    out = subprocess.run([sys.executable, '-m', 'valhalla', 'print_bin_path'],
                         check=True, capture_output=True, text=True).stdout.strip().splitlines()[-1]
    return str(Path(out) / name)


def tile_config(tile_dir: Path) -> Dict:
    """Default Valhalla config pointing at tile_dir (via pyvalhalla's
    valhalla_build_config, which knows every default for this version)."""
    exe = shutil.which('valhalla_build_config', path=str(Path(sys.executable).parent)) \
        or shutil.which('valhalla_build_config')
    out = subprocess.run([exe, '--mjolnir-tile-dir', str(tile_dir / 'tiles'),
                          '--mjolnir-tile-extract', str(tile_dir / 'tiles.tar'),
                          '--mjolnir-timezone', str(tile_dir / 'timezones.sqlite'),
                          '--mjolnir-admin', str(tile_dir / 'admins.sqlite')],
                         check=True, capture_output=True, text=True).stdout
    cfg = json.loads(out)
    # routing logs "no route" errors per failed pair; keep them out of the console
    cfg['logging'] = {'type': 'file', 'color': False, 'file_name': str(tile_dir / 'valhalla.log')}
    cfg['loki']['service_defaults']['minimum_reachability'] = 30
    return cfg


def build_tiles(pbf: Path, tile_dir: Path, force: bool = False) -> Path:
    cfg_path = tile_dir / 'valhalla.json'
    if cfg_path.exists() and any((tile_dir / 'tiles').glob('*/*')) and not force:
        return cfg_path
    tile_dir.mkdir(parents=True, exist_ok=True)
    cfg = tile_config(tile_dir)
    cfg_path.write_text(json.dumps(cfg, indent=1))
    log = tile_dir / 'build.log'
    print(f'building tiles for {pbf.name} -> {tile_dir} (log: {log})')
    with open(log, 'w') as f:
        subprocess.run([_bin('valhalla_build_tiles'), '-c', str(cfg_path), str(pbf)],
                       check=True, stdout=f, stderr=subprocess.STDOUT)
    # the tar extract is optional; tile_dir works on its own
    return cfg_path


# ---------------------------------------------------------------------------
# endpoint sampling

def _haversine_km(a, b):
    la1, lo1, la2, lo2 = map(math.radians, (a['lat'], a['lon'], b['lat'], b['lon']))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 6371 * 2 * math.asin(math.sqrt(h))


class EndpointSampler:
    """Draws origin/destination pairs.

    Endpoints mix population-weighted places (towns, neighbourhoods), address
    points and random road nodes; the last two reach the residential streets
    whose names are the interesting part. Trip length is drawn log-uniformly
    between min_km and max_km so short in-town hops and long highway runs both
    show up."""

    def __init__(self, endpoints: Dict, seed: int = 0, min_km: float = 0.5, max_km: float = 150,
                 mix=(0.25, 0.35, 0.40)):
        self.rng = random.Random(seed)
        self.pools = [endpoints.get('places', []), endpoints.get('addresses', []), endpoints.get('roads', [])]
        self.weights = [m if p else 0 for m, p in zip(mix, self.pools)]
        if not any(self.weights):
            raise ValueError('no endpoints harvested for this region')
        places = self.pools[0]
        self.place_w = [math.log10(max(p.get('pop', 1), 10)) for p in places]
        self.min_km, self.max_km = min_km, max_km

    def point(self):
        pool = self.rng.choices(range(3), weights=self.weights)[0]
        if pool == 0:
            return self.rng.choices(self.pools[0], weights=self.place_w)[0]
        return self.rng.choice(self.pools[pool])

    def pair(self, tries: int = 200):
        target = math.exp(self.rng.uniform(math.log(self.min_km), math.log(self.max_km)))
        a = self.point()
        best, best_err = None, float('inf')
        for _ in range(tries):
            b = self.point()
            err = abs(math.log(max(_haversine_km(a, b), 0.05) / target))
            if err < best_err:
                best, best_err = b, err
            if err < 0.25:
                break
        return a, best


def sample_routes(cfg_path: Path, endpoints: Dict, n: int, out: Path, seed: int = 0,
                  units: str = 'miles', costing: str = 'auto', max_km: float = 150) -> int:
    import valhalla
    actor = valhalla.Actor(str(cfg_path))
    sampler = EndpointSampler(endpoints, seed=seed, max_km=max_km)
    done = 0
    existing = sum(1 for _ in open(out)) if out.exists() else 0
    with open(out, 'a', encoding='utf-8') as f:
        attempts = 0
        while done + existing < n and attempts < n * 4:
            attempts += 1
            a, b = sampler.pair()
            req = {'locations': [{'lat': a['lat'], 'lon': a['lon']}, {'lat': b['lat'], 'lon': b['lon']}],
                   'costing': costing, 'units': units,
                   'directions_options': {'units': units, 'language': 'en-US'}}
            try:
                r = actor.route(req)
            except Exception:                     # no route between the points
                continue
            r = json.loads(r) if isinstance(r, str) else r
            legs = r.get('trip', {}).get('legs', [])
            if not legs:
                continue
            man = [{k: m[k] for k in LOG_FIELDS if k in m} for m in legs[0]['maneuvers']]
            rec = {'from': a, 'to': b, 'units': units, 'maneuvers': man,
                   'length': r['trip']['summary']['length'], 'time': r['trip']['summary']['time']}
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            done += 1
    return done + existing


def read_routes(path: Path) -> Iterator[Dict]:
    with open(path, encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)
