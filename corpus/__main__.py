"""Distillation corpus builder.

    python -m corpus fetch    --regions hawaii,utah_slc     # Geofabrik downloads
    python -m corpus harvest  --regions hawaii               # names, lexicon, endpoints
    python -m corpus tiles    --regions hawaii               # Valhalla tiles
    python -m corpus routes   --regions hawaii [--n 500]     # sample routes
    python -m corpus osm      --regions hawaii               # fetch+harvest+tiles+routes
    python -m corpus prose-llm --batches 50                  # optional: Claude-written prose (via Claude Code)
    python -m corpus build    --profiles eval,tiny,small,general
    python -m corpus render   --manifest corpus_work/manifests/tiny.jsonl --model models/glados-new.pt
    python -m corpus say "In 0.3 mi, turn L onto Kalanianaʻole Hwy" --nav

--regions all selects every region in config/regions.yaml. Work files go to
--work (default ./corpus_work). A local .pbf can stand in for a download:
put it at <work>/osm/<pbf name> or pass --pbf-dir.
"""
import argparse
import json
import sys
from pathlib import Path

import yaml

from corpus import build as build_mod
from corpus.build import CONFIG


def _regions(args):
    cfg = yaml.safe_load(open(args.regions_config))
    regions = cfg['regions']
    if args.regions in (None, '', 'all'):
        names = list(regions)
    else:
        names = [r.strip() for r in args.regions.split(',')]
        missing = [r for r in names if r not in regions]
        if missing:
            sys.exit(f'unknown regions: {missing}; see {args.regions_config}')
    return cfg, {r: regions[r] for r in names}


def _pbf(args, cfg, region_cfg):
    return Path(args.pbf_dir or Path(args.work) / 'osm') / region_cfg['pbf']


def cmd_fetch(args):
    from corpus.osm import download
    cfg, regions = _regions(args)
    for pbf in sorted({r['pbf'] for r in regions.values()}):
        download(f"{cfg['geofabrik']}/{pbf}", Path(args.pbf_dir or Path(args.work) / 'osm') / pbf, force=args.force)


def cmd_harvest(args):
    from corpus.osm import harvest
    cfg, regions = _regions(args)
    for name, r in regions.items():
        harvest(_pbf(args, cfg, r), Path(args.work) / 'regions' / name, bbox=r.get('bbox'),
                seed=args.seed, hawaiian=bool(r.get('hawaiian')))


def cmd_tiles(args):
    from corpus.routes import build_tiles
    cfg, regions = _regions(args)
    built = {}
    for name, r in regions.items():
        # regions cut from the same state share one tile set
        tile_dir = Path(args.work) / 'tiles' / Path(r['pbf']).name.replace('-latest.osm.pbf', '')
        if tile_dir not in built:
            build_tiles(_pbf(args, cfg, r), tile_dir, force=args.force)
            built[tile_dir] = True


def cmd_routes(args):
    from corpus.routes import build_tiles, sample_routes
    cfg, regions = _regions(args)
    for name, r in regions.items():
        rdir = Path(args.work) / 'regions' / name
        tile_dir = Path(args.work) / 'tiles' / Path(r['pbf']).name.replace('-latest.osm.pbf', '')
        cfg_path = build_tiles(_pbf(args, cfg, r), tile_dir)
        endpoints = json.load(open(rdir / 'endpoints.json'))
        n = args.n or r.get('routes', 300)
        total = sample_routes(cfg_path, endpoints, n, rdir / 'routes.jsonl', seed=args.seed,
                              max_km=r.get('max_km', 150))
        print(f'{name}: {total} routes')


def cmd_osm(args):
    cmd_fetch(args)
    cmd_harvest(args)
    cmd_tiles(args)
    cmd_routes(args)


def cmd_prose_llm(args):
    from corpus.prose_llm import generate
    path = generate(Path(args.work) / 'prose', batches=args.batches, per_batch=args.per_batch, seed=args.seed,
                    backend=args.backend, model=args.model, jobs=args.jobs)
    print(f'wrote {path}')


def cmd_build(args):
    _, regions = _regions(args)
    report = build_mod.build(Path(args.work), [p.strip() for p in args.profiles.split(',')], regions,
                             extra_dirs=[Path(d) for d in args.extra_dir], seed=args.seed,
                             profiles_cfg=Path(args.profiles_config))
    for section, cats in report.items():
        tot_h = sum(c['hours'] for c in cats.values())
        tot_n = sum(c['items'] for c in cats.values())
        print(f'\n[{section}]  {tot_n} items, {tot_h:.1f} h')
        for c, v in cats.items():
            print(f'  {c:16} {v["items"]:7d}  {v["hours"]:6.2f} h')


def cmd_render(args):
    from corpus.teacher import render
    manifest = Path(args.manifest)
    out = Path(args.out or Path(args.work) / 'render' / manifest.stem)
    n = render(manifest, out, Path(args.model), Path(args.vocoder), Path(args.emb),
               shard=args.shard, device=args.device, limit=args.limit)
    print(f'rendered {n} utterances into {out}')


def cmd_say(args):
    from corpus.frontend import Frontend
    lex = sorted(Path(args.work).glob('regions/*/hawaiian_lexicon.tsv'))
    fe = Frontend(lexicons=lex)
    print('spoken:  ', fe.spoken(args.text, args.nav))
    print('phonemes:', fe(args.text, args.nav))


def main(argv=None):
    ap = argparse.ArgumentParser(prog='python -m corpus', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', default='corpus_work')
    ap.add_argument('--regions-config', default=str(CONFIG / 'regions.yaml'))
    ap.add_argument('--profiles-config', default=str(CONFIG / 'profiles.yaml'))
    ap.add_argument('--seed', type=int, default=0)
    sub = ap.add_subparsers(dest='cmd', required=True)

    def region_cmd(name, fn, help):
        p = sub.add_parser(name, help=help)
        p.add_argument('--regions', default='hawaii')
        p.add_argument('--pbf-dir')
        p.add_argument('--force', action='store_true')
        p.set_defaults(fn=fn)
        return p

    region_cmd('fetch', cmd_fetch, 'download Geofabrik extracts')
    region_cmd('harvest', cmd_harvest, 'harvest names, Hawaiian lexicon and route endpoints')
    region_cmd('tiles', cmd_tiles, 'build Valhalla tiles')
    region_cmd('routes', cmd_routes, 'sample routes').add_argument('--n', type=int)
    region_cmd('osm', cmd_osm, 'fetch + harvest + tiles + routes').add_argument('--n', type=int)

    p = sub.add_parser('prose-llm', help='generate original prose with Claude (Claude Code or the API)')
    p.add_argument('--backend', choices=['claude-code', 'api'], default='claude-code')
    p.add_argument('--model', help='default: claude-sonnet-5-5 (claude-code), claude-opus-5-5 (api)')
    p.add_argument('--batches', type=int, default=20)
    p.add_argument('--per-batch', type=int, default=40)
    p.add_argument('--jobs', type=int, default=2, help='batches in flight at once')
    p.set_defaults(fn=cmd_prose_llm)

    p = sub.add_parser('build', help='assemble pool and write profile manifests')
    p.add_argument('--profiles', default='eval,tiny,small,general')
    p.add_argument('--regions', default='all')
    p.add_argument('--extra-dir', action='append', default=[],
                   help='more text: curated files ("# category: X" header) or plain books')
    p.set_defaults(fn=cmd_build)

    p = sub.add_parser('render', help='render a manifest with the teacher model')
    p.add_argument('--manifest', required=True)
    p.add_argument('--out')
    p.add_argument('--model', default='models/glados-new.pt')
    p.add_argument('--vocoder', default='models/vocoder-gpu.pt')
    p.add_argument('--emb', default='models/emb/glados_p2.pt')
    p.add_argument('--shard', default='0/1')
    p.add_argument('--device', default='auto')
    p.add_argument('--limit', type=int, default=0)
    p.set_defaults(fn=cmd_render)

    p = sub.add_parser('say', help='show normalization and phonemes for a line')
    p.add_argument('text')
    p.add_argument('--nav', action='store_true')
    p.set_defaults(fn=cmd_say)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
