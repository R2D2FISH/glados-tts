"""Select the sentences for a synthetic GLaDOS corpus.

Mixes general text (nltk corpora or your own files) with generated in-domain
prompts (navigation with your street names, numbers, times), phonemizes
everything with the project's front-end (including the Hawaiian rules and
lexicon.txt), then greedily picks sentences that cover the most phoneme
pairs, so a modest corpus still contains every sound transition.

    python -m tools.corpus.build_text --out corpus/text.tsv --hours 60 \\
        --streets my_streets.txt --text extra_sentences.txt

Output: TSV with columns id, kind, text, phonemes.
"""
import argparse
import heapq
import math
import os
import random
import sys
from collections import Counter

sys.path.insert(0, os.getcwd())

# A starter list of Oʻahu street names; pass --streets for a full list (e.g. from OpenStreetMap).
OAHU_STREETS = """Kalanianaʻole Highway|Kapiʻolani Boulevard|Kalākaua Avenue|Ala Moana Boulevard|Likelike Highway|
Pali Highway|Kamehameha Highway|Farrington Highway|Piʻikoi Street|Keʻeaumoku Street|Kapahulu Avenue|Waiʻalae Avenue|
Kūhiō Avenue|Nuʻuanu Avenue|Punahou Street|Wilder Avenue|Makiki Street|Round Top Drive|Tantalus Drive|
Monsarrat Avenue|Diamond Head Road|Paikau Street|Pākī Avenue|Beretania Street|King Street|Dillingham Boulevard|
Nimitz Highway|Kaʻahumanu Street|Moanalua Road|Kalihi Street|School Street|Liliha Street|Kīlauea Avenue|
Hawaiʻi Kai Drive|Lunalilo Home Road|Keolu Drive|Kailua Road|Oneawa Street|Kāneʻohe Bay Drive|Kahekili Highway|
Fort Weaver Road|Kualakaʻi Parkway|Kapolei Parkway|Ward Avenue|Kamakeʻe Street|Isenberg Street|University Avenue|
Dole Street|Manoa Road|Oʻahu Avenue|Kaimukī Avenue|Kīlauea Avenue|Koko Head Avenue|Hunakai Street|Kāhala Avenue"""
PLACES = ['ʻEwa', 'Waikīkī', 'Kāneʻohe', 'Kailua', 'Pearl City', 'Downtown', 'Kapolei', 'Mililani', 'Hawaiʻi Kai',
          'Wahiawā', 'Haleʻiwa', 'Diamond Head', 'the airport', 'Mānoa', 'Kalihi', 'Waipahu', 'ʻAiea', 'Makakilo']
HIGHWAYS = ['H-1', 'H-2', 'H-3', 'H-201']
PH_PER_SEC = 15


def distance(rng):
    return rng.choice([f'{rng.randrange(1, 10) * 100} feet', 'a quarter mile', 'half a mile', 'three quarters of a mile',
                       '1 mile', f'{rng.randint(2, 15)} miles', f'{rng.randint(1, 9)}.{rng.randint(1, 9)} miles'])


def navigation(rng, streets, n):
    side = lambda: rng.choice(['left', 'right'])  # noqa: E731
    ordinal = lambda: rng.choice(['first', 'second', 'third', 'fourth'])  # noqa: E731
    templates = [
        lambda s: f'In {distance(rng)}, turn {side()} onto {s}.',
        lambda s: f'Turn {side()} onto {s}.',
        lambda s: f'Turn {side()} onto {s}, then turn {side()} onto {rng.choice(streets)}.',
        lambda s: f'Continue on {s} for {distance(rng)}.',
        lambda s: f'Keep {side()} to stay on {s}.',
        lambda s: f'Head {rng.choice(["north", "south", "east", "west"])} on {s} toward {rng.choice(streets)}.',
        lambda s: f'At the roundabout, take the {ordinal()} exit onto {s}.',
        lambda s: f'In {distance(rng)}, use the {side()} lane to merge onto {rng.choice(HIGHWAYS)} '
                  f'{rng.choice(["East", "West"])} toward {rng.choice(PLACES)}.',
        lambda s: f'Take exit {rng.randint(1, 30)}{rng.choice(["", "A", "B"])} toward {s}.',
        lambda s: f'Your destination, {rng.randint(1, 3999)} {s}, is on the {side()}.',
        lambda s: f'Make a U-turn when possible, then continue on {s}.',
        lambda s: f'Rerouting. Turn {side()} onto {s}.',
        lambda s: f'Traffic ahead on {s}. You will arrive at {rng.randint(1, 12)}:{rng.randint(0, 59):02d}.',
    ]
    return [rng.choice(templates)(rng.choice(streets)) for _ in range(n)]


def numbers(rng, n):
    t = [
        lambda: f'The time is {rng.randint(1, 12)}:{rng.randint(0, 59):02d}.',
        lambda: f'You have {rng.randint(2, 999)} new messages.',
        lambda: f'It is {rng.randint(55, 95)} degrees and {rng.choice(["sunny", "cloudy", "raining"])}.',
        lambda: f'Test chamber {rng.randint(1, 99)} is now available.',
        lambda: f'In {rng.randint(1900, 2099)}, the facility was {rng.choice(["expanded", "flooded", "evacuated"])}.',
        lambda: f'That will cost ${rng.randint(1, 500)}.{rng.randint(0, 99):02d}.',
        lambda: f'Approximately {rng.randint(2, 99)}.{rng.randint(1, 9)} percent of subjects survived.',
        lambda: f'The {rng.randint(1, 40)}{rng.choice(["st", "nd", "rd", "th"])} attempt also failed.',
    ]
    return [rng.choice(t)() for _ in range(n)]


def pair_counts(ph):
    s = f'^{ph}$'
    return Counter(s[i:i + 2] for i in range(len(s) - 1))


def greedy_cover(cands, n_target):
    """Pick sentences that maximise coverage of phoneme bigrams, with
    diminishing returns so common pairs keep appearing but rare ones get in.
    Scores only fall as coverage grows, so lazy greedy (re-score the top of a
    heap until it stays on top) gives the same picks as full greedy, fast."""
    have = Counter()

    def score(i):
        pc = cands[i][3]
        return sum(c / (1 + have[p]) ** 2 for p, c in pc.items()) / math.sqrt(1 + len(cands[i][2]))

    heap = [(-score(i), i) for i in range(len(cands))]
    heapq.heapify(heap)
    chosen = []
    while heap and len(chosen) < n_target:
        _, i = heapq.heappop(heap)
        s = score(i)
        if heap and s < -heap[0][0]:
            heapq.heappush(heap, (-s, i))  # stale: re-queue with its current score
            continue
        chosen.append(i)
        have.update(cands[i][3])
    return chosen, have


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default='corpus/text.tsv')
    ap.add_argument('--hours', type=float, default=60, help='target audio duration (estimated from phoneme count)')
    ap.add_argument('--text', nargs='*', default=[], help='extra text files (default general source: nltk corpora)')
    ap.add_argument('--streets', help='file with one street name per line (adds to the built-in Oʻahu list)')
    ap.add_argument('--nav-frac', type=float, default=0.25, help='share of navigation prompts')
    ap.add_argument('--num-frac', type=float, default=0.05, help='share of number/time prompts')
    ap.add_argument('--candidates', type=int, default=60000, help='general sentences to consider')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    from tools.distill.make_dataset import load_sentences
    from utils.text.cleaners import Cleaner

    rng = random.Random(args.seed)
    streets = [s.strip() for s in OAHU_STREETS.replace('\n', '').split('|') if s.strip()]
    if args.streets:
        streets += [s.strip() for s in open(args.streets, encoding='utf-8') if s.strip()]
    # ~15 phonemes per second at GLaDOS's pace (measured); used only to size the corpus
    target_ph = args.hours * 3600 * PH_PER_SEC

    general = []
    for f in args.text:
        general += load_sentences(f, args.candidates, args.seed)
    if not args.text:
        general = load_sentences(None, args.candidates, args.seed)
    n_general_target = int(target_ph * (1 - args.nav_frac - args.num_frac) / 60)  # ~60 phonemes / sentence
    nav = navigation(rng, streets, max(10, int(target_ph * args.nav_frac / 45)))
    num = numbers(rng, max(5, int(target_ph * args.num_frac / 30)))

    cleaner = Cleaner('english_cleaners', True, 'en-us')
    def ph(t):  # noqa: E306
        return cleaner(t if t[-1] in '.?!' else t + '.')

    print(f'phonemizing {len(general)} general + {len(nav)} navigation + {len(num)} number sentences...')
    cands = [('general', t, p, pair_counts(p)) for t in dict.fromkeys(general) if (p := ph(t))]
    chosen, cover = greedy_cover(cands, n_general_target)
    rows = [cands[i][:3] for i in chosen]
    rows += [('nav', t, ph(t)) for t in dict.fromkeys(nav)]
    rows += [('numbers', t, ph(t)) for t in dict.fromkeys(num)]
    rng.shuffle(rows)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        f.write('id\tkind\ttext\tphonemes\n')
        for i, (kind, t, p) in enumerate(rows):
            f.write(f's{i:06d}\t{kind}\t{t}\t{p}\n')
    all_pairs = Counter()
    for c in cands:
        all_pairs.update(c[3])
    est_h = sum(len(r[2]) for r in rows) / PH_PER_SEC / 3600
    print(f'wrote {args.out}: {len(rows)} sentences (~{est_h:.1f} h), '
          f'{len(cover)}/{len(all_pairs)} phoneme pairs of the candidate pool covered by general text')


if __name__ == '__main__':
    main()
