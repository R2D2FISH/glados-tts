"""Optional: generate original GLaDOS-tone prose with the Claude API.

The general profiles want prose to dominate (tens of thousands of lines), and
the hand-written seed set is only ~100 lines. This asks Claude for original
lines across many topics and situations, in the same dry clinical register,
and writes them in the curated-file format so `build --extra-dir` picks them
up. Lines are original; the prompt forbids quoting game scripts.

Needs `pip install anthropic` and credentials (ANTHROPIC_API_KEY or an
`ant auth login` profile).
"""
import random
import re
from pathlib import Path
from typing import List

MODEL = 'claude-opus-5-5'

TOPICS = [
    'giving driving directions', 'a traffic jam', 'getting lost', 'arriving at a destination', 'a road trip in Hawaii',
    'morning alarms', 'the weather', 'cooking dinner', 'a smart home', 'grocery shopping', 'doing laundry',
    'a science experiment', 'safety training', 'a fire drill', 'a job interview', 'a birthday party',
    'learning to play piano', 'gardening', 'a pet cat', 'a broken printer', 'airport security', 'a museum tour',
    'astronomy', 'chemistry class', 'a power outage', 'exercise and fitness', 'a dentist appointment',
    'a camping trip', 'reading a bedtime story', 'a sports game', 'a spelling bee', 'a parade',
    'the ocean', 'volcanoes', 'robots', 'a library', 'a train station', 'a long elevator ride',
    'a customer service call', 'a software update', 'a museum of failed inventions', 'the history of cake',
]
SHAPES = [
    'short one-sentence quips', 'two- or three-sentence passages', 'questions addressed to the listener',
    'lists with commas and semicolons', 'lines with numbers, times, dates and measurements written as words or digits',
    'long, winding sentences with parenthetical asides', 'exclamations and interruptions',
    'calm, slow, deliberate announcements with pauses marked by ellipses',
]

PROMPT = """Write {n} original lines of dialogue for a calm, dry, clinical, passive-aggressive AI narrator \
who runs a research facility and now also helps with everyday life. The humor is deadpan and understated; \
she is polite on the surface and quietly insulting underneath.

Topic: {topic}
Form: {shape}

Requirements:
- Every line must be original. Do not quote or paraphrase lines from any video game, film or book.
- Vary sentence length, rhythm and punctuation. Use plain American English words a TTS system can read aloud.
- No stage directions, emoji, markdown, speaker names or numbering.
- One line per output line, nothing else."""


def generate(out_dir: Path, batches: int = 20, per_batch: int = 40, seed: int = 0, effort: str = 'medium') -> Path:
    import anthropic

    client = anthropic.Anthropic()
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'prose_llm_{seed}.txt'
    seen = set()
    if path.exists():
        seen = {l.strip() for l in open(path, encoding='utf-8') if l.strip() and not l.startswith('#')}
    with open(path, 'a', encoding='utf-8') as f:
        if not seen:
            f.write('# category: prose\n# tags: glados, llm\n')
        for b in range(batches):
            topic, shape = rng.choice(TOPICS), rng.choice(SHAPES)
            try:
                response = client.beta.messages.create(
                    model=MODEL,
                    max_tokens=16000,
                    output_config={'effort': effort},
                    # on a policy decline, re-run on a fallback model chosen server-side
                    betas=['server-side-fallback-2026-07-01'],
                    fallbacks='default',
                    messages=[{'role': 'user', 'content': PROMPT.format(n=per_batch, topic=topic, shape=shape)}],
                )
            except anthropic.RateLimitError:
                print('rate limited; stopping early (re-run to continue)')
                break
            except anthropic.APIStatusError as e:
                print(f'batch {b}: API error {e.status_code}: {e.message}')
                continue
            if response.stop_reason == 'refusal':
                print(f'batch {b}: declined ({topic}); skipping')
                continue
            text = ''.join(block.text for block in response.content if block.type == 'text')
            new = 0
            for line in _clean(text):
                if line not in seen:
                    seen.add(line)
                    f.write(line + '\n')
                    new += 1
            print(f'batch {b + 1}/{batches}: {topic} / {shape}: +{new}')
    return path


def _clean(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', line).strip().strip('"“”')
        if len(line.split()) >= 2 and not line.startswith('#') and not re.search(r'[*_#<>\[\]{}|]', line):
            out.append(line)
    return out
