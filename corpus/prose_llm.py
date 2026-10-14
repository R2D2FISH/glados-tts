"""Optional: generate original GLaDOS-tone prose with Claude.

The general profiles want prose to dominate (tens of thousands of lines), and
the hand-written seed set is only ~100 lines. This asks Claude for original
lines across many topics and situations, in the same dry clinical register,
and writes them in the curated-file format so `build` picks them up. Lines
are original; the prompt forbids quoting game scripts.

Two backends:
  claude-code  runs `claude -p` headless (default model claude-sonnet-5-5),
               on whatever account Claude Code is logged in with; no API key.
               Tools, MCP servers and project settings are switched off and
               the coding-agent system prompt is replaced, so each call is a
               plain writing request.
  api          the Anthropic Python SDK (default model claude-opus-5-5); needs
               `pip install anthropic` and ANTHROPIC_API_KEY or `ant auth login`.
"""
import json
import random
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional

DEFAULT_MODEL = {'claude-code': 'claude-sonnet-5-5', 'api': 'claude-opus-5-5'}
SYSTEM = 'You are a comedy writer producing original voice lines for a text-to-speech dataset.'

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


def _ask_claude_code(prompt: str, model: str, timeout: int = 600) -> Optional[str]:
    exe = shutil.which('claude')
    if not exe:
        raise SystemExit('claude CLI not found; install Claude Code or use --backend api')
    proc = subprocess.run(
        [exe, '-p', prompt, '--model', model, '--output-format', 'json',
         '--system-prompt', SYSTEM, '--tools', '', '--strict-mcp-config',
         '--setting-sources', '', '--no-session-persistence'],
        capture_output=True, text=True, timeout=timeout)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        print(f'claude -p failed (exit {proc.returncode}): {proc.stderr.strip()[:300]}')
        return None
    if out.get('is_error'):
        print(f"claude -p error: {str(out.get('result'))[:300]}")
        return None
    return out.get('result', '')


def _ask_api(prompt: str, model: str, effort: str = 'medium') -> Optional[str]:
    import anthropic

    client = anthropic.Anthropic()
    try:
        response = client.beta.messages.create(
            model=model,
            max_tokens=16000,
            system=SYSTEM,
            output_config={'effort': effort},
            # on a policy decline, re-run on a fallback model chosen server-side
            betas=['server-side-fallback-2026-07-01'],
            fallbacks='default',
            messages=[{'role': 'user', 'content': prompt}],
        )
    except anthropic.RateLimitError:
        print('rate limited; skipping batch (re-run to continue)')
        return None
    except anthropic.APIStatusError as e:
        print(f'API error {e.status_code}: {e.message}')
        return None
    if response.stop_reason == 'refusal':
        return None
    return ''.join(block.text for block in response.content if block.type == 'text')


def generate(out_dir: Path, batches: int = 20, per_batch: int = 40, seed: int = 0,
             backend: str = 'claude-code', model: Optional[str] = None, jobs: int = 2) -> Path:
    model = model or DEFAULT_MODEL[backend]
    ask = _ask_claude_code if backend == 'claude-code' else _ask_api
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f'prose_llm_{seed}.txt'
    seen = set()
    if path.exists():
        seen = {l.strip() for l in open(path, encoding='utf-8') if l.strip() and not l.startswith('#')}
    jobs_spec = [(rng.choice(TOPICS), rng.choice(SHAPES)) for _ in range(batches)]

    def run(spec):
        topic, shape = spec
        return spec, ask(PROMPT.format(n=per_batch, topic=topic, shape=shape), model)

    with open(path, 'a', encoding='utf-8') as f, ThreadPoolExecutor(max_workers=jobs) as pool:
        if not seen:
            f.write('# category: prose\n# tags: glados, llm\n')
        for b, ((topic, shape), text) in enumerate(pool.map(run, jobs_spec)):
            if text is None:
                print(f'batch {b + 1}/{batches}: no output ({topic})')
                continue
            new = 0
            for line in _clean(text):
                if line not in seen:
                    seen.add(line)
                    f.write(line + '\n')
                    new += 1
            f.flush()
            print(f'batch {b + 1}/{batches}: {topic} / {shape}: +{new}')
    return path


def _clean(text: str) -> List[str]:
    out = []
    for line in text.splitlines():
        line = re.sub(r'^\s*(?:[-*•]|\d+[.)])\s*', '', line).strip()
        if len(line) > 1 and line[0] in '"“' and line[-1] in '"”':     # whole line quoted
            line = line[1:-1].strip()
        if len(line.split()) >= 2 and not line.startswith('#') and not re.search(r'[*_#<>\[\]{}|]', line):
            out.append(line)
    return out
