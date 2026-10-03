// Browser port of the Python text front-end (utils/tools.py prepare_text):
//   english_cleaners (unidecode, inflect-style numbers, abbreviations)
//   -> DeepPhonemizer (dictionary lookup, ONNX transformer for unknown words)
//   -> phoneme filter -> token ids.
// It has no DOM dependencies so it can be tested under Node against Python.

// ---------------------------------------------------------------- symbols
const _pad = '_';
const _punctuation = "!'(),.:;? ";
const _special = '-';
const _vowels = 'iyɨʉɯuɪʏʊeøɘəɵɤoɛœɜɞʌɔæɐaɶɑɒᵻ';
const _nonPulmonic = 'ʘɓǀɗǃʄǂɠǁʛ';
const _pulmonic = 'pbtdʈɖcɟkɡqɢʔɴŋɲɳnɱmʙrʀⱱɾɽɸβfvθðszʃʒʂʐçʝxɣχʁħʕhɦɬɮʋɹɻjɰlɭʎʟ';
const _supra = 'ˈˌːˑ';
const _other = 'ʍwɥʜʢʡɕʑɺɧ';
const _diacritics = 'ɚ˞ɫ';
const _extra = ['g', 'ɝ', '̃', '̍', '̥', '̩', '̯', '͡'];
export const PHONEMES = [
  ...(_pad + _punctuation + _special + _vowels + _nonPulmonic + _pulmonic + _supra + _other + _diacritics),
  ..._extra,
];
const PHONEME_ID = new Map(PHONEMES.map((p, i) => [p, i]));

// ---------------------------------------------------------------- unidecode (approximation)
const ASCII_MAP = {
  '‘': "'", '’': "'", '‚': "'", '‛': "'", '“': '"', '”': '"', '„': '"', '‟': '"',
  '–': '-', '—': '--', '―': '--', '‐': '-', '‑': '-', '−': '-', '…': '...', '·': '*',
  '«': '<<', '»': '>>', '×': 'x', '÷': '/', '°': 'deg', 'ß': 'ss', 'æ': 'ae', 'Æ': 'AE',
  'ø': 'o', 'Ø': 'O', 'œ': 'oe', 'Œ': 'OE', 'đ': 'd', 'Đ': 'D', 'ł': 'l', 'Ł': 'L',
  'þ': 'th', 'Þ': 'Th', 'ð': 'd', 'Ð': 'D', '€': 'EUR', '£': 'PS', '¥': 'Y=', ' ': ' ',
};

export function unidecode(text) {
  let out = '';
  for (const ch of text.normalize('NFKD')) {
    const cp = ch.codePointAt(0);
    if (cp < 128) out += ch;
    else if (cp >= 0x300 && cp <= 0x36f) continue; // combining marks
    else if (ch in ASCII_MAP) out += ASCII_MAP[ch];
  }
  return out;
}

// ---------------------------------------------------------------- inflect.number_to_words
const UNIT = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine'];
const TEEN = ['ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen', 'seventeen', 'eighteen', 'nineteen'];
const TEN = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety'];
const MILL = [' ', ' thousand', ' million', ' billion', ' trillion', ' quadrillion', ' quintillion',
  ' sextillion', ' septillion', ' octillion', ' nonillion', ' decillion'];
const ORDINAL = { ty: 'tieth', one: 'first', two: 'second', three: 'third', five: 'fifth', eight: 'eighth', nine: 'ninth', twelve: 'twelfth' };
const ORDINAL_SUFF = new RegExp(`(${Object.keys(ORDINAL).join('|')})$`);
const NTH_SUFF = new Set(['th', 'st', 'nd', 'rd']);

function millfn(i) {
  if (i > MILL.length - 1) throw new RangeError('number out of range');
  return MILL[i];
}
function tenfn(tens, units, mindex = 0) {
  if (tens !== 1) return TEN[tens] + (tens && units ? '-' : '') + UNIT[units] + millfn(mindex);
  return TEEN[units] + MILL[mindex];
}
function hundfn(h, t, u, mindex, args) {
  if (h) {
    const andword = t || u ? ` ${args.andword} ` : '';
    return `${UNIT[h]} hundred${andword}${tenfn(t, u)}${millfn(mindex)}, `;
  }
  if (t || u) return `${tenfn(t, u)}${millfn(mindex)}, `;
  return '';
}

function enword(num, group, args) {
  if (group === 2) {
    num = num.replace(/(\d)(\d)/g, (_, a, b) => {
      const t = +a, u = +b;
      if (t) return `${tenfn(t, u)}, `;
      if (u) return ` ${args.zero} ${UNIT[u]}, `;
      return ` ${args.zero} ${args.zero}, `;
    });
    return num.replace(/(\d)/, (_, a) => (+a ? `${UNIT[+a]}, ` : ` ${args.zero}, `));
  }
  if (/^0*$/.test(num)) return args.zero;
  if (/^0*1$/.test(num)) return args.one;
  num = num.trimStart().replace(/^0+/, '');
  let mill = 0;
  const three = /(\d)(\d)(\d)(?=\D*$)/;
  while (three.test(num)) {
    num = num.replace(three, (_, a, b, c) => hundfn(+a, +b, +c, mill++, args));
  }
  num = num.replace(/(\d)(\d)(?=\D*$)/, (_, a, b) => `${tenfn(+a, +b, mill)}, `);
  num = num.replace(/(\d)(?=\D*$)/, (_, a) => `${UNIT[+a]}${millfn(mill)}, `);
  return num;
}

function subOrd(val) {
  const n = val.replace(ORDINAL_SUFF, (m) => ORDINAL[m]);
  return n === val ? n + 'th' : n;
}

// Mirrors inflect.engine().number_to_words for digit strings (optionally with
// an ordinal suffix), which is all normalize_numbers ever passes it.
export function numberToWords(num, { group = 0, andword = 'and', zero = 'zero', one = 'one' } = {}) {
  const args = { andword, zero, one };
  num = String(num);
  const myord = NTH_SUFF.has(num.slice(-2));
  if (myord) num = num.slice(0, -2);
  let chunk = num.replace(/\D/g, '') || '0';
  chunk = enword(chunk, group, args);
  if (chunk.endsWith(', ')) chunk = chunk.slice(0, -2);
  chunk = chunk.replace(/\s+,/g, ',');
  if (group === 0) chunk = chunk.replace(/, (\S+)\s+$/, ` ${andword} $1`);
  chunk = chunk.replace(/\s+/g, ' ').trim();
  const parts = chunk.split(', ');
  if (myord) parts[parts.length - 1] = subOrd(parts[parts.length - 1]);
  return parts.join(', ');
}

// ---------------------------------------------------------------- utils/text/numbers.py
function expandDollars(_, match) {
  const parts = match.split('.');
  if (parts.length > 2) return match + ' dollars';
  const dollars = parts[0] ? parseInt(parts[0], 10) : 0;
  const cents = parts.length > 1 && parts[1] ? parseInt(parts[1], 10) : 0;
  const du = dollars === 1 ? 'dollar' : 'dollars';
  const cu = cents === 1 ? 'cent' : 'cents';
  if (dollars && cents) return `${dollars} ${du}, ${cents} ${cu}`;
  if (dollars) return `${dollars} ${du}`;
  if (cents) return `${cents} ${cu}`;
  return 'zero dollars';
}

function expandNumber(m) {
  const digits = m.replace(/^0+(?=\d)/, '');
  const num = digits.length < 16 ? parseInt(digits, 10) : Infinity;
  try {
    if (num > 1000 && num < 3000) {
      if (num === 2000) return 'two thousand';
      if (num > 2000 && num < 2010) return 'two thousand ' + numberToWords(String(num % 100), { andword: '' });
      if (num % 100 === 0) return numberToWords(String(Math.floor(num / 100)), { andword: '' }) + ' hundred';
      return numberToWords(digits, { andword: '', zero: 'oh', group: 2 }).replaceAll(', ', ' ');
    }
    return numberToWords(digits, { andword: '' });
  } catch {
    return digits.split('').map((d) => UNIT[+d] || 'zero').join(' ');
  }
}

export function normalizeNumbers(text) {
  text = text.replace(/([0-9][0-9,]+[0-9])/g, (_, g) => g.replaceAll(',', ''));
  text = text.replace(/£([0-9,]*[0-9]+)/g, '$1 pounds');
  text = text.replace(/\$([0-9.,]*[0-9]+)/g, expandDollars);
  text = text.replace(/([0-9]+\.[0-9]+)/g, (_, g) => g.replaceAll('.', ' point '));
  text = text.replace(/[0-9]+(st|nd|rd|th)/g, (m) => numberToWords(m));
  text = text.replace(/[0-9]+/g, expandNumber);
  return text;
}

// ---------------------------------------------------------------- cleaners.py
const ABBREVIATIONS = [
  ['mrs', 'misess'], ['mr', 'mister'], ['dr', 'doctor'], ['st', 'saint'], ['co', 'company'],
  ['jr', 'junior'], ['maj', 'major'], ['gen', 'general'], ['drs', 'doctors'], ['rev', 'reverend'],
  ['lt', 'lieutenant'], ['hon', 'honorable'], ['sgt', 'sergeant'], ['capt', 'captain'],
  ['esq', 'esquire'], ['ltd', 'limited'], ['col', 'colonel'], ['ft', 'fort'],
].map(([a, b]) => [new RegExp(`\\b${a}\\.`, 'gi'), b]);

export function englishCleaners(text) {
  text = unidecode(text);
  text = normalizeNumbers(text);
  for (const [re, rep] of ABBREVIATIONS) text = text.replace(re, rep);
  return text;
}

// ---------------------------------------------------------------- DeepPhonemizer
const DP_PUNCT = '().,:?!/–';
const DP_PUNC_SET = new Set([...DP_PUNCT, '-', ' ']);
const DP_SPLIT = /([().,:?!/– ])/;
const isAlnum = (c) => /[\p{L}\p{N}]/u.test(c);
const pyTitle = (w) => w.toLowerCase().replace(/(^|[^\p{L}])(\p{L})/gu, (_, a, b) => a + b.toUpperCase());

function expandAcronym(word) {
  return word.split('-').map((sub) => {
    const chars = [...sub];
    let out = '';
    chars.forEach((c, i) => {
      out += c;
      const n = chars[i + 1];
      if (n !== undefined && n !== n.toLowerCase() && n === n.toUpperCase()) out += '-';
    });
    return out;
  }).join('-');
}

export class Phonemizer {
  /**
   * @param {object} cfg  contents of onnx/phonemizer.json
   * @param {(ids: number[]) => Promise<{data: Float32Array, dims: number[]}>} runModel
   *        runs phonemizer.onnx on a (1, L) token sequence and returns logits (1, L, V)
   */
  constructor(cfg, runModel) {
    this.dict = cfg.dict;
    this.textIdx = cfg.text_symbols;
    this.phonemeSyms = cfg.phoneme_symbols;
    this.repeats = cfg.char_repeats;
    this.start = cfg.start_index;
    this.end = cfg.end_index;
    this.special = new Set(['_', '<de>', '<en_us>', '<end>']);
    this.runModel = runModel;
    this.cache = new Map();
  }

  lookup(word) {
    if (DP_PUNC_SET.has(word) || word.length === 0) return word;
    for (const w of [word, word.toLowerCase(), pyTitle(word)]) {
      if (Object.prototype.hasOwnProperty.call(this.dict, w)) return this.dict[w];
    }
    return null;
  }

  async predict(word) {
    if (this.cache.has(word)) return this.cache.get(word);
    const ids = [];
    for (const c of word) {
      for (let r = 0; r < this.repeats; r++) {
        const id = this.textIdx[c.toLowerCase()];
        if (id !== undefined) ids.push(id);
      }
    }
    let phons = '';
    if (ids.length) {
      const { data, dims } = await this.runModel([this.start, ...ids, this.end]);
      const [, L, V] = dims;
      const toks = [];
      for (let t = 0; t < L; t++) {
        let best = 0, bv = -Infinity;
        for (let v = 0; v < V; v++) {
          const x = data[t * V + v];
          if (x > bv) { bv = x; best = v; }
        }
        if (best !== 0 && toks[toks.length - 1] !== best) toks.push(best);
      }
      const stop = toks.indexOf(this.end);
      const kept = stop >= 0 ? toks.slice(0, stop + 1) : toks;
      phons = kept.map((t) => this.phonemeSyms[t]).filter((s) => s !== undefined && !this.special.has(s)).join('');
    }
    this.cache.set(word, phons);
    return phons;
  }

  async phonemize(text) {
    const cleaned = [...text].filter((c) => isAlnum(c) || DP_PUNC_SET.has(c)).join('');
    const words = cleaned.split(DP_SPLIT).filter((s) => s.length > 0);
    const out = [];
    for (const word of words) {
      let p = this.lookup(word);
      if (p === null) {
        const subs = expandAcronym(word).split(/([-])/);
        if (subs.length <= 1) {
          p = await this.predict(word);
        } else {
          const parts = [];
          for (const s of subs) {
            const sp = this.lookup(s);
            parts.push(sp !== null ? sp : await this.predict(s));
          }
          p = parts.join('');
        }
      }
      out.push(p);
    }
    return out.join('');
  }
}

// ---------------------------------------------------------------- utils/text/hawaiian.py
const OKINA = 'ʔ';
const OKINA_CHARS = "ʻ‘’'`";
const MACRON = { 'ā': 'a', 'ē': 'e', 'ī': 'i', 'ō': 'o', 'ū': 'u' };
const HAW_CONS = 'hklmnpw' + OKINA;
const DIPHTHONGS = new Set(['ai', 'ae', 'ao', 'au', 'ei', 'eu', 'oi', 'ou', 'iu']);
const HAW_WORD = /^(?:[hklmnpwʔ]?(?:[aeiou]|[āēīōū]))+$/;
const SHORT = { a: ['ɑ', 'ə'], e: ['ɛ', 'ɛ'], i: ['iː', 'i'], o: ['oʊ', 'oʊ'], u: ['uː', 'u'] };
const LONG = { a: 'ɑː', e: 'eɪ', i: 'iː', o: 'oʊ', u: 'uː' };
const DIPH = { ai: 'aɪ', ae: 'aɪ', ao: 'aʊ', au: 'aʊ', ei: 'eɪ', eu: 'ɛu', oi: 'ɔɪ', ou: 'oʊ', iu: 'iu' };
const FORCE_HAWAIIAN = new Set(('hawaii kamehameha waikiki kailua honolulu kona maui kauai oahu molokai hilo lahaina ' +
  'kihei kaneohe manoa makaha waianae wahiawa haleiwa kapalua kaanapali kapaa lihue hana kahului wailuku aloha ' +
  'mahalo pali punahou nuuanu moana kalakaua kuhio kapiolani liliuokalani kaimuki kahala aina ewa kakaako ' +
  'kalihi moiliili makiki hauula laie kahuku pupukea ala').split(' '));

export function hawNormalize(word) {
  return [...word.normalize('NFC').toLowerCase()].map((c) => (OKINA_CHARS.includes(c) ? OKINA : c)).join('');
}

export function isHawaiian(word, englishDict) {
  const w = hawNormalize(word);
  if (!HAW_WORD.test(w)) return false;
  if ([...w].some((c) => c in MACRON) || (w.includes(OKINA) && !w.startsWith(OKINA) && !w.endsWith(OKINA))) return true;
  const plain = w.replaceAll(OKINA, '');
  if (FORCE_HAWAIIAN.has(plain)) return true;
  if (plain.length < 3) return false;
  return !!englishDict && !Object.prototype.hasOwnProperty.call(englishDict, plain);
}

function hawSyllables(w) {
  const syls = [];
  let i = 0;
  while (i < w.length) {
    let onset = '';
    if (HAW_CONS.includes(w[i])) onset = w[i++];
    const c = w[i];
    if (c in MACRON) { syls.push([onset, MACRON[c], true]); i += 1; }
    else if (i + 1 < w.length && DIPHTHONGS.has(w.slice(i, i + 2))) { syls.push([onset, w.slice(i, i + 2), true]); i += 2; }
    else { syls.push([onset, c, false]); i += 1; }
  }
  return syls;
}

function hawStress(syls) {
  const s = new Array(syls.length).fill(0);
  let i = syls.length - 1, first = true;
  while (i >= 0) {
    if (syls[i][2]) { s[i] = first ? 2 : 1; i -= 1; }
    else if (i > 0 && !syls[i - 1][2]) { s[i - 1] = first ? 2 : 1; i -= 2; }
    else { i -= 1; continue; }
    first = false;
  }
  return s;
}

export function hawaiianToPhonemes(word) {
  const syls = hawSyllables(hawNormalize(word));
  const stress = hawStress(syls);
  let out = '', prev = '';
  syls.forEach(([onset, nuc, heavy], k) => {
    if (onset === OKINA) { if (k > 0) out += OKINA; }
    else if (onset === 'w') out += ['i', 'e'].includes(prev.slice(-1)) ? 'v' : 'w';
    else if (onset) out += onset;
    else if (prev) {
      if (prev.slice(-1) === 'i' && nuc[0] !== 'i') out += 'j';
      else if (prev.slice(-1) === 'u' && nuc[0] !== 'u') out += 'w';
    }
    if (nuc.length === 2) out += DIPH[nuc];
    else if (heavy) out += LONG[nuc];
    else out += SHORT[nuc][stress[k] ? 0 : 1];
    prev = nuc;
  });
  return out;
}

// ---------------------------------------------------------------- prepare_text
const SPECIAL_WORD = /[A-Za-zĀĒĪŌŪāēīōūʻ‘’'`]+/g;
const QUOTES = "‘’'`";

export async function textToPhonemes(text, phonemizer, { hawaiianNames = true, lexicon = {} } = {}) {
  if (!/[.?!]$/.test(text)) text += '.';
  const english = async (seg) => (seg ? phonemizer.phonemize(englishCleaners(seg)) : '');
  const parts = [];
  let last = 0;
  for (const m of text.matchAll(SPECIAL_WORD)) {
    const word = m[0];
    let a = 0, b = word.length;
    while (a < b && QUOTES.includes(word[a])) a++;
    while (b > a && QUOTES.includes(word[b - 1])) b--;
    const core = word.slice(a, b);
    if (!core) continue;
    const key = hawNormalize(core);
    let ph = null;
    if (Object.prototype.hasOwnProperty.call(lexicon, key)) ph = lexicon[key];
    else if (hawaiianNames && isHawaiian(core, phonemizer.dict)) ph = hawaiianToPhonemes(core);
    if (ph === null) continue;
    const start = m.index + a;
    parts.push(await english(text.slice(last, start)), ph);
    last = start + core.length;
  }
  parts.push(await english(text.slice(last)));
  const t = [...parts.join('')].filter((c) => PHONEME_ID.has(c)).join('');
  return t.replace(/\s+/g, ' ').trim();
}

export async function textToTokens(text, phonemizer, opts) {
  const phonemes = await textToPhonemes(text, phonemizer, opts);
  return { phonemes, ids: [...phonemes].map((c) => PHONEME_ID.get(c)) };
}

// Rough stand-in for nltk's sent_tokenize, used to stream long inputs.
export function splitSentences(text) {
  const parts = text.replace(/\s+/g, ' ').trim().match(/[^.!?]+(?:[.!?]+["')\]]*|$)/g) || [];
  return parts.map((s) => s.trim()).filter((s) => /[\p{L}\p{N}]/u.test(s));
}
