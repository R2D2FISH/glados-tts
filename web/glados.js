// In-browser GLaDOS TTS on onnxruntime-web (WebGPU with WASM fallback).
import { Phonemizer, textToTokens, splitSentences } from './text.js';

const ORT_VERSION = '1.30.0';
const ORT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.mjs`;
export const SAMPLE_RATE = 22050;

export const MODEL_FILES = {
  tacotron: 'glados-fp16.onnx',
  phonemizer: 'phonemizer-int8.onnx',
  phonemizerDict: 'phonemizer.json',
  vocoder: { lq: 'vocoder-lq.onnx', hq: 'vocoder.onnx', 'lq-fp16': 'vocoder-lq-fp16.onnx', 'hq-fp16': 'vocoder-fp16.onnx' },
  speaker: { p1: 'speaker_p1.bin', p2: 'speaker_p2.bin' },
};

const CACHE_NAME = 'glados-tts-models-v1';

// Fetch with download progress, keeping a copy in the Cache API so repeat
// visits (and offline use) skip the ~90 MB download.
async function fetchBytes(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open(CACHE_NAME);
    const hit = await cache.match(url);
    if (hit) {
      const buf = new Uint8Array(await hit.arrayBuffer());
      onProgress?.(buf.length, buf.length);
      return buf;
    }
  } catch { cache = null; }

  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get('content-length') || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got, total);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  try { await cache?.put(url, new Response(buf)); } catch { /* quota or private mode */ }
  return buf;
}

// {available, f16}: many mobile GPUs (and software adapters) lack shader-f16,
// and fp16 models fail on them, so pick weights per device.
export async function webgpuInfo() {
  try {
    const a = navigator.gpu && await navigator.gpu.requestAdapter();
    return { available: !!a, f16: !!a?.features.has('shader-f16') };
  } catch { return { available: false, f16: false }; }
}

export class GladosTTS {
  /**
   * @param {object} opts
   * @param {string} opts.base       URL prefix where the onnx/ files are served
   * @param {'webgpu'|'wasm'} opts.device   where the vocoder runs
   * @param {'webgpu'|'wasm'} [opts.tacotronDevice]  defaults to wasm: its GRU/LSTM
   *        layers have no WebGPU kernels, so running it on the GPU just adds copies
   * @param {'lq'|'hq'} opts.vocoder
   * @param {(msg: string, frac: number) => void} [opts.onProgress]
   */
  static async create({ base = './onnx/', device = 'webgpu', tacotronDevice = 'wasm', vocoder = 'lq', onProgress } = {}) {
    const ort = await import(ORT_URL);
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 1) : 1;

    const f16 = device === 'webgpu' && (await webgpuInfo()).f16;
    const vocFile = MODEL_FILES.vocoder[f16 ? `${vocoder}-fp16` : vocoder];
    const files = [MODEL_FILES.tacotron, MODEL_FILES.phonemizer, MODEL_FILES.phonemizerDict, vocFile,
      MODEL_FILES.speaker.p1, MODEL_FILES.speaker.p2];
    const loaded = {}, progress = {};
    const report = () => {
      let got = 0, total = 0;
      for (const [g, t] of Object.values(progress)) { got += g; total += t || g; }
      onProgress?.(`Downloading models: ${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB`, total ? got / total : 0);
    };
    await Promise.all(files.map(async (f) => {
      loaded[f] = await fetchBytes(base + f, (g, t) => { progress[f] = [g, t]; report(); });
    }));

    onProgress?.('Creating sessions…', 1);
    const opts = (ep) => ({
      executionProviders: ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'],
      graphOptimizationLevel: 'all',
    });
    // Sequential on purpose: concurrent creates race ORT's one-time WebGPU init
    // and it silently drops the webgpu provider.
    const voc = await ort.InferenceSession.create(loaded[vocFile], opts(device));
    const taco = await ort.InferenceSession.create(loaded[MODEL_FILES.tacotron], opts(tacotronDevice));
    const phon = await ort.InferenceSession.create(loaded[MODEL_FILES.phonemizer], opts('wasm'));
    const dict = JSON.parse(new TextDecoder().decode(loaded[MODEL_FILES.phonemizerDict]));
    const f32 = (b) => new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength));
    const tts = new GladosTTS(ort, taco, voc, phon, dict, {
      p1: f32(loaded[MODEL_FILES.speaker.p1]), p2: f32(loaded[MODEL_FILES.speaker.p2]),
    });
    tts.device = device + (device === 'webgpu' ? (f16 ? ' (fp16)' : ' (fp32)') : '');
    tts.vocoderName = vocoder;
    onProgress?.('Warming up…', 1);
    await tts.synthesize('Hello.');
    return tts;
  }

  constructor(ort, taco, voc, phon, dict, speakers) {
    this.ort = ort;
    this.taco = taco;
    this.voc = voc;
    this.speakers = speakers;
    this.phonemizer = new Phonemizer(dict, async (ids) => {
      const t = new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]);
      const out = await phon.run({ text: t });
      return out.logits;
    });
  }

  /** Synthesize one sentence. Returns {audio: Float32Array, timings}. */
  async synthesize(text, { speaker = 'p2', speed = 1.0 } = {}) {
    const { ort } = this;
    const t0 = performance.now();
    const { ids, phonemes } = await textToTokens(text, this.phonemizer);
    const t1 = performance.now();
    if (ids.length === 0) return { audio: new Float32Array(0), phonemes, timings: {} };
    const { mel } = await this.taco.run({
      tokens: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, ids.length]),
      speaker: new ort.Tensor('float32', this.speakers[speaker], [1, 256]),
      alpha: new ort.Tensor('float32', new Float32Array([speed]), [1]),
    });
    const t2 = performance.now();
    const { audio } = await this.voc.run({ mel });
    const samples = await audio.getData();
    const t3 = performance.now();
    return {
      audio: samples,
      phonemes,
      timings: { frontend: t1 - t0, tacotron: t2 - t1, vocoder: t3 - t2, total: t3 - t0, seconds: samples.length / SAMPLE_RATE },
    };
  }

  /** Synthesize sentence by sentence, yielding each as soon as it's ready. */
  async *stream(text, opts) {
    for (const s of splitSentences(text)) yield { sentence: s, ...(await this.synthesize(s, opts)) };
  }
}

export function encodeWav(samples, sampleRate = SAMPLE_RATE) {
  const buf = new ArrayBuffer(44 + samples.length * 2);
  const v = new DataView(buf);
  const str = (o, s) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)));
  str(0, 'RIFF'); v.setUint32(4, 36 + samples.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sampleRate, true); v.setUint32(28, sampleRate * 2, true);
  v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  str(36, 'data'); v.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, samples[i])) * 32767, true);
  }
  return new Blob([buf], { type: 'audio/wav' });
}
