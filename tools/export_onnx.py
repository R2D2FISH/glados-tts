"""Export the GLaDOS TTS models to ONNX (for onnxruntime / onnxruntime-web + WebGPU).

Run from the repo root:  python tools/export_onnx.py
Writes to onnx/:
  glados.onnx           ForwardTacotron: tokens (1,N) int64, speaker (1,256), alpha (1,) -> mel (1,80,T)
  vocoder.onnx          HiFi-GAN: mel (1,80,T) -> audio (1,1,T*256)
  vocoder-fp16.onnx     fp16 weights/activations, float32 I/O (best for WebGPU)
  vocoder-lq.onnx       small HiFi-GAN recovered from vocoder-cpu-lq.pt (~10x faster)
  vocoder-lq-fp16.onnx
  speaker_p1.bin / speaker_p2.bin   raw float32 speaker embeddings for the browser
  glados-fp16.onnx      fp16 ForwardTacotron (half the download; what the web demo uses)
  phonemizer.onnx       DeepPhonemizer forward transformer: text (1,L) int64 -> logits (1,L,V)
  phonemizer-int8.onnx  int8 phonemizer (only runs for words missing from the dictionary)
  phonemizer.json       its 124k-word en_us dictionary + tokenizer tables
"""
import inspect
import os
import sys
import warnings

import torch
import torch.nn.functional as F

warnings.filterwarnings('ignore')
sys.path.insert(0, os.getcwd())
from tools.forward_tacotron import load_from_torchscript  # noqa: E402
from tools import hifigan_lq  # noqa: E402

OUT = 'onnx'
OPSET = 17
# Newer torch defaults to the dynamo exporter; these graphs need the TorchScript one.
LEGACY = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}


def export_tacotron():
    model, _ = load_from_torchscript()
    x = torch.randint(1, 60, (1, 40))
    semb = torch.load('models/emb/glados_p2.pt', weights_only=False)
    alpha = torch.tensor([1.0])
    torch.onnx.export(
        model, (x, semb, alpha), f'{OUT}/glados.onnx',
        input_names=['tokens', 'speaker', 'alpha'], output_names=['mel'],
        dynamic_axes={'tokens': {1: 'N'}, 'mel': {2: 'T'}},
        opset_version=OPSET, **LEGACY)


def export_vocoder(voc, name):
    torch.onnx.export(
        voc, (torch.randn(1, 80, 200),), f'{OUT}/{name}.onnx',
        input_names=['mel'], output_names=['audio'],
        dynamic_axes={'mel': {2: 'T'}, 'audio': {2: 'S'}},
        opset_version=OPSET, **LEGACY)


def convert_fp16(name):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    m = onnx.load(f'{OUT}/{name}.onnx')
    m16 = convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, f'{OUT}/{name}-fp16.onnx')


def export_phonemizer():
    import json
    from dp.model.model import ForwardTransformer
    ck = torch.load('models/en_us_cmudict_ipa_forward.pt', weights_only=False, map_location='cpu')
    model = ForwardTransformer.from_config(ck['config'])
    model.load_state_dict(ck['model'])
    model.eval()

    class Wrap(torch.nn.Module):
        """Re-states the encoder with explicit attention math. nn.MultiheadAttention
        unpacks the sequence length as a Python int, which tracing freezes into
        the graph. Batch size is always 1, so the padding mask is a no-op."""
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, text):
            m = self.m
            x = m.pos_encoder(m.embedding(text.transpose(0, 1)))[:, 0]  # (L, d)
            for layer in m.encoder.layers:
                attn = layer.self_attn
                h, d = attn.num_heads, attn.head_dim
                q, k, v = F.linear(x, attn.in_proj_weight, attn.in_proj_bias).chunk(3, dim=-1)
                q, k, v = (t.reshape(-1, h, d).transpose(0, 1) for t in (q, k, v))
                a = torch.softmax(q @ k.transpose(1, 2) / d ** 0.5, dim=-1) @ v
                a = attn.out_proj(a.transpose(0, 1).reshape(-1, h * d))
                x = layer.norm1(x + a)
                x = layer.norm2(x + layer.linear2(F.relu(layer.linear1(x))))
            x = m.encoder.norm(x)
            return m.fc_out(x)[None]

    x = torch.tensor([[2] + [10] * 21 + [3]])
    torch.onnx.export(
        Wrap(model), (x,), f'{OUT}/phonemizer.onnx',
        input_names=['text'], output_names=['logits'],
        dynamic_axes={'text': {1: 'L'}, 'logits': {1: 'L'}},
        opset_version=OPSET, **LEGACY)
    pre = ck['preprocessor']
    with open(f'{OUT}/phonemizer.json', 'w', encoding='utf-8') as f:
        json.dump({
            'text_symbols': pre.text_tokenizer.token_to_idx,
            'phoneme_symbols': pre.phoneme_tokenizer.idx_to_token,
            'char_repeats': pre.text_tokenizer.char_repeats,
            'start_index': pre.text_tokenizer._get_start_index('en_us'),
            'end_index': pre.text_tokenizer.end_index,
            'dict': ck['phoneme_dict']['en_us'],
        }, f, ensure_ascii=False, separators=(',', ':'))


def export_speakers():
    for p in ('p1', 'p2'):
        e = torch.load(f'models/emb/glados_{p}.pt', weights_only=False)
        e.detach().float().contiguous().numpy().tofile(f'{OUT}/speaker_{p}.bin')


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    export_tacotron()
    export_vocoder(torch.jit.load('models/vocoder-gpu.pt', map_location='cpu').eval(), 'vocoder')
    export_vocoder(hifigan_lq.load(), 'vocoder-lq')
    convert_fp16('vocoder')
    convert_fp16('vocoder-lq')
    convert_fp16('glados')
    export_speakers()
    export_phonemizer()
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(f'{OUT}/phonemizer.onnx', f'{OUT}/phonemizer-int8.onnx', weight_type=QuantType.QInt8)
    for f in sorted(os.listdir(OUT)):
        print(f'{f:24s} {os.path.getsize(os.path.join(OUT, f)) / 1e6:8.1f} MB')
