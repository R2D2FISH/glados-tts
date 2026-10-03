"""Export a trained student to onnx/vocoder-nsf.onnx (+ fp16), same mel -> audio
interface as the other vocoders, so glados_onnx.py and the web demo can use it.

    python -m tools.distill.export distill/run/latest.pt
"""
import argparse
import inspect
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
from tools.distill.nsf_vocoder import ExportWrapper, NSFGenerator  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('checkpoint')
    ap.add_argument('--out', default='onnx/vocoder-nsf.onnx')
    args = ap.parse_args()

    s = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    G = NSFGenerator(s['preset'])
    G.load_state_dict(s['G'])
    G.eval().remove_weight_norm()
    model = ExportWrapper(G).eval()

    legacy = {'dynamo': False} if 'dynamo' in inspect.signature(torch.onnx.export).parameters else {}
    mel = torch.randn(1, 80, 200)
    torch.onnx.export(model, (mel,), args.out, input_names=['mel'], output_names=['audio'],
                      dynamic_axes={'mel': {2: 'T'}, 'audio': {2: 'S'}}, opset_version=17, **legacy)

    import onnx
    import onnxruntime as ort
    from onnxruntime.transformers.float16 import convert_float_to_float16
    fp16 = args.out.replace('.onnx', '-fp16.onnx')
    onnx.save(convert_float_to_float16(onnx.load(args.out), keep_io_types=True), fp16)

    # check the exported graph against PyTorch at a different length than traced
    mel = torch.randn(1, 80, 123) - 4
    with torch.no_grad():
        ref = model(mel).numpy()
    out = ort.InferenceSession(args.out).run(None, {'mel': mel.numpy()})[0]
    print(f'{args.out}: step {s.get("step")}, preset {s["preset"]}, max |onnx - torch| = {np.abs(out - ref).max():.2e}')
    print(f'{fp16}: written')


if __name__ == '__main__':
    main()
