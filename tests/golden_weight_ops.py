import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys


def sha(t):
    import torch
    return hashlib.sha256(t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--provider', default='canonical')
    p.add_argument('--plugin-root')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    import comfy_kitchen
    torch.backends.cuda.matmul.allow_tf32 = False
    if a.plugin_root:
        sys.path.insert(0, a.plugin_root)
    private = None if a.provider == 'canonical' else importlib.import_module(a.provider)
    rows = []
    for device in ('cpu', 'cuda'):
        canonical = importlib.import_module('comfy_kitchen.backends.' + ('eager' if device == 'cpu' else 'cuda'))
        backend = private or canonical
        for dtype in (torch.bfloat16, torch.float16, torch.float32):
            for group in (64, 256):
                for seed in (0, 11235):
                    g = torch.Generator(device=device).manual_seed(37042)
                    w = torch.randn((128, 512), dtype=dtype, device=device, generator=g)
                    q, s = backend.quantize_int8_convrot_weight(w, group, stochastic_rounding=seed)
                    decoded = backend.dequantize_int8_convrot_weight_dtype(q, s, group, {torch.float32: 0, torch.float16: 1, torch.bfloat16: 2}[dtype])
                    rows.append({'case':f'{device}/{dtype}/{group}/{seed}', 'q':sha(q), 's':sha(s), 'dequant':sha(decoded), 'finite':bool(torch.isfinite(decoded).all())})
    Path(a.output).write_text(json.dumps({'provider':a.provider,'torch':torch.__version__, 'cases':rows},indent=2),encoding='utf-8')
    print(json.dumps({'provider':a.provider,'cases':len(rows),'all_finite':all(x['finite'] for x in rows)}))


if __name__ == '__main__':
    main()
