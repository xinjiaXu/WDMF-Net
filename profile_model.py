import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

from models.WDMFNet import BaseNet


def main():
    parser = argparse.ArgumentParser(description="Profile WDMF-Net parameters and FLOPs.")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    model = BaseNet(pretrained=False).eval().to(device)
    inputs = tuple(
        torch.randn(1, 3, args.height, args.width, device=device)
        for _ in range(2)
    )

    flop_error = None
    unsupported = {}
    try:
        analysis = FlopCountAnalysis(model, inputs)
        flops = analysis.total()
        unsupported = dict(Counter(analysis.unsupported_ops()))
    except Exception as exc:
        flops = None
        flop_error = f"{type(exc).__name__}: {exc}"

    params = sum(param.numel() for param in model.parameters())
    result = {
        "model": "WDMF-Net-BaseNet",
        "input": "two RGB images, batch=1",
        "image_size": [args.height, args.width],
        "parameters_total": params,
        "size_m": params / 1_000_000,
        "fvcore_flops_per_pair": flops,
        "flops_g": flops / 1_000_000_000 if flops is not None else None,
        "fvcore_convention": "one multiply-add is counted as one FLOP",
        "unsupported_ops": unsupported,
        "fvcore_error": flop_error,
        "note": "ImageNet encoder weights are not needed for profiling. FLOPs are partial if unsupported_ops is non-empty.",
    }
    print(json.dumps(result, indent=2))
    print(f"Size(M): {result['size_m']:.3f}")
    if result["flops_g"] is None:
        print("Flops(G): unavailable; see fvcore_error")
    else:
        print(f"Flops(G): {result['flops_g']:.3f}")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
