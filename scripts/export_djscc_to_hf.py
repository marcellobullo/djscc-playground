#!/usr/bin/env python3
"""Convert a raw djscc-demo ``.pth`` checkpoint into a HuggingFace model folder.

The output folder loads with ``AutoModel.from_pretrained(folder,
trust_remote_code=True)`` and can be published with ``--push-to-hub <repo>``. It
contains ``config.json`` (with ``auto_map``), the weights, and the custom
modeling/config/nn code (Kaira must be installed to load it).

Example:
    python scripts/export_djscc_to_hf.py \
        --ckpt checkpoints/custom_djscc/compratio-6_latest.pth \
        --out  checkpoints/hf/djscc-r6 --comp-ratio 6
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jscc.djscc.configuration_djscc import DJSCCConfig          # noqa: E402
from jscc.djscc.modeling_djscc import DJSCCModel                # noqa: E402
from jscc.djscc.codec_djscc import _comp_ratio_to_M, _load_split_state  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="raw .pth checkpoint")
    ap.add_argument("--out", default=None, help="output HF model folder")
    ap.add_argument("--comp-ratio", type=float, default=6)
    ap.add_argument("--N", type=int, default=256)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--push-to-hub", default=None,
                    help="optional HF repo id to push to")
    args = ap.parse_args()

    cfg = DJSCCConfig(
        N=args.N, M=_comp_ratio_to_M(args.comp_ratio),
        img_height=args.height, img_width=args.width,
    )
    model = DJSCCModel(cfg)
    _load_split_state(model, args.ckpt)

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        model.save_pretrained(args.out)
        print(f"[*] saved HF model -> {os.path.abspath(args.out)}")

    if args.push_to_hub:
        model.push_to_hub(args.push_to_hub)
        print(f"[*] pushed -> {args.push_to_hub}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
