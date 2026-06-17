#!/usr/bin/env python3
"""Convert a raw ADJSCC ``.pth`` checkpoint into a HuggingFace model folder.
Mirrors ``export_djscc_to_hf.py``.

Example:
    python scripts/export_adjscc_to_hf.py \
        --ckpt checkpoints/adjscc/AWGN_rate_16_AD_JSCC_SNR_random_EP_3.pth \
        --out checkpoints/hf/adjscc-cr6-awgn --comp-ratio 6 \
        --push-to-hub marcellobullo/adjscc-cr6-awgn
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jscc.djscc.codec_djscc import _comp_ratio_to_M  # noqa: E402
from jscc.adjscc.codec_adjscc import _load_adjscc_state  # noqa: E402
from jscc.adjscc.configuration_adjscc import AdjsccConfig  # noqa: E402
from jscc.adjscc.modeling_adjscc import AdjsccModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", required=True, help="raw ADJSCC .pth checkpoint")
    ap.add_argument("--out", default=None, help="output HF model folder")
    ap.add_argument("--comp-ratio", type=float, default=6,
                    help="cr6 = rate16 (tcn 16); cr12 = rate8 (tcn 8)")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--push-to-hub", default=None, help="optional HF repo id to push to")
    args = ap.parse_args()

    cfg = AdjsccConfig(
        M=_comp_ratio_to_M(args.comp_ratio),
        img_height=args.height, img_width=args.width,
    )
    model = AdjsccModel(cfg)
    _load_adjscc_state(model, args.ckpt)

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
