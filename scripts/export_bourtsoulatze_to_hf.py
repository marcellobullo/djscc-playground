#!/usr/bin/env python3
"""Initialize / convert a Bourtsoulatze-2019 DeepJSCC baseline into a HuggingFace
model folder. Mirrors ``export_djscc_to_hf.py``.

Unlike the other exporters, ``--ckpt`` is optional: the Bourtsoulatze baseline has
no published djscc-demo checkpoint, so omitting it writes a randomly-initialized
folder — the intended starting point for ``training/train.py`` (train from
scratch; ``--reinit`` is unnecessary since the weights are already random).

Examples:
    # initialize a fresh folder, then train it from scratch
    python scripts/export_bourtsoulatze_to_hf.py \
        --out checkpoints/hf/bourtsoulatze-cr6 --comp-ratio 6

    # convert an already-trained .pth and push
    python scripts/export_bourtsoulatze_to_hf.py \
        --ckpt checkpoints/bourtsoulatze/compratio-6.pth \
        --out checkpoints/hf/bourtsoulatze-cr6 --comp-ratio 6 \
        --push-to-hub marcellobullo/bourtsoulatze-cr6
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from jscc.djscc.codec_djscc import _comp_ratio_to_M, _load_split_state  # noqa: E402
from jscc.bourtsoulatze.configuration_bourtsoulatze import BourtsoulatzeConfig  # noqa: E402
from jscc.bourtsoulatze.modeling_bourtsoulatze import BourtsoulatzeModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None,
                    help="raw .pth checkpoint (optional; omit to init random weights)")
    ap.add_argument("--out", default=None, help="output HF model folder")
    ap.add_argument("--comp-ratio", type=float, default=6)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=768)
    ap.add_argument("--push-to-hub", default=None, help="optional HF repo id to push to")
    args = ap.parse_args()

    cfg = BourtsoulatzeConfig(
        M=_comp_ratio_to_M(args.comp_ratio),
        img_height=args.height, img_width=args.width,
    )
    model = BourtsoulatzeModel(cfg)
    if args.ckpt:
        _load_split_state(model, args.ckpt)
        print(f"[*] loaded weights <- {args.ckpt}")
    else:
        print("[*] no --ckpt: randomly-initialized model (train from scratch)")

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
