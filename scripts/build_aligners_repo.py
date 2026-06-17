#!/usr/bin/env python3
"""Assemble the ``djscc-semantic-aligners`` HuggingFace repo and (optionally) push it.

One repo, many files: each aligner lives in its own subfolder with its weights
(``aligner.pth``) and an ``aligner_config.json`` ({kind, src_arch, tgt_arch,
drop_prob, tcn, ...}) that ``jscc.load_aligner`` reads. Select one at load time
with a subfolder, e.g.:

    --aligner marcellobullo/djscc-semantic-aligners/conv-attn2convnext

Example:
    python scripts/build_aligners_repo.py --push-to-hub marcellobullo/djscc-semantic-aligners
"""

import argparse
import json
import os
import shutil
import sys

import torch

# Maps a source checkpoint -> {subfolder, src_arch, tgt_arch, drop_prob}.
# (aligner_conv.pth is intentionally excluded — unidentified arch pair.)
_MANIFEST = [
    {"src": "aligner_conv_attention-custom.pth",
     "subfolder": "conv-attn2convnext",
     "src_arch": "attention", "tgt_arch": "convnext", "drop_prob": 0.0},
    {"src": "aligner_conv_attention-custom_drop-prob-0-03.pth",
     "subfolder": "conv-attn2convnext-drop03",
     "src_arch": "attention", "tgt_arch": "convnext", "drop_prob": 0.03},
    {"src": "aligner_conv_custom-customnobands_drop-prob-0-03.pth",
     "subfolder": "conv-convnext2convnext-spatialcsi-drop03",
     "src_arch": "convnext", "tgt_arch": "convnext-spatialcsi", "drop_prob": 0.03},
]

_README = """\
# djscc-semantic-aligners

Convolutional **semantic aligners** for the djscc-playground demo. An aligner
fixes the color cast from pairing a TX encoder and an RX decoder that were
trained independently, by mapping the received latent from the encoder's space
into the decoder's space. All are `conv` type (`Conv2d(tcn, tcn, 5)`), applied
at the **receiver**, and trained for `tcn=16` (cr6) models.

Based on *DJSCC Semantic Equalization* (arXiv:2510.04674).

## Aligners

| Subfolder | src encoder -> tgt decoder | erasure robustness |
|-----------|----------------------------|--------------------|
| `conv-attn2convnext` | AD-JSCC (attention) -> custom ConvNeXt | none |
| `conv-attn2convnext-drop03` | AD-JSCC (attention) -> custom ConvNeXt | 3% packet drop |
| `conv-convnext2convnext-spatialcsi-drop03` | custom ConvNeXt -> ConvNeXt spatial-CSI | 3% packet drop |

## Use (receiver only; pair with the matching tgt model)

```bash
python receiver/socket_rx.py --model marcellobullo/djscc-convnext-cr6-awgn \\
    --aligner marcellobullo/djscc-semantic-aligners/conv-attn2convnext
```
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src-dir", default="checkpoints/aligners",
                    help="dir with the raw aligner .pth files")
    ap.add_argument("--out", default="checkpoints/hf/djscc-semantic-aligners",
                    help="assembled repo folder")
    ap.add_argument("--push-to-hub", default=None, help="optional HF repo id")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    built = []
    for entry in _MANIFEST:
        src = os.path.join(args.src_dir, entry["src"])
        if not os.path.isfile(src):
            print(f"[!] skip (missing): {src}")
            continue
        sub = os.path.join(args.out, entry["subfolder"])
        os.makedirs(sub, exist_ok=True)
        shutil.copy(src, os.path.join(sub, "aligner.pth"))

        # Infer tcn from the conv weight shape [out=tcn, in=tcn, k, k].
        state = torch.load(src, map_location="cpu")
        w = next(v for k, v in state.items() if k.endswith("conv.weight"))
        tcn = int(w.shape[0])

        cfg = {"kind": "conv", "src_arch": entry["src_arch"],
               "tgt_arch": entry["tgt_arch"], "drop_prob": entry["drop_prob"],
               "tcn": tcn, "comp_ratio": 6, "applied_at": "receiver"}
        with open(os.path.join(sub, "aligner_config.json"), "w") as f:
            json.dump(cfg, f, indent=2)
        built.append((entry["subfolder"], tcn))
        print(f"[*] {entry['subfolder']}  (tcn={tcn}, drop={entry['drop_prob']})")

    with open(os.path.join(args.out, "README.md"), "w") as f:
        f.write(_README)
    print(f"[*] assembled {len(built)} aligner(s) -> {os.path.abspath(args.out)}")

    if args.push_to_hub:
        from huggingface_hub import HfApi, create_repo
        create_repo(args.push_to_hub, repo_type="model", exist_ok=True, private=True)
        HfApi().upload_folder(folder_path=args.out, repo_id=args.push_to_hub,
                              repo_type="model")
        print(f"[*] pushed -> {args.push_to_hub} (private)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
