#!/usr/bin/env python
"""Plot channel SNR vs. reconstruction PSNR for DJSCC results.

The ``results/`` folder holds one sub-folder per channel SNR
(``snr_0dB``, ``snr_4dB`` ... ``snr_28dB``); each contains the images
reconstructed by the receiver. This script pairs every reconstruction
with its Kodak original, computes PSNR, averages per SNR and plots the
SNR-vs-PSNR curve.

Notes on the data
-----------------
* All reconstructions are landscape (768x512). The portrait Kodak
  originals (kodim04/09/10/17/18/19) were rotated to landscape before
  transmission, so the original is rotated to match before scoring.
* By default (``--match index``) each reconstruction is paired with its
  original by number (``snrX_NNN -> kodimNN``), so a garbled/decode-failed
  frame still scores low PSNR against its *own* original. Originals with no
  matching index (e.g. a stray ``snr12_025``) are skipped. Pass
  ``--match best`` to instead pick the max-PSNR original, which auto-aligns
  numbering/rotation but will mis-attribute badly corrupted frames.

Usage
-----
    python scripts/plot_snr_psnr.py \
        --results results \
        --originals /Users/marcellobullo/Downloads/kodak_dataset
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
from PIL import Image

SNR_DIR_RE = re.compile(r"snr_?(-?\d+)\s*dB", re.IGNORECASE)
RESULT_IDX_RE = re.compile(r"_(\d+)\.png$", re.IGNORECASE)


def load_rgb(path: Path) -> np.ndarray:
    """Load an image as an HxWx3 uint8 array (drops alpha if present)."""
    return np.asarray(Image.open(path).convert("RGB"))


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    """Standard RGB PSNR in dB (MAX=255). Returns inf for identical images."""
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    if mse == 0:
        return float("inf")
    return 10.0 * np.log10((255.0 ** 2) / mse)


def orientations(orig: np.ndarray, target_shape) -> list[tuple[int, np.ndarray]]:
    """Return (rotation_degrees, image) candidates matching target HxW."""
    out = []
    for k, deg in ((0, 0), (1, 90), (2, 180), (3, 270)):
        rot = np.rot90(orig, k)
        if rot.shape[:2] == target_shape[:2]:
            out.append((deg, rot))
    return out


def load_originals(originals_dir: Path) -> dict[str, np.ndarray]:
    originals = {}
    for p in sorted(originals_dir.glob("*.png")):
        originals[p.stem] = load_rgb(p)
    if not originals:
        raise SystemExit(f"No PNG originals found in {originals_dir}")
    return originals


def best_match(recon: np.ndarray, originals: dict[str, np.ndarray]):
    """Find the (name, rotation, psnr) original that best matches recon."""
    best = (None, 0, -np.inf)
    for name, orig in originals.items():
        for deg, cand in orientations(orig, recon.shape):
            val = psnr(recon, cand)
            if val > best[2]:
                best = (name, deg, val)
    return best


def index_match(recon: np.ndarray, idx: int, originals: dict[str, np.ndarray]):
    """Match recon to kodim<idx> (best matching orientation)."""
    name = f"kodim{idx:02d}"
    if name not in originals:
        return (None, 0, -np.inf)
    cands = orientations(originals[name], recon.shape)
    if not cands:
        return (name, 0, -np.inf)
    deg, cand = max(cands, key=lambda dc: psnr(recon, dc[1]))
    return (name, deg, psnr(recon, cand))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parent.parent
    ap.add_argument("--results", type=Path, default=repo / "results",
                    help="Folder containing snr_*dB sub-folders.")
    ap.add_argument("--originals", type=Path,
                    default=Path("/Users/marcellobullo/Downloads/kodak_dataset"),
                    help="Folder with the original Kodak PNGs.")
    ap.add_argument("--match", choices=["index", "best"], default="index",
                    help="How to pair reconstructions with originals. "
                         "'index' maps snrX_NNN -> kodimNN (true pairing, so a "
                         "corrupted frame scores low against its own original). "
                         "'best' picks the max-PSNR original (auto-aligns "
                         "numbering/rotation but mis-attributes garbled frames).")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output plot path (default: <results>/snr_vs_psnr.png).")
    ap.add_argument("--csv", type=Path, default=None,
                    help="Per-image CSV path (default: <results>/snr_vs_psnr.csv).")
    args = ap.parse_args()

    out_plot = args.out or args.results / "snr_vs_psnr.png"
    out_csv = args.csv or args.results / "snr_vs_psnr.csv"

    originals = load_originals(args.originals)
    print(f"Loaded {len(originals)} originals from {args.originals}")

    snr_dirs = []
    for d in sorted(args.results.iterdir()):
        if not d.is_dir():
            continue
        m = SNR_DIR_RE.search(d.name)
        if m:
            snr_dirs.append((int(m.group(1)), d))
    snr_dirs.sort(key=lambda t: t[0])
    if not snr_dirs:
        raise SystemExit(f"No snr_*dB sub-folders found in {args.results}")

    rows = []          # per-image records for the CSV
    summary = []       # (snr, mean_psnr, std_psnr, n)

    for snr, d in snr_dirs:
        vals = []
        print(f"\nSNR {snr:>3} dB  ({d.name})")
        for img_path in sorted(d.glob("*.png")):
            recon = load_rgb(img_path)
            if args.match == "best":
                name, deg, val = best_match(recon, originals)
            else:
                m = RESULT_IDX_RE.search(img_path.name)
                idx = int(m.group(1)) if m else -1
                name, deg, val = index_match(recon, idx, originals)
            if name is None or not np.isfinite(val):
                print(f"  {img_path.name:<18} -> no match, skipped")
                continue
            vals.append(val)
            rows.append({"snr_db": snr, "result": img_path.name,
                         "matched_original": name, "rotation_deg": deg,
                         "psnr_db": round(val, 4)})
            print(f"  {img_path.name:<18} -> {name} (rot {deg:>3}d)  PSNR {val:6.2f} dB")
        if vals:
            arr = np.asarray(vals)
            summary.append((snr, float(arr.mean()), float(arr.std()), len(arr)))

    if not summary:
        raise SystemExit("No PSNR values computed; nothing to plot.")

    # Warn about originals matched more than once at a given SNR (possible
    # mismatch when using --match best).
    if args.match == "best":
        from collections import Counter
        by_snr: dict[int, Counter] = {}
        for r in rows:
            by_snr.setdefault(r["snr_db"], Counter())[r["matched_original"]] += 1
        for snr, c in sorted(by_snr.items()):
            dups = {k: v for k, v in c.items() if v > 1}
            if dups:
                print(f"\n[warn] SNR {snr} dB: originals matched multiple "
                      f"times (check alignment): {dups}")

    # Write per-image CSV.
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["snr_db", "result",
                                          "matched_original", "rotation_deg",
                                          "psnr_db"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote per-image PSNR table -> {out_csv}")

    # Plot.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summary.sort(key=lambda t: t[0])
    snrs = [s[0] for s in summary]
    means = [s[1] for s in summary]
    stds = [s[2] for s in summary]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    # Light scatter of individual images.
    ax.scatter([r["snr_db"] for r in rows], [r["psnr_db"] for r in rows],
               s=14, color="tab:blue", alpha=0.25, label="per-image PSNR")
    ax.errorbar(snrs, means, yerr=stds, marker="o", color="tab:red",
                capsize=3, linewidth=2, label="mean PSNR (±1 std)")
    ax.set_xlabel("Channel SNR (dB)")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("DJSCC: channel SNR vs. reconstruction PSNR")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_plot, dpi=150)
    print(f"Wrote plot -> {out_plot}")

    print("\nSummary (SNR dB -> mean PSNR dB, n):")
    for snr, mean, std, n in summary:
        print(f"  {snr:>3} dB : {mean:6.2f} +/- {std:4.2f}  (n={n})")


if __name__ == "__main__":
    main()
