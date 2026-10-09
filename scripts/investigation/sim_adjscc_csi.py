"""ADJSCC on an ideal complex AWGN channel (no OFDM chain): PSNR vs true SNR for different encoder+decoder CSI inputs.
Tells whether the model itself prefers the matched SNR."""
import glob, os, sys, json
import numpy as np, cv2, torch
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT); os.environ.setdefault("HF_HUB_OFFLINE", "1")
from jscc import load_codec
from PIL import Image
W, H, PKT = 768, 512, 960
paths = sorted(glob.glob("/Users/marcellobullo/Downloads/kodak_dataset/kodim*.png"))[:int(sys.argv[1]) if len(sys.argv) > 1 else 8]
model = sys.argv[2] if len(sys.argv) > 2 else "marcellobullo/adjscc-cr6-awgn"
enc = load_codec(model, "encoder", packet_len=PKT, snr_db=5); dec = load_codec(model, "decoder", packet_len=PKT, snr_db=5)
exp = enc.expected_complex_items
rgbs = [cv2.cvtColor(cv2.resize(cv2.imread(p), (W, H)), cv2.COLOR_BGR2RGB) for p in paths]
origs = [cv2.resize(np.array(Image.open(p))[..., :3], (W, H)) for p in paths]
def psnr(a, b): return 10 * np.log10(255 ** 2 / np.mean((a.astype(float) - b.astype(float)) ** 2))
snrs = [0, 4, 8, 12, 16, 20, 24, 28]
cfgs = [0, 4, 8, 12, 16, 20, 24, 28]
rng = np.random.default_rng(0)
res = {}
for c in cfgs:
    row = []
    for s in snrs:
        ci = c
        enc.set_csi(ci); dec.set_csi(ci)
        ps = []
        for rgb, o in zip(rgbs, origs):
            sym = enc.encode(rgb.tobytes())
            p = np.mean(np.abs(sym[:exp]) ** 2)
            n = np.sqrt(p / 10 ** (s / 10) / 2) * (rng.standard_normal(len(sym)) + 1j * rng.standard_normal(len(sym)))
            y = (sym + n).astype(np.complex64)
            ps.append(psnr(np.frombuffer(dec.decode(y), np.uint8).reshape(H, W, 3), o))
        row.append(float(np.mean(ps)))
    res[str(c)] = row
    print(f"CSI {str(c):>8}:", " ".join(f"{v:5.1f}" for v in row), flush=True)
os.makedirs(os.path.join(ROOT, "results", "gr-channel", "csi_matrix"), exist_ok=True)
out = sys.argv[3] if len(sys.argv) > 3 else os.path.join(ROOT, "results", "gr-channel", "csi_matrix", "ideal_awgn_matrix.json")
json.dump(dict(snrs=snrs, psnr=res, model=model), open(out, "w"), indent=1)
