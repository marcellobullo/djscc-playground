"""Software-only ceilings: DJSCC over ideal channels vs JPEG at the SSCC budgets.

Same preprocessing + PSNR as the slide notebook: Kodak squashed to 768x512 with
cv2.resize, uint8 reconstruction, RGB PSNR, mean over the 24 images.
"""
import io, json, math, os, sys, glob
import numpy as np, cv2, torch
from PIL import Image

ROOT = "/Users/marcellobullo/research/FINAL/github/djscc-playground"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "training"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from train import forward_train, make_channel  # noqa
from transformers import AutoModel, AutoConfig

KODAK = "/Users/marcellobullo/Downloads/kodak_dataset"
W, H = 768, 512
dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
imgs = []
for p in sorted(glob.glob(f"{KODAK}/kodim*.png")):
    bgr = cv2.imread(p)
    imgs.append(cv2.cvtColor(cv2.resize(bgr, (W, H)), cv2.COLOR_BGR2RGB))

def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 10 * np.log10(255.0 ** 2 / mse)

out = {}

class CPUWrap(torch.nn.Module):
    # Kaira's SNR helper casts to float64, unsupported on MPS -> run channel on CPU
    def __init__(self, ch): super().__init__(); self.ch = ch
    def forward(self, z):
        y, csi = self.ch(z.cpu())
        return y.to(z.device), csi.to(z.device)
mode = sys.argv[1] if len(sys.argv) > 1 else "all"

# ---------------------------------------------------------------- JPEG ceilings
if mode in ("all", "jpeg"):
    N_TARGET = 196608
    combos = [(2304, 1152, 1), (2304, 1152, 2), (2304, 1152, 4), (2304, 1728, 1),
              (2304, 1728, 2), (2304, 1920, 1), (2304, 1920, 2), (2304, 1920, 4)]
    jres = {}
    for n, k, bps in combos:
        nb = (N_TARGET * bps + n - 1) // n
        align = 960 * bps // 8
        while (nb * n // 8) % align:
            nb += 1
        budget = nb * k // 8
        syms = nb * n // bps
        ps, bpp = [], []
        for im in imgs:
            pil = Image.fromarray(im)
            lo, hi, best = 1, 95, None
            for _ in range(8):
                if lo > hi: break
                mid = (lo + hi) // 2
                b = io.BytesIO(); pil.save(b, "JPEG", quality=mid, optimize=True)
                if len(b.getvalue()) <= budget: best = b.getvalue(); lo = mid + 1
                else: hi = mid - 1
            if best is None:
                ps.append(float("nan")); continue
            rec = np.array(Image.open(io.BytesIO(best)).convert("RGB"))
            ps.append(psnr(rec, im)); bpp.append(len(best) * 8 / (W * H))
        key = f"LDPC {k}/{n} mod{bps}"
        jres[key] = dict(budget_bytes=budget, channel_symbols=syms,
                         psnr=float(np.nanmean(ps)), bpp=float(np.mean(bpp)))
        print(key, jres[key], flush=True)
    out["jpeg"] = jres

# ---------------------------------------------------------------- DJSCC sims
if mode in ("all", "djscc"):
    models = sys.argv[2].split(",") if len(sys.argv) > 2 else [
        "marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi",
        "marcellobullo/djscc-convnext-cr6-awgn-spatialcsi",
        "marcellobullo/djscc-convnext-cr6-awgn",
        "marcellobullo/adjscc-cr6-awgn",
        "marcellobullo/djscc-bourtsoulatze-cr6-ofdm",
    ]
    snrs = [0, 4, 8, 12, 16, 20, 24, 28, 40]
    channels = ["awgn"] + (["ofdm"] if os.environ.get("WITH_OFDM") else [])
    dres = {}
    torch.manual_seed(0)
    for mid in models:
        cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
        model = AutoModel.from_pretrained(mid, trust_remote_code=True).to(dev).eval()
        mt = cfg.model_type
        r = {}
        for ch in channels + ["none"]:
            for s in (snrs if ch != "none" else [28]):
                ps = []
                for im in imgs:
                    x = torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(dev)
                    chan = make_channel(ch, s) if ch != "none" else None
                    if chan is not None: chan = CPUWrap(chan)
                    with torch.no_grad():
                        y = forward_train(model, mt, x, s, chan, 0.0, 960, -20.0)
                    rec = (y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
                    ps.append(psnr(rec, im))
                r[f"{ch}@{s}"] = float(np.mean(ps))
                print(mid.split("/")[-1], ch, s, round(r[f"{ch}@{s}"], 2), flush=True)
        dres[mid] = r
        del model; torch.mps.empty_cache() if dev.type == "mps" else None
    out["djscc"] = dres

json.dump(out, open(os.path.join(os.path.dirname(__file__), f"sim_{mode}.json"), "w"), indent=1)
