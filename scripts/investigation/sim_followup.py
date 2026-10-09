"""Follow-ups:
 (1) CSI clamp: deployed model, true AWGN SNR s, decoder told min(s, c).
 (2) Forensics: which model produced the hardware reconstructions? Compare each
     hardware image at emulator SNR 28 dB with each model's simulated reconstruction
     (noiseless-ish, 20 dB) -> PSNR(hw, sim) should be highest for the model used.
 (3) Gain/phase sensitivity of the deployed decoder (latent * g * e^{j phi}).
"""
import os, sys, glob, json
import numpy as np, cv2, torch
ROOT = "/Users/marcellobullo/research/FINAL/github/djscc-playground"
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "training"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from transformers import AutoModel, AutoConfig
from jscc.djscc_spatialcsi.nn import packetwise_to_element_map

dev = torch.device("mps")
W, H = 768, 512
KODAK = "/Users/marcellobullo/Downloads/kodak_dataset"
imgs = [cv2.cvtColor(cv2.resize(cv2.imread(p), (W, H)), cv2.COLOR_BGR2RGB)
        for p in sorted(glob.glob(f"{KODAK}/kodim*.png"))]
psnr = lambda a, b: 10 * np.log10(255 ** 2 / np.mean((a.astype(float) - b.astype(float)) ** 2))
to_t = lambda im: torch.from_numpy(im).permute(2, 0, 1)[None].float().div(255).to(dev)
to_u8 = lambda y: (y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)


def awgn(z, snr_db, g=1.0):
    zc = z.cpu()
    p = zc.pow(2).mean()
    y = g * zc + torch.randn_like(zc) * torch.sqrt(p / 10 ** (snr_db / 10))
    return y.to(dev)


def rotate_latent(z, phi):
    M = z.shape[1]; re, im = z[:, :M // 2], z[:, M // 2:]
    c, s = np.cos(phi), np.sin(phi)
    return torch.cat([c * re - s * im, s * re + c * im], 1)


def decode(model, mt, z, csi_db):
    B, M, h, w = z.shape
    if mt == "djscc_spatialcsi":
        n = int(np.ceil(M * h * w / 2 / 960))
        m = packetwise_to_element_map(torch.full((B, n), float(csi_db), device=dev), M=M, H=h, W=w, pkt_len_complex=960)
        return model.decoder(z, m)
    if mt == "adjscc":
        return model.decoder(z, torch.full((B, 1), float(csi_db), device=dev))
    return model.decoder(z, torch.full((B, 1), float(csi_db), device=dev))


def encode(model, mt, x, snr):
    if mt == "adjscc":
        from jscc.adjscc.nn import powerConstraint
        z = model.encoder(x, torch.full((1, 1), float(snr), device=dev))
        return powerConstraint(z.flatten(), P=model.power).reshape(z.shape)
    return model.constraint(model.encoder(x))


def load(mid):
    cfg = AutoConfig.from_pretrained(mid, trust_remote_code=True)
    return AutoModel.from_pretrained(mid, trust_remote_code=True).to(dev).eval(), cfg.model_type


out = {}
torch.manual_seed(0)
what = sys.argv[1]

if what == "clamp":
    m, mt = load("marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi")
    res = {}
    for s in [16, 20, 24, 28, 40, 60]:
        for c in [None, 20, 16, 12]:
            ps = []
            for im in imgs:
                with torch.no_grad():
                    z = encode(m, mt, to_t(im), s)
                    y = decode(m, mt, awgn(z, s), s if c is None else min(s, c))
                ps.append(psnr(to_u8(y), im))
            res[f"snr{s}_csi{'true' if c is None else 'clamp' + str(c)}"] = float(np.mean(ps))
            print(s, c, round(float(np.mean(ps)), 2), flush=True)
    out = res

if what == "gain":
    m, mt = load("marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi")
    res = {}
    for g, phi in [(1, 0), (1.03, 0), (0.75, 0), (1.5, 0), (1, np.radians(5)), (1, np.radians(15))]:
        ps = []
        for im in imgs:
            with torch.no_grad():
                z = encode(m, mt, to_t(im), 20)
                y = decode(m, mt, rotate_latent(awgn(z, 20, g), phi), 20)
            ps.append(psnr(to_u8(y), im))
        res[f"g{g}_phi{np.degrees(phi):.0f}"] = float(np.mean(ps))
        print(g, np.degrees(phi), round(float(np.mean(ps)), 2), flush=True)
    out = res

if what == "forensic":
    hw = {}
    for p in sorted(glob.glob(f"{ROOT}/results/djscc-psnr-vs-snr/snr_28dB/*.png")):
        idx = int(os.path.basename(p).split("_")[1].split(".")[0])
        hw[idx] = cv2.cvtColor(cv2.resize(cv2.imread(p), (W, H)), cv2.COLOR_BGR2RGB)
    res = {}
    for mid in ["marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi", "marcellobullo/djscc-convnext-cr6-awgn-spatialcsi",
                "marcellobullo/djscc-convnext-cr6-awgn", "marcellobullo/adjscc-cr6-awgn",
                "marcellobullo/djscc-bourtsoulatze-cr6-ofdm"]:
        m, mt = load(mid)
        best = {}
        for s in [12, 16, 20]:
            ps = []
            for i, im in enumerate(imgs, 1):
                if i not in hw: continue
                with torch.no_grad():
                    z = encode(m, mt, to_t(im), s)
                    y = decode(m, mt, awgn(z, s), s)
                ps.append(psnr(to_u8(y), hw[i]))
            best[s] = float(np.mean(ps))
        res[mid] = best
        print(mid.split("/")[-1], {k: round(v, 2) for k, v in best.items()}, flush=True)
        del m
    out = res

if what == "fixedcsi":
    res = {}
    for mid in ["marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi", "marcellobullo/djscc-convnext-cr6-awgn"]:
        m, mt = load(mid)
        for s in [-12, -9, -6, -3, 0, 4, 8, 12, 16, 20, 24, 28]:
            for c in ["true", 19]:
                ps = []
                for im in imgs:
                    with torch.no_grad():
                        z = encode(m, mt, to_t(im), s)
                        y = decode(m, mt, awgn(z, s), s if c == "true" else c)
                    ps.append(psnr(to_u8(y), im))
                res[f"{mid.split('/')[-1]}|snr{s}|csi{c}"] = float(np.mean(ps))
                print(mid.split('/')[-1], s, c, round(float(np.mean(ps)), 2), flush=True)
        del m
    out = res

if what == "drops":
    m, mt = load("marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi")
    res = {}
    n = 205
    for p in [0.0, 0.01, 0.025, 0.05, 0.1]:
        for informed in [False, True]:
            ps = []
            g = torch.Generator().manual_seed(1)
            for im in imgs:
                with torch.no_grad():
                    z = awgn(encode(m, mt, to_t(im), 20), 20)
                    B, M, h, w = z.shape
                    mask = (torch.rand(1, n, generator=g) < p).float().to(dev)
                    dm = packetwise_to_element_map(mask, M=M, H=h, W=w, pkt_len_complex=960)
                    z = z * (1 - dm)
                    csi = torch.full((1, n), 19.0, device=dev)
                    if informed: csi = csi.masked_fill(mask.bool(), -20.0)
                    y = m.decoder(z, packetwise_to_element_map(csi, M=M, H=h, W=w, pkt_len_complex=960))
                ps.append(psnr(to_u8(y), im))
            res[f"p{p}_informed{informed}"] = float(np.mean(ps)); print(p, informed, round(float(np.mean(ps)), 2), flush=True)
    out = res

if what == "adjscc_mismatch":
    m, mt = load("marcellobullo/adjscc-cr6-awgn")
    res = {}
    for s in [-6, 0, 4, 8, 12, 20, 28]:
        for c in ["true", 19, 10, 5]:
            ps = []
            for im in imgs:
                cs = s if c == "true" else c
                with torch.no_grad():
                    z = encode(m, mt, to_t(im), cs)
                    y = decode(m, mt, awgn(z, s), cs)
                ps.append(psnr(to_u8(y), im))
            res[f"snr{s}|cfg{c}"] = float(np.mean(ps)); print(s, c, round(float(np.mean(ps)), 2), flush=True)
    out = res

json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"followup_{what}.json"), "w"), indent=1)
