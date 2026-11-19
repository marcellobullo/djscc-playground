"""Per-packet diagnostics of the replica: which packets are lost / degraded and why."""
import sys, os, argparse, numpy as np, cv2, glob
sys.argv += [] 
import e2e_link as E
from jscc import load_codec

ap = argparse.ArgumentParser()
ap.add_argument("--snr", type=float, default=28)
ap.add_argument("--images", type=int, default=2)
ap.add_argument("--gap-frames", type=int, default=0, help="idle samples between OFDM frames")
ap.add_argument("--nred", type=int, default=16)
ap.add_argument("--thr", type=float, default=0.65)
ap.add_argument("--cpe", action="store_true")
ap.add_argument("--noise", action="store_true", help="add AWGN (else noiseless)")
a = ap.parse_args()
os.environ.setdefault("HF_HUB_OFFLINE", "1")
enc = load_codec("marcellobullo/djscc-convnext-cr6-ofdm-spatialcsi", "encoder")
exp = enc.expected_complex_items
paths = sorted(glob.glob("/Users/marcellobullo/Downloads/kodak_dataset/kodim*.png"))[:a.images]
air = []
for p in paths:
    rgb = cv2.cvtColor(cv2.resize(cv2.imread(p), (768, 512)), cv2.COLOR_BGR2RGB)
    s = enc.encode(rgb.tobytes()); air.append(s)     # no interleave: slot k == packet k
air = np.concatenate(air)
bb = E.gr_tx(air, 4.0)
fr = (2 + 1 + 20) * 80
nfr = len(bb) // fr
if a.gap_frames:
    bb = np.concatenate([np.concatenate([bb[i*fr:(i+1)*fr], np.zeros(a.gap_frames, np.complex64)]) for i in range(nfr)])
x = np.concatenate([np.zeros(20000, np.complex64), bb, np.zeros(20000, np.complex64)])
act = np.abs(x) > 0
class A: cfo_hz=0; sfo_ppm=0; pn_linewidth=0; iq_amp_db=0; iq_phase_deg=0; dc=0; clip=0; taps=0
rng = np.random.default_rng(0)
y = E.channel(x, act, a.snr if a.noise else 300, A, rng)
# patch chanest noise reduction length
orig = E.digital.ofdm_chanest_vcvc
E.digital.ofdm_chanest_vcvc = lambda s1, s2, n, red, mco, f: orig(s1, s2, n, a.nred, mco, f)
frames = E.gr_rx(y, 3.0, a.thr)
E.digital.ofdm_chanest_vcvc = orig
pns = [f[0] for f in frames]
print(f"frames rx {len(frames)}/{nfr}; unique pn {len(set(pns))}; dup {len(pns)-len(set(pns))}")
snrs, gains, phs, drift = [], [], [], []
for pn, t, vec in frames:
    if pn >= nfr: continue
    r = E.serialize(vec, a.cpe) * 2.0
    s = air[pn*960:(pn+1)*960]
    ph = np.unwrap([np.angle(np.vdot(s[j*48:(j+1)*48], r[j*48:(j+1)*48])) for j in range(20)])
    drift.append(np.degrees(ph[-1]-ph[0]))
    g = np.vdot(s, r) / np.vdot(s, s); e = r - g*s
    snrs.append(10*np.log10(abs(g)**2*np.mean(abs(s)**2)/np.mean(abs(e)**2))); gains.append(abs(g)); phs.append(np.degrees(np.angle(g)))
snrs = np.array(snrs)
print("per-packet SNR_eff: median %.1f  p10 %.1f  p1 %.1f  min %.1f" % (np.median(snrs), np.percentile(snrs,10), np.percentile(snrs,1), snrs.min()))
print("gain median %.3f  p5 %.3f p95 %.3f | phase p5 %.1f p95 %.1f deg" % (np.median(gains), np.percentile(gains,5), np.percentile(gains,95), np.percentile(phs,5), np.percentile(phs,95)))
print("drift over packet (deg): median |d| %.2f  p95 |d| %.2f" % (np.median(np.abs(drift)), np.percentile(np.abs(drift),95)))
missing = sorted(set(range(nfr)) - set(pns))
print("missing pns (first 30):", missing[:30])
