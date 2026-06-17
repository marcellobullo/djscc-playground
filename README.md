# DJSCC Playground

A real-channel DJSCC / SSCC image-transmission testbed where **the
encoder/decoder model is chosen at runtime** — by HuggingFace id, local folder,
short alias, or raw checkpoint — without editing any transmit/receive code.

It is the evolution of `djscc-demo` (frozen as the EuCNC release). The physical
layer (GNU Radio OFDM + USRP) and the ZMQ seam are reused unchanged; the model
layer is made pluggable.

## Architecture

```
                model selected at runtime (HF id | folder | alias | .pth)
                                     │
                          jscc.load_codec(...)            jscc.load_aligner(...)
                                     │                            │
                        ┌────────────┴────────────┐   composed at the adapter layer
                        │   BaseCodec adapter     │◀── (any model × any aligner)
                        │  encode / decode / CSI  │
                        └────────────┬────────────┘
                                     |
        socket_tx.py ────────────────┼─────────────── socket_rx.py
              │            ZMQ (cf32 symbols | bytes)         │
        djscc_tx.grc / conventional_tx.grc        djscc_rx.grc / conventional_rx.grc
              │              OFDM + USRP                      │
              └──────────────  real RF link  ─────────────────┘
```

Three layers, each doing what it is best at:

- **Author / train — Kaira.** Models are `kaira.models.BaseModel` subclasses
  composed with `kaira.constraints` / `kaira.channels`. End-to-end, one process.
- **Package — HuggingFace.** `modeling_*.py` is a `PreTrainedModel` whose config
  carries the build recipe. `push_to_hub` ships weights + code + config;
  `from_pretrained(id_or_folder, trust_remote_code=True)` rebuilds it. (Kaira-native:
  loading a repo needs `kaira` installed.)
- **Deploy — `jscc` BaseCodec.** A thin adapter that runs `.encode` on the TX host
  and `.decode` on the RX host, owns the real/complex packing, `packet_len`
  padding and CSI tensor, and substitutes the **real USRP link** for the simulated
  channel. Aligners are composed here, not baked into any model.

## Layout

```
jscc/
  base.py            BaseCodec / BaseEncoderCodec / BaseDecoderCodec / BaseAligner
  loader.py          load_codec(model, role) / load_aligner(spec, ...)
  registry.py        short aliases (e.g. "conventional", "djscc-r6")
  data.py            image-folder dataset + loaders (training)
  djscc/             custom ConvNeXt DJSCC (channel-blind enc + FiLM CSI decoder)
  djscc_spatialcsi/  no-band variant (per-element CSI decoder, anti-banding)
  adjscc/            attention DJSCC (SNR-adaptive encoder + decoder)
  bourtsoulatze/     Bourtsoulatze-2019 DeepJSCC baseline (fixed-SNR, non-adaptive)
  aligners/          aligner modules + BaseAligner wrapper
  conventional/      classical SSCC baseline (output_kind="bytes")  [encode/decode TODO]
transmitter/socket_tx.py    receiver/socket_rx.py
transmitter/gnu_radio/*.grc receiver/gnu_radio/*.grc
training/train.py           one script to train any model (Kaira recipe)
scripts/                    export_*_to_hf.py, build_aligners_repo.py
checkpoints/         (gitignored; copied from djscc-demo)
```

## Quickstart (bundled checkpoint)

```bash
# Receiver (decoder)
python receiver/socket_rx.py \
    --model checkpoints/custom_djscc/compratio-6_latest.pth --comp-ratio 6 \
    --use-live-snr --count 5

# Transmitter (encoder) — needs the djscc_tx.grc flowgraph running, or use --direct-zmq
python transmitter/socket_tx.py \
    --model checkpoints/custom_djscc/compratio-6_latest.pth --comp-ratio 6 \
    --source folder --path /path/to/images
```

Add `--aligner checkpoints/aligners/aligner_conv.pth` to either side to compose a
semantic aligner (must match on both ends).

## Training

One script trains any model. It loads the architecture straight from the Hub (or
a local HF folder) with `AutoModel`, runs the proven recipe — encoder → power
constraint → Kaira channel (+ optional packet drop) → decoder, MSE loss, random
or fixed SNR — and saves the result with `save_pretrained`. The output is itself
an `AutoModel`-loadable model, so there is **no separate export step**: train,
then deploy or `push_to_hub` the same folder.

```bash
# fine-tune a published model over SNR ~ U[0,20] dB with 10% packet drops
python training/train.py --model marcellobullo/djscc-convnext-cr6-awgn \
    --train-data-dir /data/DIV2K_train_HR --val-data-dir /data/DIV2K_valid_HR \
    --channel awgn --snr-min 0 --snr-max 20 --drop-prob 0.10 \
    --epochs 50 --batch-size 4 --out ckpts/run

# deploy the result immediately (or add --push-to-hub <repo> to the command above)
python receiver/socket_rx.py --model ckpts/run/best --use-live-snr
```

Key options:

- `--model` — HF id or local HF folder to start from. Add `--reinit` to train from
  scratch (keeps the architecture, randomizes the weights).
- `--channel awgn|rayleigh|rician|ofdm|none` — the channel (`ofdm` =
  frequency-selective multipath + per-subcarrier equalization, see
  `jscc/channels.py`; the others are Kaira channels). OFDM tuning:
  `--ofdm-subcarriers` (match the link's `fft_len`), `--ofdm-taps`, `--ofdm-decay`,
  `--ofdm-eq zf|mmse`.
- `--snr-db X` (fixed) **or** `--snr-min A --snr-max B` (sampled `U[A,B]` per batch).
- `--drop-prob P` — fixed per-packet erasure probability (each packet dropped
  i.i.d. w.p. `P`). **Or** `--drop-min A --drop-max B` to sample the erasure rate
  `U[A,B]` per batch — the drop-side analogue of `--snr-min/--snr-max`, so the
  decoder sees a range of erasure conditions.
- `--loss <name>` — training loss from `kaira.losses.image` (see **Losses**
  below). Validation always reports **PSNR + MS-SSIM + LPIPS** regardless.
- `--resume ckpts/run/last.pth`, `--push-to-hub <repo>`.

`--train-data-dir` (and optional `--val-data-dir`) take one or more flat image
folders (DIV2K-style). Per-family conditioning — scalar SNR (`djscc`), attention
SNR (`adjscc`), or the per-element CSI map with drops → sentinel SNR
(`djscc_spatialcsi`) — is selected automatically from the model's
`config.model_type`.

### Losses

All values accepted by `--loss` (from `kaira.losses.image`; each returns a single
minimizable scalar). Regardless of which you train with, validation always reports
PSNR + MS-SSIM + LPIPS.

| `--loss`    | Kaira class    | Optimizes / when to use |
|-------------|----------------|-------------------------|
| `mse`       | `MSELoss`      | Pixel MSE. Best PSNR; the default. Tends to blur texture at low rate. |
| `l1`        | `L1Loss`       | Mean absolute error. Sharper than MSE, more robust to outliers. |
| `ssim`      | `SSIMLoss`     | Single-scale structural similarity (luminance/contrast/structure). |
| `msssim`    | `MSSSIMLoss`   | Multi-scale SSIM. Best perceived structure / MS-SSIM score. |
| `vgg`       | `VGGLoss`      | VGG feature-space (perceptual); lighter-weight than LPIPS. |
| `lpips`     | `LPIPSLoss`    | Learned perceptual (deep features). Most perceptual realism, but can hallucinate texture; downloads a small net on first use. |
| `mse_lpips` | `MSELPIPSLoss` | Weighted MSE + LPIPS — fidelity *and* perceptual. Tune with `--mse-weight` / `--lpips-weight` (both default `1.0`). |

`--lpips-net alex|vgg|squeeze` (default `alex`) selects the backbone for the LPIPS
*validation metric* — independent of the training loss.

**Train a brand-new architecture:** copy `jscc/djscc/` as a template, implement
your `nn.py` + `configuration_*.py` / `modeling_*.py` / `codec_*.py`,
`save_pretrained` an initialized model once, then
`python training/train.py --model <that folder> --reinit ...`.

## Publish a model to HuggingFace

Each model family has an exporter in `scripts/` that turns a raw `.pth` checkpoint
into an `AutoModel`-loadable HF folder (`config.json` + weights + bundled
modeling/config code, `kaira` required to load) and optionally pushes it:

| Family | Script | `model_type` |
|--------|--------|--------------|
| Custom ConvNeXt DJSCC (scalar FiLM CSI) | `export_djscc_to_hf.py`            | `djscc` |
| Spatial-CSI / no-band variant           | `export_djscc_spatialcsi_to_hf.py` | `djscc_spatialcsi` |
| Attention SNR-adaptive (ADJSCC)         | `export_adjscc_to_hf.py`           | `adjscc` |
| Bourtsoulatze-2019 baseline             | `export_bourtsoulatze_to_hf.py`    | `deepjscc_b2019` |

```bash
python scripts/export_djscc_to_hf.py \
    --ckpt checkpoints/custom_djscc/compratio-6_latest.pth \
    --out checkpoints/hf/djscc-r6 --comp-ratio 6 \
    --push-to-hub <your-username>/djscc-r6
```

Then anyone runs `--model <your-username>/djscc-r6` — no code changes.

`export_bourtsoulatze_to_hf.py` is the exception: `--ckpt` is **optional** — the
baseline has no published checkpoint, so omitting it writes a randomly-initialized
folder to train from scratch (no `--reinit` needed):

```bash
python scripts/export_bourtsoulatze_to_hf.py \
    --out checkpoints/hf/bourtsoulatze-cr6 --comp-ratio 6
```

## Add your own model

1. Implement it as a `PreTrainedModel` + `PretrainedConfig` (see `jscc/djscc/`).
2. Add a `codec_*.py` exposing `ENCODER_CODEC` / `DECODER_CODEC` (BaseCodec halves)
   and point `config.codec_module` at it.
3. `push_to_hub`, then `--model <repo>`. Done.
