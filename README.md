# djscc-playground

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
                        │   BaseCodec adapter      │◀── (any model × any aligner)
                        │  encode / decode / CSI   │
                        └────────────┬────────────┘
        socket_tx.py ───────────────┼─────────────── socket_rx.py
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
  djscc/             custom DJSCC: nn.py, configuration_*.py, modeling_*.py, codec_*.py
  aligners/          aligner modules + BaseAligner wrapper
  conventional/      classical SSCC baseline (output_kind="bytes")  [encode/decode TODO]
transmitter/socket_tx.py    receiver/socket_rx.py
transmitter/gnu_radio/*.grc receiver/gnu_radio/*.grc
scripts/export_djscc_to_hf.py
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

## Publish a model to HuggingFace

```bash
python scripts/export_djscc_to_hf.py \
    --ckpt checkpoints/custom_djscc/compratio-6_latest.pth \
    --out checkpoints/hf/djscc-r6 --comp-ratio 6 \
    --push-to-hub <your-username>/djscc-r6
```

Then anyone runs `--model <your-username>/djscc-r6` — no code changes.

## Add your own model

1. Implement it as a `PreTrainedModel` + `PretrainedConfig` (see `jscc/djscc/`).
2. Add a `codec_*.py` exposing `ENCODER_CODEC` / `DECODER_CODEC` (BaseCodec halves)
   and point `config.codec_module` at it.
3. `push_to_hub`, then `--model <repo>`. Done.
