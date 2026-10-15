# Live alignment dashboard

Interactive demo of the alignment experiment in `alignment-ota.md`, over a **simulated** channel:
the ADJSCC matched-SNR encoder transmits, and you switch the receiver live between

| Key | Receiver | Path |
|-----|----------|------|
| `1` | matched    | ADJSCC → ADJSCC |
| `2` | misaligned | ADJSCC → ConvNeXt-AWGN, no aligner |
| `3` | aligned    | ADJSCC → aligner (5×5 conv) → ConvNeXt-AWGN |
| `4` | target     | ConvNeXt → ConvNeXt (ceiling) |

matched, misaligned and aligned decode the **same** received latent (same noise and lost packets),
so the receiver is the only thing that changes when you switch.

## Run

```bash
python demo/alignment_dashboard/server.py --images data/kodak --port 8765
```

Open http://localhost:8765. From a laptop: `ssh -L 8765:localhost:8765 <gpu-host>` and open the same
URL (the live camera needs `localhost` or https). Needs `fastapi uvicorn python-multipart` on top of
the training requirements. Models default to the paths in `alignment-ota.md`
(`--src-model`, `--tgt-model`, `--aligner` to override; `--channel awgn|rayleigh|rician|ofdm`).

## Keys

`1`–`4` receiver · `A` aligner on/off · `↑`/`↓` SNR · `←`/`→` image · `Space` new noise ·
`C` wipe-compare with the original · `G` all four side by side · hold `O` original ·
`S` PSNR-vs-SNR sweep · `L` live camera · `H` hide the key legend
