# Over-the-air alignment experiment

The transmitter runs the **matched-SNR ADJSCC** encoder; the receiver decodes with a
**different** model, the **ConvNeXt-AWGN** decoder, through a small learned **aligner**
applied to the received latent. Four receiver runs on the same images and the same radio
setup tell how much of the mismatch the aligner recovers:

| Run | TX encoder | RX decoder | Role |
|-----|------------|------------|------|
| **matched** | ADJSCC matched-SNR | ADJSCC matched-SNR | quality of the source model on this link |
| **aligned** | ADJSCC matched-SNR | ConvNeXt-AWGN + aligner | the experiment |
| **misaligned** | ADJSCC matched-SNR | ConvNeXt-AWGN, no aligner | lower bound |
| **target** | ConvNeXt-AWGN | ConvNeXt-AWGN | ceiling for the aligned run |

The score is PSNR / SSIM against the originals and the **Alignment Recovery Ratio**
`ARR = (aligned - misaligned) / (target - misaligned)`. `scripts/eval_ota_alignment.py`
uses the matched run as the reference, so pass the target folder as `matched=` when you want
the ARR against the target ceiling (see step 5).

## Models

| What | Where |
|------|-------|
| ADJSCC matched-SNR (encoder + decoder take the true SNR) | `marcellobullo/adjscc-matched-snr`, folder `best/` |
| Aligner ADJSCC matched-SNR -> ConvNeXt-AWGN | `marcellobullo/adjscc-matched-snr`, folder `aligner-convnext-awgn/` |
| ConvNeXt-AWGN | `marcellobullo/djscc-convnext-cr6-awgn` (downloaded on first use) |

The aligner is a single 5x5 convolution on the 16-channel latent (`--kind conv`), applied
at the receiver only. The transmitter does not need it.

## Requirements

- Branch `feat/snr-estimation-matched-adjscc` on all machines (TX/RX flowgraphs with the
  frame gap, CFO-ramp corrector and pilot SNR logger; pilot-based live SNR in `socket_rx.py`;
  `--init-aligner` in `train_aligner.py`; the scoring script).
- Radio laptops: the GNU Radio environment from the main README (`demo`), plus
  `pip install compressai` (ADJSCC layers).
- Training machine (GPU): Python 3.11 with
  `pip install torch torchvision pykaira==0.2.1 compressai "transformers==5.12.1" huggingface_hub safetensors "numpy<2" pillow opencv-python-headless scipy torchmetrics pyzmq`
  (the Kaira package is called `pykaira` on PyPI) and DIV2K in `data/`:
  ```bash
  mkdir -p data && cd data && wget http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_train_HR.zip http://data.vision.ee.ethz.ch/cvl/DIV2K/DIV2K_valid_HR.zip && unzip -q DIV2K_train_HR.zip && unzip -q DIV2K_valid_HR.zip && cd ..
  ```
- If the Hugging Face repo is private: `hf auth login` (or
  `python -c "from huggingface_hub import login; login()"`) on every machine.

## 1. Train the aligner (GPU machine, optional)

Skip this if `aligner-convnext-awgn/` is already on Hugging Face. Download the source model
first if the machine does not have it:

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('marcellobullo/adjscc-matched-snr', allow_patterns=['best/*'], local_dir='ckpts/adjscc-matched-snr-final')"
```

Train end-to-end through the frozen ConvNeXt-AWGN decoder over an AWGN channel at 25 dB
(this worked better over the air than a 0-28 dB range), with up to 10 % lost packets:

```bash
python training/train_aligner.py --src-model ckpts/adjscc-matched-snr-final/best --tgt-model marcellobullo/djscc-convnext-cr6-awgn --kind conv --channel awgn --snr-db 25 --drop-min 0 --drop-max 0.1 --loss mse --train-data-dir data/DIV2K_train_HR --val-data-dir data/DIV2K_valid_HR --epochs 30 --out ckpts/aligner-adjscc2convnextawgn
```

To continue training an aligner (weights only; the optimizer and the cosine schedule start
again; the run validates the starting aligner first and only saves when it is beaten):

```bash
python training/train_aligner.py --src-model ckpts/adjscc-matched-snr-final/best --tgt-model marcellobullo/djscc-convnext-cr6-awgn --kind conv --channel awgn --snr-db 25 --drop-min 0 --drop-max 0.1 --loss mse --train-data-dir data/DIV2K_train_HR --val-data-dir data/DIV2K_valid_HR --epochs 30 --lr 5e-5 --init-aligner ckpts/aligner-adjscc2convnextawgn --out ckpts/aligner-adjscc2convnextawgn-cont
```

Use another GPU with `CUDA_VISIBLE_DEVICES=1 python training/train_aligner.py ...`.
Upload the aligner you keep:

```bash
hf upload marcellobullo/adjscc-matched-snr ckpts/aligner-adjscc2convnextawgn aligner-convnext-awgn
```

## 2. Download the models (radio laptops)

TX laptop (the source model; the ConvNeXt model is only needed for the target run):

```bash
python -c "from huggingface_hub import snapshot_download as d; d('marcellobullo/adjscc-matched-snr', allow_patterns=['best/*'], local_dir='ckpts/adjscc-matched-snr-final'); d('marcellobullo/djscc-convnext-cr6-awgn')"
```

RX laptop (source model, aligner and ConvNeXt model):

```bash
python -c "from huggingface_hub import snapshot_download as d; d('marcellobullo/adjscc-matched-snr', allow_patterns=['best/*','aligner-convnext-awgn/*'], local_dir='ckpts/adjscc-matched-snr-final'); d('marcellobullo/djscc-convnext-cr6-awgn')"
```

Check on the RX: `ls ckpts/adjscc-matched-snr-final/best ckpts/adjscc-matched-snr-final/aligner-convnext-awgn`
must show `config.json`, `model.safetensors`, `aligner.pth` and `aligner_config.json`.

## 3. Start the radios

Each laptop runs the flowgraph in one terminal and the socket script in another. Change the
USRP addresses if yours differ (`--carrier-freq` defaults to 2.45 GHz). Keep the gains, the
antennas and their placement fixed for all four runs.

RX:
```bash
python receiver/gnu_radio/djscc_rx.py --device-address 192.168.1.67
```

TX:
```bash
python transmitter/gnu_radio/djscc_tx.py --device-address 192.168.1.68
```

## 4. The four runs

Start the RX socket first, then the TX socket. Each RX stops after 24 images. The ADJSCC
encoder takes an SNR (`--snr-db`): run **matched** first, read the
`[snr:pilots] live=XX.XX dB` line the RX prints, and use that value (rounded) for every
ADJSCC transmission.

**matched**
```bash
python receiver/socket_rx.py --model ckpts/adjscc-matched-snr-final/best --use-live-snr --interleave --count 24 --no-timestamp --output-dir received_images/ota_align/matched
```
```bash
python transmitter/socket_tx.py --model ckpts/adjscc-matched-snr-final/best --source folder --path /path/to/kodak_dataset --interleave --snr-db 28
```

**aligned** (TX command as above). At start-up the RX must print
`[*] aligner attached: kind=conv mode=residual`.
```bash
python receiver/socket_rx.py --model marcellobullo/djscc-convnext-cr6-awgn --aligner ckpts/adjscc-matched-snr-final/aligner-convnext-awgn --use-live-snr --interleave --count 24 --no-timestamp --output-dir received_images/ota_align/aligned
```

**misaligned** (TX command as above)
```bash
python receiver/socket_rx.py --model marcellobullo/djscc-convnext-cr6-awgn --use-live-snr --interleave --count 24 --no-timestamp --output-dir received_images/ota_align/misaligned
```

**target** (the ConvNeXt encoder takes no SNR)
```bash
python receiver/socket_rx.py --model marcellobullo/djscc-convnext-cr6-awgn --use-live-snr --interleave --count 24 --no-timestamp --output-dir received_images/ota_align/target
```
```bash
python transmitter/socket_tx.py --model marcellobullo/djscc-convnext-cr6-awgn --source folder --path /path/to/kodak_dataset --interleave
```

For a live demo, use `--source camera` on the TX and drop `--count 24` on the RX (add
`--duplicate-threshold 0.99` to keep near-identical camera frames).

## 5. Score

On the RX laptop. Each received image is matched to its original by content, so a lost image
does not shift the pairing; per-image scores go to a CSV next to the run folders.

```bash
python scripts/eval_ota_alignment.py --kodak /path/to/kodak_dataset --run aligned=received_images/ota_align/aligned --run misaligned=received_images/ota_align/misaligned --run matched=received_images/ota_align/target --run source=received_images/ota_align/matched
```

Here `matched=` points at the **target** folder, so the printed ARR is measured against the
ConvNeXt ceiling, which is the most the aligned run can reach. Pass
`matched=received_images/ota_align/matched` instead for the ARR against the source model.

## 6. Distance sweep with latent dumps (paper protocol)

Record the received latents once per position and decode them offline with every aligner,
so all aligners see the same received symbols (same noise, same lost packets). Per position
only **two** transmissions are needed: the ADJSCC encoder (for matched, misaligned and every
aligned run) and the ConvNeXt encoder (for target).

Fix the TX/RX gains and the antennas for the whole session; move only the RX. At each
position (e.g. 1, 2, 4, 6, 8, 10 m, plus one non-line-of-sight spot), with
`P=received_images/ota/pos_<d>m`:

ADJSCC transmission (use the same `--snr-db` at every position, e.g. the live SNR read at the
closest position; note the `[snr:pilots] live=` values in the RX log for every position):
```bash
python receiver/socket_rx.py --model ckpts/adjscc-matched-snr-final/best --use-live-snr --interleave --count 24 --no-timestamp --output-dir $P/matched --dump-dir $P/dump_adjscc
```
```bash
python transmitter/socket_tx.py --model ckpts/adjscc-matched-snr-final/best --source folder --path /path/to/kodak_dataset --interleave --snr-db 28
```

ConvNeXt transmission:
```bash
python receiver/socket_rx.py --model marcellobullo/djscc-convnext-cr6-awgn --use-live-snr --interleave --count 24 --no-timestamp --output-dir $P/target --dump-dir $P/dump_convnext
```
```bash
python transmitter/socket_tx.py --model marcellobullo/djscc-convnext-cr6-awgn --source folder --path /path/to/kodak_dataset --interleave
```

Each dump frame stores the received latent and the live CSI the RX measured. Decode offline
(any machine), e.g. for the ADJSCC dump:
```bash
python scripts/decode_ota_dumps.py --dump $P/dump_adjscc --kodak /path/to/kodak_dataset --model marcellobullo/djscc-convnext-cr6-awgn --run misaligned=none --run conv=ckpts/adjscc-matched-snr-final/aligner-convnext-awgn --run rescnn=ckpts/aligners/rescnn --run jw=ckpts/aligners/jw_t1_best --run wiener=ckpts/aligners/wiener5_100 --out results/ota/pos_<d>m
```
and the matched / target references from the same dumps with their own decoders
(`--model ckpts/adjscc-matched-snr-final/best --run matched=none` on `dump_adjscc`,
`--model marcellobullo/djscc-convnext-cr6-awgn --run target=none` on `dump_convnext`).
Repeat each position two or three times (or send 48 images) if time allows, to report a spread.

## Reference numbers (simulation, not over the air)

AWGN at 14 dB, DIV2K validation images: source matched 32.8 dB, target 28.6 dB, misaligned
11.2 dB. In the offline GNU Radio replica at high SNR the source model reaches about 39 dB and
ConvNeXt-AWGN about 33-34 dB, so the aligned run is capped near 33-34 dB.

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `Repo id must be in the form 'repo_name' or 'namespace/repo_name'` | The `--model` / `--src-model` folder does not exist where you run the command (`ls -d ckpts/*/best`); a missing local folder is taken as a Hugging Face id. |
| `huggingface-cli is deprecated` | Use `hf upload ...` and `hf auth login`. |
| No `aligner attached` line | Wrong `--aligner` path; it must be the folder with `aligner.pth` and `aligner_config.json`. |
| No `[snr:pilots]` lines | Run the RX with `--snr-method legacy` to fall back to the old estimate (the pilot logger is newer and less tested on hardware). |
| `unrecognized arguments: --init-aligner` | The machine is on an older commit: `git pull`. |
| Target or aligned images soft at high SNR | The ConvNeXt-AWGN decoder was trained up to 20 dB; try the RX with `--snr-db 19` instead of `--use-live-snr`. |
| Aligned well below target | The `conv` aligner is linear and small; try `--kind twoconv` or `--loss mse_lpips`. |
