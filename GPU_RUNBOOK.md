# TinyStories GPU Runbook

This repo can now run the TinyStories pipeline on a rented GPU, export the trained model, and serve a downloadable artifact.

## 1. First Command After Cloning

Run a smoke test before spending time on real training:

```bash
python3 remote_gpu_run.py --mode smoke
```

This checks:

- PyTorch imports
- CUDA visibility
- TinyStories download
- tokenizer cache
- forward pass
- loss
- backward pass
- gradient clipping
- optimizer step
- checkpoint writing

If this fails, fix the error before running real training.

## 2. Small Learning Run

This is the safe first real run:

```bash
python3 remote_gpu_run.py --mode train --preset learning-1pct
```

It uses the 1% TinyStories setup and packages a weights-only export.

## 3. First Larger GPU Run

This preset is the first 30M-ish experiment shape:

```bash
python3 remote_gpu_run.py --mode train --preset gpu-30m
```

The `gpu-30m` preset uses:

```text
TRAIN_FRACTION=0.10
TARGET_VOCAB_SIZE=2048
BLOCK_SIZE=512
D_MODEL=512
NUM_LAYERS=8
NUM_HEADS=8
MLP_HIDDEN_SIZE=2048
COMPILE=1
MIXED_PRECISION=1
```

This is deliberately not full TinyStories + 8k BPE yet. Our BPE trainer is educational and from scratch, so scaling tokenizer training all the way up should be done after this run proves the training loop and artifact flow.

## 4. Throughput Comparison

Baseline:

```bash
python3 remote_gpu_run.py --mode train --preset learning-1pct --set COMPILE=0 --set MIXED_PRECISION=0 --set RUN_DIR=runs/baseline_speed
```

Optimized:

```bash
python3 remote_gpu_run.py --mode train --preset learning-1pct --set COMPILE=1 --set MIXED_PRECISION=1 --set RUN_DIR=runs/optimized_speed
```

Compare:

```text
average train-only tokens/sec
best single-step train-only tokens/sec
```

The CSV logs are in each run directory:

```text
runs/<run-name>/loss_log.csv
```

## 5. Downloading The Trained Model

To package and serve a clickable download link:

```bash
python3 remote_gpu_run.py --mode train --preset learning-1pct --serve --public-host YOUR_SERVER_IP
```

The script prints a URL like:

```text
http://YOUR_SERVER_IP:8000/<artifact>.tar.gz
```

If the URL does not open, check the cloud firewall/security group for port `8000`.

Alternative: use `scp`:

```bash
scp root@YOUR_SERVER_IP:/root/projects/machine_learning/artifacts/*.tar.gz .
```

## 6. What The Artifact Contains

The tarball contains:

- weights-only model export
- model config
- BPE merge rules
- vocab tokens
- metadata JSON
- loss log CSV
- README

It does not include optimizer state unless you pass:

```bash
--include-checkpoint
```

Use weights-only artifacts for local sampling/eval. Use full checkpoints only if you need to resume training.
