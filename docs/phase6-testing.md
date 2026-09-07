# Phase 6 local testing

The committed ONNX model runs through the normal OpenCV dependency. Retraining
requires the optional ONNX exporter:

```bash
python -m pip install -e .
# Only when retraining:
python -m pip install -e '.[training]'
```

## Test an MP4

Create a repeatable disturbed sample and analyze it without simulator state or
ground truth:

```bash
qlyraxis record configs/scenarios/fog_figure_eight.json \
  work/phase6-demo/fog.mp4 --frames 240

qlyraxis analyze work/phase6-demo/fog.mp4 \
  --model models/beacon_verifier.onnx \
  --output-dir work/phase6-analysis
```

The analyzer prints decoded-frame count, verified-detection count, acquisition
time, final state, lock retention, and measured CPU throughput. It also writes
an annotated final frame. `--no-ai` runs the same file with only the classical
candidate detector for comparison.

## Test an image sequence

Place PNG, JPEG, BMP, or TIFF frames in one directory. Numeric file names are
naturally ordered, so `frame2.png` comes before `frame10.png`.

```bash
qlyraxis analyze path/to/frames --fps 30 --max-frames 300
```

## Retrain the verifier

```bash
qlyraxis train-ai \
  --samples-per-class 500 \
  --epochs 300 \
  --seed 26169 \
  --dataset-output work/beacon_patches.npz \
  --model models/beacon_verifier.onnx \
  --weights models/beacon_verifier.npz
```

Training is deterministic for a given seed. The command reports accuracy on a
separately generated held-out dataset and exports both NumPy weights and ONNX.

## Automated checks

```bash
python -m unittest discover -s tests -v
```

The tests cover MP4 decoding, timestamping, image ordering, source reset,
dataset reproducibility, learning convergence, weight serialization, patch
extraction, and numerical agreement between NumPy and OpenCV ONNX inference.
