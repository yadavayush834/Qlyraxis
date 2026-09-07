# Beacon verifier model

`beacon_verifier.onnx` is the portable Phase 6 candidate verifier. It accepts
`float32` grayscale tensors shaped `[N, 1, 32, 32]` with values in `[0, 1]` and
returns one beacon probability per patch.

Architecture: eight 3 x 3 convolution filters, ReLU, 2 x 2 max pooling, and a
trained logistic output layer. The convolution bank captures compact blobs,
edges, streaks, and center-surround contrast. The 2,049 trainable head
parameters are fitted on balanced synthetic beacon and artifact patches.

The committed model was generated deterministically with:

```bash
qlyraxis train-ai --samples-per-class 500 --epochs 300 --seed 26169
```

Held-out synthetic accuracy at generation time: 97.20%. OpenCV DNN is the
default CPU inference engine; ONNX Runtime can be installed as an optional
alternative.
