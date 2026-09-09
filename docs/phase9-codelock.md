# Phase 9: CodeLock optical identity

CodeLock prevents the coarse-alignment loop from locking onto an unregistered
bright object. A coded scenario declares an identity, a repeating binary
pattern, the number of frames per symbol, a detectable low intensity, and
optional decoy patterns.

```json
"beacon_code": {
  "identity": "QLX-07",
  "pattern": "0000011001010",
  "symbol_frames": 1,
  "low_intensity": 90,
  "decoy_patterns": ["1111100110101"]
}
```

For each visual candidate, the detector maintains a spatially associated
history of frame index and local contrast. It compares those observations with
every cyclic phase of the registered pattern using normalized correlation.
Candidates remain hidden from the Kalman tracker until a complete code period
has been observed and correlation is at least 0.70. Missing frames do not shift
the code because expected symbols are indexed by the original frame number.

Run the controlled experiment with:

```bash
qlyraxis compare configs/scenarios/codelock_decoy.json --frames 600 \
  --output-dir reports/codelock
```

Both sides use the same scenario, random seed, AI verifier, motion estimator,
and controller. Their camera views can diverge after different lock decisions,
as expected in a closed loop. The baseline disables only CodeLock, making decoy
rejection the single experimental variable.
