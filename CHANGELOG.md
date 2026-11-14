# Changelog

All notable changes to this project are documented in this file. Versions follow the project roadmap.

## [v0.1] - Single Perceptron in C++

### Added

- 'Perceptron' class ('p1::Perceptron'): 'forward' / 'predict' / 'train' / 'loss' / 'accuracy', with **Step** and **Sigmoid** activations and seeded, reproducible initialization.
- Both learning rules: classic perceptron rule (step) and gradient descent with `δ = (t − y)·y(1−y)` (sigmoid)
- Dataset generators: linearly separable Gaussian blobs ('make-blobs') and canonical non-separable XOR task ('make-xor)
- Assert based test suite with forward-pass and single update checks pinned to 1e-4
- Demo binary ('examples/perceptron_demo.cpp) with per-epoch training logs
- 'v0.1/EXPLANATION.md': ull mathematical treatment — model, empirical-risk framing, both rules derived, Novikoff convergence bound with proof sketch, SGD vs. batch-GD discussion, control-theoretic (LMS / adaptive-gain) interpretation

### Results (measured, seed-fixed)

- Sigmoid + SGD: 100% train/test accuracy on separable blobs; loss 0.0015 → 0.0001 over 20 epochs
- Step + perceptron rule: converges within epoch 1; weights freeze afterwards (Novikoff bound in action)
- XOR: plateaus at 53.25% after 50 epochs — chance level; the motivating cliff for the full MLP (v0.4)
