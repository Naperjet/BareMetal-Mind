# RESULTS — v0.1 — Single Perceptron in C++

## 1. Test environment

- CPU: Intel(R) Core(TM) i7-6600U CPU @ 2.60GHz (2 cores / 4 threads, Skylake)
- Memory: 16 GB
- OS: Linux Mint 22.3
- Compiler: GCC 13.3.0 (Ubuntu 13.3.0-6ubuntu2~24.04.1), CMake 3.28.3, Release build (-O3)
- GPU: Intel(R) HD Graphics 520 (24 compute units, max work-group 256)
- OpenCL runtime: Intel(R) OpenCL Graphics, OpenCL 3.0 NEO driver 23.43.027642
- Capture date: 2026-11-01

## 2. How to reproduce these numbers

```bash
sudo apt-get install -y ocl-icd-opencl-dev intel-opencl-icd clinfo   # CPU-only stage; OpenCL not required
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
./build/perceptron_tests | tee results_tests.txt
./build/perceptron_demo   | tee results_demo.txt
ctest --test-dir build
```

The test suite prints its **hardware block** at startup — every result in this file can be tied to the machine that produced it by comparing that block.

## 3. Test-suite output (captured 2026-11-01)

```text
=== Hardware ===
  CPU: Intel(R) Core(TM) i7-6600U CPU @ 2.60GHz
  logical cores: 4

All tests passed.
```

## 4. Demo output (captured 2026-11-01)

```text
v0.1 - Single Perceptron in C++

=== Sigmoid + gradient descent on separable blobs ===
  epoch  1  loss 0.0015  accuracy 100.00%
  epoch  2  loss 0.0007  accuracy 100.00%
  epoch  3  loss 0.0005  accuracy 100.00%
  epoch  4  loss 0.0004  accuracy 100.00%
  epoch  5  loss 0.0003  accuracy 100.00%
  epoch  6  loss 0.0002  accuracy 100.00%
  epoch  7  loss 0.0002  accuracy 100.00%
  epoch  8  loss 0.0002  accuracy 100.00%
  epoch  9  loss 0.0002  accuracy 100.00%
  epoch 10  loss 0.0001  accuracy 100.00%
  epoch 11  loss 0.0001  accuracy 100.00%
  epoch 12  loss 0.0001  accuracy 100.00%
  epoch 13  loss 0.0001  accuracy 100.00%
  epoch 14  loss 0.0001  accuracy 100.00%
  epoch 15  loss 0.0001  accuracy 100.00%
  epoch 16  loss 0.0001  accuracy 100.00%
  epoch 17  loss 0.0001  accuracy 100.00%
  epoch 18  loss 0.0001  accuracy 100.00%
  epoch 19  loss 0.0001  accuracy 100.00%
  epoch 20  loss 0.0001  accuracy 100.00%
  weights: w = (1.07, 1.38), b = -0.00
  decision boundary: 1.07*x1 + 1.38*x2 + -0.00 = 0
  train accuracy: 100.00%
  test  accuracy: 100.00%

=== Step + perceptron rule on separable blobs ===
  epoch  1  loss 0.0000  accuracy 100.00%
  epoch  2  loss 0.0000  accuracy 100.00%
  epoch  3  loss 0.0000  accuracy 100.00%
  epoch  4  loss 0.0000  accuracy 100.00%
  epoch  5  loss 0.0000  accuracy 100.00%
  epoch  6  loss 0.0000  accuracy 100.00%
  epoch  7  loss 0.0000  accuracy 100.00%
  epoch  8  loss 0.0000  accuracy 100.00%
  epoch  9  loss 0.0000  accuracy 100.00%
  epoch 10  loss 0.0000  accuracy 100.00%
  epoch 11  loss 0.0000  accuracy 100.00%
  epoch 12  loss 0.0000  accuracy 100.00%
  epoch 13  loss 0.0000  accuracy 100.00%
  epoch 14  loss 0.0000  accuracy 100.00%
  epoch 15  loss 0.0000  accuracy 100.00%
  epoch 16  loss 0.0000  accuracy 100.00%
  epoch 17  loss 0.0000  accuracy 100.00%
  epoch 18  loss 0.0000  accuracy 100.00%
  epoch 19  loss 0.0000  accuracy 100.00%
  epoch 20  loss 0.0000  accuracy 100.00%
  weights: w = (0.13, 0.39), b = 0.10
  decision boundary: 0.13*x1 + 0.39*x2 + 0.10 = 0
  train accuracy: 100.00%
  test  accuracy: 100.00%

=== XOR: the failure case ===
  accuracy after 50 epochs: 53.25%
  XOR is not linearly separable: a single perceptron cannot learn it.
  This is the motivating cliff for v0.4 (full MLP with backpropagation).
```

## 5. Notes on variance

No timing claims are made in this stage. All results are deterministic, seeded accuracies and weights — they reproduce bit-for-bit on any machine.

