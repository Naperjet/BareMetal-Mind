# BareMetal-Mind - Neural Networks, OpenCL, and Deep RL from Scratch

BareMetal-Mind is a neural network framework built entirely from first principles in C++, scaled up stage by stage onto OpenCL GPU kernels, and finally used as the function approximator for reinforcement learning. No PyTorch, no BLAS, no shortcuts - every dot product, gradient and kernel is written by hand, with a CPU reference implementation kept at every stage for correctness and a benchmark harness from the first GPU stage onward.

## Roadmap

| Version  | Target                                              | Status  |
|----------|-----------------------------------------------------|---------|
| v0.1     | Single Perceptron in C++                            | Planned |
| v0.2     | Forward Pass in OpenCL                              | Planned |
| v0.3     | Batch Processing with multiple parallel perceptron  | Planned |
| v0.4     | Full MLP on OpenCL                                  | Planned |
| v0.5     | Shared Memory Optimization                          | Planned |
| v1.0     | DQN on top of MLP                                   | Planned |
| v1.1     | Q-Learning on top of MLP                            | Planned |

Each version is in its own dedicated folder. Each version is an evolved copy of the previous one, so every stage of the progression stays frozen, inescapable, and independently buildable.
