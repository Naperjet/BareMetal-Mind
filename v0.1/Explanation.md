# How a Single Perceptron Works — v0.1

*Version:* v0.1 — Single Perceptron in C++
*Audience:* written for a control systems engineer transitioning from MATLAB to C++/GPU computing.

---

## 1. What a perceptron is

The perceptron (Rosenblatt, 1958) is the smallest possible learning machine: a single artificial neuron that maps an input vector to a scalar output. It is a **static, memoryless, nonlinear map** whose coefficients are *adapted from data* rather than designed from a plant model.

For a control engineer, the cleanest mental model is a block diagram:

```
x ──▶ [ static gain  wᵀ ] ──▶ [ nonlinear element  f ] ──▶ y
                 ▲
                 └── offset b
```

There are no states, no dynamics, no transfer function in s — just an algebraic map. Everything "intelligent" about it lives in *how the gain is updated*.

---

## 2. The model

Given input **x** ∈ ℝⁿ, parameters θ = (**w**, b) with **w** ∈ ℝⁿ (weights) and b ∈ ℝ (bias):

```
z = w₁x₁ + … + wₙxₙ + b  =  wᵀx + b        (pre-activation, a dot product)
y = f(z)                                    (activation output)
```

One forward pass costs O(n) multiply–adds — a single dot product. This is exactly the operation that becomes a GPU kernel in v0.2 and a tiled GEMM in v0.5; the arithmetic never changes, only where it executes.

### 2.1 Learning as empirical risk minimization

Formally, learning is optimization over a hypothesis class. With labels encoded as t ∈ {0, 1} (equivalently y* = 2t − 1 ∈ {−1, +1}), the model family is

```
H = { h_θ : x ↦ f(wᵀx + b) },      θ = (w, b) ∈ ℝⁿ⁺¹,
```

and learning means solving the **empirical risk minimization** problem

```
minimize   L_D(θ) = (1/|D|) Σᵢ ℓ(h_θ(xᵢ), tᵢ)      over θ ∈ ℝⁿ⁺¹.
```

Two properties of the resulting optimization landscape are worth stating precisely:

- For the **step** activation with the perceptron criterion ℓ = max(0, −y*·(wᵀx + b)) — a hinge-like surrogate — the loss is **convex** in θ; piecewise-linear, in fact. This convexity is exactly what the Novikoff proof (§6) exploits.
- For the **sigmoid** with squared loss ℓ = ½(t − σ(wᵀx + b))², the landscape is **non-convex** in θ (an S-curve composed with a quadratic), though benign for a single neuron: in the separable regime every stationary point sits at, or numerically near, the global minimum.

---

## 3. Geometry of the decision boundary

The neuron classifies by thresholding. The set of points where the output flips,

```
H = { x ∈ ℝⁿ : wᵀx + b = 0 },
```

is a **hyperplane**: **w** is its normal vector, b/‖w‖ is its signed offset from the origin. In 2D this is a straight line.

A dataset {(xᵢ, tᵢ)} is **linearly separable** if there exists (**w**, b) such that

```
tᵢ (wᵀxᵢ + b) > 0        for all i,
```

i.e. every point lies strictly on the correct side of H. The smallest such strict distance, expressed with a unit-norm (w, b), is the **margin γ**. Separability and margin are the two ingredients of the convergence guarantee in §6.

---

## 4. Activation functions

Two activations are implemented in v0.1, chosen for the contrast they teach:

| Activation | f(z) | f′(z) | Character |
|---|---|---|---|
| Step | 1 if z ≥ 0, else 0 | 0 almost everywhere | Ideal relay; cheap, but piecewise-constant → carries no slope information |
| Sigmoid | 1 / (1 + e⁻ᶻ) | f(z)(1 − f(z)) | Smooth saturator; derivative expressible through the output itself |

Two properties of the sigmoid matter later:

1. **Self-derivative.** f′ = y(1 − y) means the local slope can be recovered from the forward output without recomputing z — a numerical convenience that becomes structural in backpropagation (v0.4), where every layer's δ term is built from f′ of the stored forward activations.
2. **Saturation.** y(1 − y) → 0 as y → 0 or 1: a confident neuron stops learning. This is a double-edged sword — it stabilizes the endgame of training, but also causes the vanishing-gradient problem in deep stacks.

---

## 5. Learning

Let t ∈ {0, 1} be the target label and y the network output. Learning adjusts θ to reduce error. Two rules are implemented, matched one-to-one to the two activations.

### 5.1 Classic perceptron rule (step activation)

```
w ← w + η·(t − y)·x
b ← b + η·(t − y)
```

with learning rate η > 0. The error e = t − y ∈ {−1, 0, +1}: the update fires only on mistakes, in the direction of the misclassified input.

*Control interpretation.* The weight integrates the correlation between input and error — this is an **adaptive-gain / integral-action** mechanism, and it is literally the same update family as the **LMS (Widrow–Hoff)** algorithm of adaptive filtering. The perceptron rule differs from LMS only in that the error is quantized by the step function.

*Formal view.* With y* = 2t − 1 ∈ {−1, +1} and the convex surrogate ℓ(w, b) = max(0, −y*·(wᵀx + b)), the perceptron rule is exactly **subgradient descent** on ℓ: on a mistake, the subgradient is −y*·(x, 1) and the update steps against it; on a correct sample, ℓ = 0 and the (sub)gradient vanishes. The Novikoff analysis (§6) is carried out on this surrogate — that is where its convexity enters.

*Cost.* Each mistaken update is O(n).

### 5.2 Gradient descent (sigmoid activation)

Define the loss on a single sample as the squared error

```
L(z) = ½ (t − f(z))².
```

Gradient descent steps against the gradient. By the chain rule:

```
∂L/∂wⱼ = (t − y) · f′(z) · ∂z/∂wⱼ  =  (t − y) · f′(z) · xⱼ
∂L/∂b  = (t − y) · f′(z)
```

so, using f′(z) = y(1 − y):

```
δ ≡ (t − y) · y (1 − y)
w ← w + η·δ·x
b ← b + η·δ
```

Same shape as the perceptron rule, with two differences:

1. The error is continuous, not quantized — small mistakes produce small updates.
2. The effective gain carries the factor y(1 − y): updates taper as the neuron saturates (§4), which settles the endgame of training.

*Forward-looking note.* δ = −∂L/∂z (up to sign convention) is the **neuron's local error signal**. In a multi-layer network (v0.4), δ is propagated backward across layers through transposed weight matrices — the chain rule applied recursively — which is precisely backpropagation. v0.1 contains the entire mechanism in miniature.

*Batch vs. stochastic.* The rule above processes one sample at a time — **stochastic gradient descent (SGD)** in the Robbins–Monro (1951) sense. The deterministic variant (§5.3) accumulates over the full dataset before stepping — **(batch) gradient descent** — whose analysis requires only smoothness and yields monotone decrease of the full loss. SGD trades monotonicity for O(1) pass memory and much cheaper epochs; its convergence theory then becomes stochastic (Robbins–Monro conditions on η decay). v0.1 implements per-sample SGD; the batch form is deferred to the parallel stages, where it becomes a GEMM.

### 5.3 Batch form (preview of v0.3)

Accumulating updates over a mini-batch before applying them,

```
w ← w + η · Σᵢ δᵢ xᵢ  =  w + η · Xᵀ δ,
```

turns the update into a matrix–vector product (X is the batch's input matrix, δ its error vector). That is the computational pattern v0.3 parallelizes and v0.5 tiles.

---

## 6. Convergence guarantee (Novikoff, 1962)

**Theorem (perceptron convergence).** *Augment each input as x̃ᵢ = (xᵢ, 1) and let R = maxᵢ ‖x̃ᵢ‖. If the training set is linearly separable with margin γ > 0 — i.e. ∃ unit w̃* with tᵢ(w̃*ᵀx̃ᵢ) ≥ γ for all i — then the classic perceptron rule (with any η > 0) makes at most*

```
(R / γ)²
```

*mistakes, after which it classifies the entire training set correctly and its parameters stop changing.*

*Proof sketch.* Index the mistakes k = 1, 2, …. Each mistake updates w̃ ← w̃ + η tᵢ x̃ᵢ in the direction of a correctly-classified-by-w̃* point, so

```
w̃ₖᵀ w̃*  ≥  w̃ₖ₋₁ᵀ w̃* + η γ   ⟹   w̃ₖᵀ w̃* ≥ k η γ,
‖w̃ₖ‖²  ≤ ‖w̃ₖ₋₁‖² + η²R²      ⟹   ‖w̃ₖ‖² ≤ k η² R²,
```

(the second line uses tᵢ(w̃ₖ₋₁ᵀx̃ᵢ) ≤ 0 — a mistake — and ‖x̃ᵢ‖ ≤ R). Combining via w̃ₖᵀw̃* ≤ ‖w̃ₖ‖ gives k η γ ≤ √(k η² R²), hence k ≤ (R/γ)². ∎

Three observations a control engineer will appreciate:

- The bound is **scale-invariant in η** — the learning rate changes trajectories, not the worst-case mistake count.
- The bound depends only on **R/γ** — data geometry, not dimension n. Tight margins (data barely separable) are the "stiff" case.
- Once converged the parameters **stop moving** — a perceptron on separable data is not a stochastic process but a terminating algorithm. Expect exactly this in the v0.1 demo: 100% accuracy, then frozen weights.

If the data is *not* separable, no such bound exists: the weights cycle indefinitely and accuracy plateaus below 1 — the behavior demonstrated by the XOR experiment in §8.

---

## 7. Training algorithm

```
input: dataset D = {(xᵢ, tᵢ)}, epochs E, learning rate η
initialize wⱼ ~ U(−0.5, 0.5), b ← 0
repeat E times:
    shuffle D
    for each (x, t) in D:
        y ← f(wᵀx + b)                      # forward
        update w, b by rule (§5.1 or §5.2)  # backward
    optionally: log loss/accuracy over D
return w, b
```

*Shuffling* breaks pathological orderings (e.g. all class-0 samples first) that would otherwise bias the trajectory within each epoch. *Evaluation* uses two metrics:

- **MSE loss**: (1/|D|) Σᵢ ½(tᵢ − yᵢ)² — the training signal, monotone-ish under gradient descent.
- **Accuracy**: fraction of samples where the thresholded prediction equals tᵢ — the deployment metric.

Per-epoch cost: O(|D| · n).

---

## 8. What it cannot do — and why that matters

The XOR function (label 1 iff x₁x₂ > 0) is the canonical counterexample (Minsky & Papert, 1969): its four points have no separating line. A single perceptron therefore cannot represent it, regardless of training — accuracy plateaus near chance (≈ 50–60 %).

This failure is the **motivating cliff for v0.4**: stacking neurons in layers and training by backpropagation composes the linear pieces into nonlinear boundaries. Every ingredient of that construction — the δ signal, the batch form Xᵀδ, the GPU kernel shape — already exists in v0.1 in miniature.

---

## 9. Implementation notes (v0.1, C++17)

| Decision | Choice | Rationale |
|---|---|---|
| Scalar type | `float` | Matches OpenCL (`cl_float`) in v0.2+; avoids a costly type migration across versions. CPU-only numerics would prefer `double`, but consistency across the roadmap wins |
| Weight init | `U(−0.5, 0.5)`, `std::mt19937` | Small random init breaks symmetry; explicit seed argument makes runs reproducible (and the test suite deterministic) |
| Shuffling | `std::shuffle` with the same seeded engine | Breaks pathological sample ordering each epoch |
| Size mismatches | `std::invalid_argument` | Fails loudly even in release builds (`NDEBUG` would disable `assert`) |
| Sigmoid stability | naive `1/(1+e⁻ᶻ)` | Graceful under IEEE-754: as z → −∞, e⁻ᶻ → +inf and 1/(1+inf) = 0 (no NaN). The overflow-free symmetric form `y = z ≥ 0 ? 1/(1+e⁻ᶻ) : eᶻ/(1+eᶻ)` is noted for later hardening |
| API surface | `forward / predict / train / loss / accuracy / set_weights` | `set_weights` exists as a verification hook: the tests pin hand-computed values of the forward pass and of single update steps |

---

## References

- F. Rosenblatt, *Principles of Neurodynamics: Perceptrons and the Theory of Brain Mechanisms*, Cornell Aeronautical Laboratory, 1962. (Scanned copy unavailable at write time; the historical account in Haykin ch. 1 covers it.)
- A. L. Novikoff, *On convergence proofs on perceptrons*, Symposium on the Mathematical Theory of Automata, 1962. (Source of the convergence bound in §6.)
- M. Minsky, S. Papert, *Perceptrons*, MIT Press, 1969. (The XOR limitation.)
- S. Haykin, *Neural Networks and Learning Machines*, 3rd ed., Pearson, 2009.
- B. Widrow, M. E. Hoff, *Adaptive switching circuits*, IRE WESCON, 1960. (LMS; §5.1.)
- H. Robbins, S. Monro, *A stochastic approximation method*, Annals of Mathematical Statistics 22(3), 1951. (SGD; §5.2.)
- Free PDFs in `../Reference Material/v0.1/` (see also `Paid Books - Shopping Notes.md` in the same folder).
