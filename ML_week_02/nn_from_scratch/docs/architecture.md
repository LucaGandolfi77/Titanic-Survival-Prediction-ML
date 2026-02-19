# Architecture Overview

## Neural Network From Scratch — Design Document

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                         Model                                │
│  ┌─────────────────────┐  ┌──────────┐  ┌──────────────┐   │
│  │     Sequential      │  │   Loss   │  │  Optimizer   │   │
│  │  ┌───────────────┐  │  │          │  │              │   │
│  │  │  DenseLayer   │  │  │ forward  │  │  step        │   │
│  │  │  + Activation  │  │  │ backward │  │  (SGD/Adam)  │   │
│  │  ├───────────────┤  │  │          │  │              │   │
│  │  │  DenseLayer   │  │  └──────────┘  └──────────────┘   │
│  │  │  + Activation  │  │                                    │
│  │  ├───────────────┤  │                                    │
│  │  │     ...       │  │                                    │
│  │  └───────────────┘  │                                    │
│  └─────────────────────┘                                    │
└──────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Forward Pass** (left → right):
   ```
   X → [Layer₀.forward] → [Layer₁.forward] → ... → [Layerₙ.forward] → Ŷ
   ```

2. **Loss Computation**:
   ```
   L = loss.forward(Ŷ, Y)
   ```

3. **Backward Pass** (right → left):
   ```
   dŶ = loss.backward()
   dŶ → [Layerₙ.backward] → ... → [Layer₁.backward] → [Layer₀.backward]
   ```

4. **Parameter Update**:
   ```
   optimizer.step(layers)   # θ ← θ − η·∇L(θ)
   ```

---

## Module Structure

### `src/core/` — Building Blocks

| File              | Contents                                  |
|-------------------|-------------------------------------------|
| `activations.py`  | ReLU, LeakyReLU, Sigmoid, Tanh, Softmax  |
| `initializers.py` | He, Xavier, LeCun, Zeros initialization   |
| `layer.py`        | `Layer` (ABC), `DenseLayer` (FC layer)    |
| `losses.py`       | CrossEntropy, MSE, BinaryCrossEntropy     |
| `optimizers.py`   | SGD, Momentum, Adam                       |

### `src/network/` — Model Containers

| File            | Contents                                    |
|-----------------|---------------------------------------------|
| `sequential.py` | `Sequential` — ordered layer chain          |
| `model.py`      | `Model` — fit/evaluate/predict interface    |

### `src/utils/` — Helpers

| File               | Contents                                 |
|--------------------|------------------------------------------|
| `data_utils.py`    | BatchGenerator, shuffle, split, one-hot  |
| `metrics.py`       | Accuracy, precision, recall, F1, CM      |
| `visualization.py` | Training curves, boundaries, heatmaps    |

### `src/validation/` — Correctness

| File                | Contents                                |
|---------------------|-----------------------------------------|
| `gradient_check.py` | Numerical gradient verification         |

---

## DenseLayer — Detailed

### Forward

```
Z = X @ W + b           # (batch, n_in) @ (n_in, n_out) + (1, n_out) → (batch, n_out)
A = activation(Z)       # element-wise
```

### Backward

Given upstream gradient `dA` (shape: batch × n_out):

```
dZ = activation.backward(dA)              # (batch, n_out)
dW = (1/m) · Xᵀ @ dZ                     # (n_in, n_out)  — averaged over batch
db = (1/m) · sum(dZ, axis=0)              # (1, n_out)
dX = dZ @ Wᵀ                             # (batch, n_in)  — passed to previous layer
```

---

## Optimizer Update Rules

### SGD
```
θ ← θ − η · g
```

### Momentum
```
v ← β·v + (1−β)·g
θ ← θ − η·v
```

### Adam
```
v ← β₁·v + (1−β₁)·g           # first moment
s ← β₂·s + (1−β₂)·g²          # second moment
v̂ = v / (1 − β₁ᵗ)             # bias correction
ŝ = s / (1 − β₂ᵗ)
θ ← θ − η · v̂ / (√ŝ + ε)
```

---

## No External Framework Dependencies

The entire neural network implementation uses **only**:
- `numpy` — matrix operations, vectorized math
- Python standard library — `pathlib`, `struct`, `gzip`, etc.

`sklearn` and `matplotlib` are used only in utilities and examples,
never in the core NN computation.
