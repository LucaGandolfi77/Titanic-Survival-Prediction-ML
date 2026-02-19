# Backpropagation — Mathematical Foundations

## Overview

Backpropagation computes the gradient of a scalar loss $L$ with respect
to every trainable parameter $\theta$ in the network.  It is a direct
application of the **chain rule** of calculus.

---

## Notation

| Symbol | Meaning |
|--------|---------|
| $X$    | Input data, shape $(m, n_{\text{in}})$ |
| $W$    | Weight matrix, shape $(n_{\text{in}}, n_{\text{out}})$ |
| $b$    | Bias vector, shape $(1, n_{\text{out}})$ |
| $Z$    | Pre-activation $= X W + b$, shape $(m, n_{\text{out}})$ |
| $A$    | Post-activation $= f(Z)$, shape $(m, n_{\text{out}})$ |
| $L$    | Scalar loss |
| $m$    | Batch size |

---

## Forward Pass (single layer)

$$
Z = X W + b
$$

$$
A = f(Z)
$$

where $f$ is a non-linear activation function.

---

## Backward Pass (single dense layer)

Given upstream gradient $\frac{\partial L}{\partial A}$ (shape $m \times n_{\text{out}}$):

### Step 1 — Through Activation

$$
\frac{\partial L}{\partial Z} = \frac{\partial L}{\partial A} \odot f'(Z)
$$

where $\odot$ denotes element-wise (Hadamard) product.

### Step 2 — Weight Gradient

$$
\frac{\partial L}{\partial W} = \frac{1}{m} X^T \frac{\partial L}{\partial Z}
$$

Shape: $(n_{\text{in}}, m) \times (m, n_{\text{out}}) = (n_{\text{in}}, n_{\text{out}})$ ✓

### Step 3 — Bias Gradient

$$
\frac{\partial L}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} \frac{\partial L}{\partial Z_i}
$$

Shape: $(1, n_{\text{out}})$ ✓

### Step 4 — Input Gradient (for previous layer)

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Z} W^T
$$

Shape: $(m, n_{\text{out}}) \times (n_{\text{out}}, n_{\text{in}}) = (m, n_{\text{in}})$ ✓

---

## Activation Derivatives

### ReLU

$$
f(z) = \max(0, z) \qquad f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \leq 0 \end{cases}
$$

### Sigmoid

$$
f(z) = \sigma(z) = \frac{1}{1 + e^{-z}} \qquad f'(z) = \sigma(z)(1 - \sigma(z))
$$

### Tanh

$$
f(z) = \tanh(z) \qquad f'(z) = 1 - \tanh^2(z)
$$

### Softmax + Cross-Entropy (combined)

When softmax output is paired with categorical cross-entropy loss:

$$
\frac{\partial L}{\partial Z} = \hat{Y} - Y
$$

This elegant simplification avoids computing the full $K \times K$ Jacobian.

---

## Loss Functions

### Cross-Entropy (multi-class)

$$
L = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} Y_{ik} \ln(\hat{Y}_{ik})
$$

### Mean Squared Error (regression)

$$
L = \frac{1}{m} \sum_{i=1}^{m} (\hat{Y}_i - Y_i)^2
$$

$$
\frac{\partial L}{\partial \hat{Y}} = \frac{2}{m}(\hat{Y} - Y)
$$

### Binary Cross-Entropy

$$
L = -\frac{1}{m} \sum_{i=1}^{m} \left[ Y_i \ln(\hat{Y}_i) + (1-Y_i)\ln(1-\hat{Y}_i) \right]
$$

---

## Gradient Checking

To verify correctness, compare the analytic gradient against a numerical
approximation using centred differences:

$$
\frac{\partial L}{\partial \theta_j} \approx
\frac{L(\theta_j + \varepsilon) - L(\theta_j - \varepsilon)}{2\varepsilon}
$$

Relative error:

$$
\text{err} = \frac{\|g_{\text{analytic}} - g_{\text{numeric}}\|_2}
                  {\|g_{\text{analytic}}\|_2 + \|g_{\text{numeric}}\|_2}
$$

| Threshold | Interpretation |
|-----------|----------------|
| $< 10^{-7}$ | ✅ Excellent |
| $< 10^{-5}$ | ✅ Good      |
| $< 10^{-3}$ | ⚠️ Suspicious |
| $> 10^{-3}$ | ❌ Bug likely  |

---

## Weight Initialization

Proper initialization prevents vanishing / exploding gradients.

### He (Kaiming) — for ReLU

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)
$$

### Xavier (Glorot) — for Sigmoid / Tanh

$$
W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}},\;
\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right]
$$

### LeCun — for SELU

$$
W \sim \mathcal{N}\left(0, \sqrt{\frac{1}{n_{\text{in}}}}\right)
$$
