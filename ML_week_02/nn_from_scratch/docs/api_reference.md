# API Reference

## Quick Start

```python
import numpy as np
from src.core import DenseLayer, ReLU, Softmax, CrossEntropyLoss, Adam
from src.network import Sequential, Model
from src.utils import one_hot_encode

# Build network
network = Sequential(
    DenseLayer(784, 256, activation=ReLU(), seed=42),
    DenseLayer(256, 128, activation=ReLU(), seed=43),
    DenseLayer(128, 10,  activation=Softmax(), seed=44),
)

# Compile
model = Model(
    network=network,
    loss_fn=CrossEntropyLoss(),
    optimizer=Adam(lr=0.001),
)

# Train
history = model.fit(X_train, Y_train, epochs=20, batch_size=128,
                    X_val=X_val, Y_val=Y_val)

# Evaluate
loss, acc = model.evaluate(X_test, Y_test)

# Predict
Y_hat = model.predict(X_new)
```

---

## Core Classes

### `DenseLayer(n_in, n_out, activation=None, weight_init=he_init, seed=None)`

Fully-connected layer.

| Parameter     | Type         | Description                          |
|---------------|--------------|--------------------------------------|
| `n_in`        | `int`        | Input dimension                      |
| `n_out`       | `int`        | Output dimension                     |
| `activation`  | `Activation` | Optional activation function         |
| `weight_init` | `callable`   | Weight initialization strategy       |
| `seed`        | `int`        | Random seed for reproducibility      |

**Methods:**
- `forward(X) → Y` — compute `Y = f(X @ W + b)`
- `backward(dY) → dX` — compute gradients and return downstream grad

**Properties:**
- `params → {"W": ndarray, "b": ndarray}`
- `grads → {"W": ndarray, "b": ndarray}`

---

### Activation Functions

All activations share the interface:
- `forward(Z) → A`
- `backward(dA) → dZ`

| Class       | Formula                                | Best With    |
|-------------|----------------------------------------|--------------|
| `ReLU()`    | `max(0, Z)`                            | Hidden layers|
| `LeakyReLU(alpha=0.01)` | `max(αZ, Z)`               | Hidden layers|
| `Sigmoid()` | `1 / (1 + exp(-Z))`                   | Binary output|
| `Tanh()`    | `(exp(Z) - exp(-Z)) / (exp(Z) + exp(-Z))` | Hidden layers  |
| `Softmax()` | `exp(Z) / sum(exp(Z))`                | Multi-class output |

---

### Loss Functions

| Class                      | Task        | Gradient               |
|---------------------------|-------------|------------------------|
| `CrossEntropyLoss()`       | Multi-class | `Ŷ − Y` (combined)    |
| `MSELoss()`               | Regression  | `2/m · (Ŷ − Y)`       |
| `BinaryCrossEntropyLoss()` | Binary      | `−(Y/Ŷ − (1−Y)/(1−Ŷ))/m` |

**Methods:**
- `forward(Y_hat, Y) → float`
- `backward() → ndarray`

---

### Optimizers

| Class                          | Key Params            |
|-------------------------------|-----------------------|
| `SGD(lr=0.01)`                | learning rate         |
| `Momentum(lr=0.01, beta=0.9)` | lr, momentum coeff   |
| `Adam(lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8)` | lr, moment decays |

**Method:** `step(layers)` — update all trainable parameters.

---

### Weight Initializers

All have signature: `fn(fan_in, fan_out, rng=None) → ndarray`

| Function       | Distribution              | Best With    |
|----------------|---------------------------|--------------|
| `he_init`      | Normal, σ = √(2/n_in)    | ReLU         |
| `xavier_init`  | Uniform, limit = √(6/(n_in+n_out)) | Sigmoid/Tanh |
| `lecun_init`   | Normal, σ = √(1/n_in)    | SELU         |
| `zeros_init`   | All zeros                 | Biases       |

---

## Network Module

### `Sequential(*layers)`

| Method            | Description                          |
|-------------------|--------------------------------------|
| `forward(X)`      | Pipe through all layers              |
| `backward(dY)`    | Reverse gradient propagation         |
| `predict(X)`      | Alias for forward                    |
| `add(layer)`      | Append a layer                       |
| `count_params()`  | Total trainable parameters           |
| `summary()`       | Keras-style summary string           |

### `Model(network, loss_fn, optimizer)`

| Method                 | Description                               |
|------------------------|-------------------------------------------|
| `fit(X, Y, ...)`       | Full training loop with batching          |
| `evaluate(X, Y)`       | Return `(loss, accuracy)`                |
| `predict(X)`           | Forward-only inference                    |
| `save_weights(path)`   | Save to `.npz`                           |
| `load_weights(path)`   | Load from `.npz`                         |

**`fit()` Parameters:**

| Parameter              | Type    | Default | Description                 |
|------------------------|---------|---------|-----------------------------|
| `epochs`               | `int`   | 100     | Training epochs             |
| `batch_size`           | `int`   | 32      | Mini-batch size             |
| `X_val`, `Y_val`       | ndarray | None    | Validation data             |
| `verbose`              | `bool`  | True    | Print per-epoch metrics     |
| `early_stop_patience`  | `int`   | 0       | Early stopping (0=disabled) |
| `shuffle`              | `bool`  | True    | Shuffle each epoch          |

---

## Utilities

### Data Utils

```python
from src.utils import BatchGenerator, shuffle_data, train_test_split, one_hot_encode

batches = BatchGenerator(X, Y, batch_size=64, shuffle=True)
X_s, Y_s = shuffle_data(X, Y)
X_tr, X_te, Y_tr, Y_te = train_test_split(X, Y, test_size=0.2, seed=42)
Y_oh = one_hot_encode(labels, n_classes=10)
```

### Metrics

```python
from src.utils.metrics import accuracy, precision, recall, f1_score, confusion_matrix

acc = accuracy(y_true, y_pred)
p   = precision(y_true, y_pred)
r   = recall(y_true, y_pred)
f1  = f1_score(y_true, y_pred)
cm  = confusion_matrix(y_true, y_pred)
```

### Visualization

```python
from src.utils.visualization import (
    plot_training_curves,
    plot_decision_boundary,
    plot_confusion_matrix,
    plot_weight_distributions,
)
```

---

## Gradient Checking

```python
from src.validation import gradient_check, gradient_check_layer

# Check a single layer
errors = gradient_check_layer(layer, X, dY, verbose=True)
# errors = {"W": 1.23e-8, "b": 4.56e-9}
```
