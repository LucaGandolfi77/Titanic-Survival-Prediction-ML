# Neural Network From Scratch 🧠

> **Complete neural network framework built with NumPy only** — no PyTorch, TensorFlow, or JAX.  
> Educational mastery of backpropagation, gradient descent, and network architecture at the matrix operation level.

## Highlights

- **Pure NumPy** — every forward/backward pass is explicit matrix math
- **Mathematical rigour** — every function includes equations, shape annotations, chain-rule derivations
- **Gradient checking** — numerical verification of all backpropagation gradients
- **Multiple examples** — XOR, spiral, sine regression, MNIST digit classification
- **Production-quality code** — type hints, docstrings, tests, CI-ready

## Project Structure

```
nn_from_scratch/
├── src/
│   ├── core/
│   │   ├── activations.py      # ReLU, LeakyReLU, Sigmoid, Tanh, Softmax
│   │   ├── initializers.py     # He, Xavier, LeCun weight init
│   │   ├── layer.py            # Layer base class + DenseLayer
│   │   ├── losses.py           # CrossEntropy, MSE, BinaryCrossEntropy
│   │   └── optimizers.py       # SGD, Momentum, Adam
│   ├── network/
│   │   ├── sequential.py       # Sequential container
│   │   └── model.py            # High-level Model (fit/evaluate/predict)
│   ├── utils/
│   │   ├── data_utils.py       # Batching, shuffling, splitting, one-hot
│   │   ├── metrics.py          # Accuracy, Precision, Recall, F1, CM
│   │   └── visualization.py    # Training curves, decision boundaries
│   └── validation/
│       └── gradient_check.py   # Numerical gradient verification
├── examples/
│   ├── xor_example.py          # XOR (classic non-linear test)
│   ├── spiral_example.py       # 2D spiral classification
│   ├── regression_example.py   # sin(x) function approximation
│   └── mnist_example.py        # MNIST digit classification
├── notebooks/
│   ├── 01_backprop_walkthrough.ipynb
│   ├── 02_activation_exploration.ipynb
│   └── 03_mnist_full_pipeline.ipynb
├── tests/
│   ├── test_activations.py
│   ├── test_layers.py
│   ├── test_losses.py
│   ├── test_network.py
│   └── test_gradient_check.py
├── docs/
│   ├── architecture.md
│   ├── backpropagation.md
│   └── api_reference.md
├── train_mnist.py              # CLI entry point
├── Makefile
├── requirements.txt
├── setup.py
└── README.md
```

## Quick Start

```bash
# Install dependencies
make install

# Run tests
make test

# Run examples
make xor          # XOR problem
make spiral       # Spiral classification
make regression   # sin(x) regression
make mnist        # MNIST digit classification (downloads data)

# Or use the CLI directly
python train_mnist.py --epochs 20 --lr 0.001 --batch-size 128
```

## Architecture

```python
from src.core import DenseLayer, ReLU, Softmax, CrossEntropyLoss, Adam
from src.network import Sequential, Model

network = Sequential(
    DenseLayer(784, 256, activation=ReLU()),
    DenseLayer(256, 128, activation=ReLU()),
    DenseLayer(128, 10,  activation=Softmax()),
)

model = Model(network, CrossEntropyLoss(), Adam(lr=0.001))
history = model.fit(X_train, Y_train, epochs=20, batch_size=128)
```

## Key Concepts Demonstrated

### Backpropagation
Every layer computes gradients step by step:
```
dZ = activation.backward(dA)      # through activation
dW = (1/m) · Xᵀ @ dZ             # weight gradient
db = (1/m) · sum(dZ, axis=0)     # bias gradient
dX = dZ @ Wᵀ                     # pass to previous layer
```

### Gradient Checking
Verify correctness with numerical differentiation:
```python
from src.validation import gradient_check_layer
errors = gradient_check_layer(dense_layer, X, dY, verbose=True)
#   W  rel_error = 1.23e-08  ✅
#   b  rel_error = 4.56e-09  ✅
```

### Weight Initialization
Choose initialization based on activation:
- **He** (ReLU): `W ~ N(0, √(2/n_in))`
- **Xavier** (Sigmoid/Tanh): `W ~ U[-√(6/(n_in+n_out)), √(6/(n_in+n_out))]`
- **LeCun** (SELU): `W ~ N(0, √(1/n_in))`

## Tests

```bash
$ make test
# 50+ tests covering activations, layers, losses, network, gradient checking
```

## Requirements

- Python ≥ 3.10
- NumPy ≥ 1.24
- Matplotlib ≥ 3.8 (visualization only)
- Scikit-learn ≥ 1.3 (validation/comparison only — NOT used in NN)

## Documentation

- [Architecture Overview](docs/architecture.md)
- [Backpropagation Math](docs/backpropagation.md)
- [API Reference](docs/api_reference.md)

## License

Educational project — MIT License.
