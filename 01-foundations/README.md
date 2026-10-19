# 01 · Foundations

Three neural networks implemented from scratch in NumPy, from a single neuron to a deep
network, plus PyTorch versions that are verified against the NumPy code. Each project has
its own code (`model.py`), unit tests, an experiments notebook and a README with results.

| Project | What is implemented | Versions | Data | Key result |
|---|---|---|---|---|
| [logistic-regression](logistic-regression) | single neuron: forward/backward pass, gradient descent | [NumPy](logistic-regression/numpy), [PyTorch](logistic-regression/pytorch) | cat vs non-cat images (NumPy), breast cancer (PyTorch) | NumPy: 99.0% train / 70.0% test, strong overfitting (12288 features, 209 images) |
| [shallow-network](shallow-network/numpy) | one tanh hidden layer, backprop, decision boundaries | [NumPy](shallow-network/numpy); PyTorch covered by deep-network | synthetic "flower" (+ moons, circles) | linear model 47.0% test vs about 93% for a network with 3-4 hidden units |
| [deep-network](deep-network) | L-layer ReLU network, small vs He initialization | [NumPy](deep-network/numpy), [PyTorch](deep-network/pytorch) | cat vs non-cat images | test accuracy 72-76% for 2, 4 and 6 layers (differences within noise); small init stalls at loss 0.644, He reaches 0.0013 |

## Common to all projects

- Forward and backward propagation, cost and gradient descent written by hand in NumPy.
- The PyTorch versions are checked against the NumPy implementation in unit tests (same
  weights and data, outputs, gradients and loss curves agree). The PyTorch deep network
  accepts any list of layer sizes, so it also covers the shallow network.
- Unit tests with `pytest`; experiments in Google Colab; data is not stored in the repository.
- Results are reported on a held-out test set, and small-sample noise is called out
  (for example, the cat dataset has only 50 test images, so 1 image = 2 percentage points).

## Run

Each project's README has Colab and local instructions.