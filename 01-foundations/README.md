# 01 · Foundations

Three neural networks implemented from scratch in NumPy, from a single neuron to a deep
network. Each project has its own code (`model.py`), unit tests, an experiments notebook
and a README with results.

| Project | What is implemented | Data | Key result |
|---|---|---|---|
| [logistic-regression](logistic-regression/numpy) | single neuron: forward/backward pass, gradient descent | cat vs non-cat images | 99.0% train / 70.0% test: strong overfitting (12288 features, 209 images) |
| [shallow-network](shallow-network/numpy) | one tanh hidden layer, backprop, decision boundaries | synthetic "flower" (+ moons, circles) | linear model 47.0% test vs 93.5% for a network with 3-4 hidden units |
| [deep-network](deep-network/numpy) | L-layer ReLU network, small vs He initialization | cat vs non-cat images | 6-layer net: 100% train / 76% test; small init stalls at cost 0.64 (34% test), He reaches 0.0013 |

## Common to all projects

- Forward and backward propagation, cost and gradient descent written by hand (NumPy only).
- Unit tests with `pytest`; experiments in Google Colab; data is not stored in the repository.
- Results are reported on a held-out test set, and small-sample noise is called out
  (the cat datasets have only 50 test images, so 1 image = 2 percentage points).

## Run

Each project's README has Colab and local instructions.