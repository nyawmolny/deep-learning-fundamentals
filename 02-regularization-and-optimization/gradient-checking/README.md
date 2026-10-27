# Gradient checking from scratch (NumPy)

A numerical gradient checker (central differences) for a small L-layer ReLU network, tested on a correct backward pass and on
backward passes with deliberate bugs.

## What is inside

| File | Purpose |
|---|---|
| `gradient_check.py` | `numerical_gradients`, `relative_difference`, `gradient_check` (overall and per-parameter report) |
| `model.py` | small L-layer ReLU network: forward pass, cost, backward pass |
| `bugs.py` | backward passes with deliberate mistakes |
| `test_gradient_check.py` | checker on a known function, correct backprop passes, every bug is caught and localized |
| `gradient_checking.ipynb` | experiments: 1-D example, correct network, bugs |

## Run

The notebook was run in Google Colab; the code is developed locally and pulled into Colab.

```python
!git clone https://github.com/nyawmolny/deep-learning-fundamentals.git
%cd deep-learning-fundamentals/02-regularization-and-optimization/gradient-checking
```

Tests locally, from this directory:

```bash
pip install numpy scikit-learn pytest
pytest
```

## Results

Network `[2, 5, 3, 1]`, 20 points, epsilon = 1e-7, threshold 1e-6.

| Backward pass | Relative difference | Parameters above threshold |
|---|---|---|
| Correct | 2.9e-09 | none |
| Missing 1/m in dW | 8.92e-01 | W1, W2, W3 |
| Missing ReLU mask | 2.57e-01 | W1, b1, W2, b2 |
| Wrong bias gradient | 6.35e-01 | b1, b2, b3 |
| Dropped last-layer term | 3.71e-01 | W3 |

## Takeaways

- The correct backward pass agrees with the numerical gradient to about 3e-09, far below the 1e-6 threshold. Every deliberate bug is off by 0.25–0.9, i.e. five to eight orders of magnitude worse, so all bugs are caught unambiguously.
- The per-parameter report shows where the bug is, not just that there is one: each mistake flags exactly the parameters it touches (wrong sign flags only biases; missing ReLU mask flags hidden layers but not the output layer; dropped term flags only W3), which points to the cause.
- Gradient checking is used to debug the backward pass, not during training: it needs two forward passes per parameter element, which is orders of magnitude more expensive than the backward pass itself.
