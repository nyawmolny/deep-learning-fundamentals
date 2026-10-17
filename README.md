# Deep Learning Fundamentals

My personal projects, inspired by the programming assignments of the Deep Learning
Specialization by DeepLearning.AI (Andrew Ng). I reimplemented the ideas in my own
structure: neural networks from scratch in NumPy, with unit tests, experiments and
written-up results for every project.

## Contents

| Section | Projects |
|---|---|
| [01-foundations](01-foundations) | logistic regression, shallow network, deep network |

## 01-foundations at a glance

- **[logistic-regression](01-foundations/logistic-regression/numpy):** a single neuron on cat vs non-cat images; shows strong overfitting (99.0% train / 70.0% test).
- **[shallow-network](01-foundations/shallow-network/numpy):** one hidden layer on planar data; a linear model gets 47.0% test, a network with 3-4 hidden units about 93.5%.
- **[deep-network](01-foundations/deep-network/numpy):** an L-layer ReLU network; He initialization trains, small initialization stalls.

## How each project is organized

- `model.py`: the implementation (NumPy only).
- `data.py`: data loading and preprocessing.
- `test_model.py` (and `test_data.py` where relevant): unit tests, run with `pytest`.
- A notebook with experiments, run in Google Colab.
- A README with setup, results and takeaways.

Datasets are not stored in the repository; each project README explains how to get them.

## Stack

Python, NumPy, Matplotlib, scikit-learn (datasets and a baseline), pytest.