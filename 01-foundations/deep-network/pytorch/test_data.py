import numpy as np
from data import flatten_and_normalize


def test_flatten_and_normalize_shape_and_range():
    images = np.random.default_rng(0).integers(0, 256, size=(5, 4, 4, 3), dtype=np.uint8)
    X = flatten_and_normalize(images)
    assert X.shape == (5, 48)
    assert X.min() >= 0.0 and X.max() <= 1.0


def test_flatten_keeps_example_order():
    images = np.zeros((2, 2, 2, 3), dtype=np.uint8)
    images[1] = 255
    X = flatten_and_normalize(images)
    assert X[0].sum() == 0.0 and X[1].min() == 1.0
