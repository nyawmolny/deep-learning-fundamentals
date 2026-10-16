import numpy as np
import pytest
from data import make_flower, make_sklearn_dataset

# ---------- make_flower ----------

def test_flower_labels_are_binary_and_balanced():
    _, Y = make_flower(400)
    assert set(np.unique(Y)) == {0.0, 1.0}
    assert Y.sum() == 200


def test_flower_without_noise_stays_within_radius():
    X, _ = make_flower(200, radius=4.0, noise=0.0)
    assert np.hypot(X[0], X[1]).max() <= 4.0 + 1e-9


def test_flower_radius_scales_points_when_noise_is_zero():
    X1, _ = make_flower(200, radius=1.0, noise=0.0)
    X3, _ = make_flower(200, radius=3.0, noise=0.0)
    np.testing.assert_allclose(X3, 3.0 * X1)

# ---------- make_sklearn_dataset ----------

@pytest.mark.parametrize("name", ["moons", "circles"])
def test_sklearn_dataset_shapes_and_labels(name):
    X, Y = make_sklearn_dataset(name, 300)
    assert X.shape == (2, 300)
    assert Y.shape == (1, 300)
    assert set(np.unique(Y)) == {0.0, 1.0}
