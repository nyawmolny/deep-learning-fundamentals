import numpy as np
from data import load_breast_cancer_split, majority_baseline


def test_split_shapes_and_labels():
    X_train, X_test, y_train, y_test = load_breast_cancer_split()
    assert X_train.shape == (455, 30) and X_test.shape == (114, 30)
    assert y_train.shape == (455,) and y_test.shape == (114,)
    assert set(np.unique(y_train)) == {0.0, 1.0}


def test_train_features_are_standardized():
    X_train, _, _, _ = load_breast_cancer_split()
    np.testing.assert_allclose(X_train.mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(X_train.std(axis=0), 1.0, atol=1e-8)


def test_majority_baseline():
    assert majority_baseline(np.array([1.0, 1.0, 0.0]), np.array([1.0, 0.0, 1.0, 1.0])) == 0.75
