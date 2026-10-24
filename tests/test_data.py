import numpy as np

from train.data import make_spirals


def test_spiral_dataset_shapes_and_labels():
    X, y = make_spirals(samples_per_class=20, seed=7)
    assert X.shape == (60, 2)
    assert y.shape == (60,)
    assert set(np.unique(y)) == {0, 1, 2}
    assert np.isfinite(X).all()


def test_spiral_dataset_is_reproducible():
    X1, y1 = make_spirals(samples_per_class=15, seed=123)
    X2, y2 = make_spirals(samples_per_class=15, seed=123)
    np.testing.assert_array_equal(X1, X2)
    np.testing.assert_array_equal(y1, y2)
