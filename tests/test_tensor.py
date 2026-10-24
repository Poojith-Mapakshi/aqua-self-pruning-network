import numpy as np
import pytest

from engine.tensor import Tensor, softmax_cross_entropy


def numerical_gradient(fn, x, eps=1e-6):
    grad = np.zeros_like(x, dtype=np.float64)
    for index in np.ndindex(x.shape):
        original = x[index]
        x[index] = original + eps
        plus = fn(x)
        x[index] = original - eps
        minus = fn(x)
        x[index] = original
        grad[index] = (plus - minus) / (2 * eps)
    return grad


def test_broadcast_add_gradient():
    a_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b_data = np.array([0.2, -0.3, 0.7])
    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    loss = ((a + b) * (a + b)).mean()
    loss.backward()
    expected_a = numerical_gradient(lambda v: np.mean((v + b_data) ** 2), a_data.copy())
    expected_b = numerical_gradient(lambda v: np.mean((a_data + v) ** 2), b_data.copy())
    np.testing.assert_allclose(a.grad, expected_a, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(b.grad, expected_b, rtol=1e-5, atol=1e-6)


def test_matmul_gradient():
    a_data = np.array([[0.2, -0.4], [0.7, 0.1]])
    b_data = np.array([[0.3, 0.8], [-0.5, 0.6]])
    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    loss = (a @ b).mean()
    loss.backward()
    expected_a = numerical_gradient(lambda v: np.mean(v @ b_data), a_data.copy())
    expected_b = numerical_gradient(lambda v: np.mean(a_data @ v), b_data.copy())
    np.testing.assert_allclose(a.grad, expected_a, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(b.grad, expected_b, rtol=1e-5, atol=1e-6)


def test_relu_tanh_and_division_gradients():
    x_data = np.array([-0.7, 0.2, 1.1])
    x = Tensor(x_data, requires_grad=True)
    loss = (x.relu() + x.tanh()) .mean()
    loss.backward()
    expected = numerical_gradient(lambda v: np.mean(np.maximum(v, 0) + np.tanh(v)), x_data.copy())
    np.testing.assert_allclose(x.grad, expected, rtol=1e-5, atol=1e-6)

    a_data = np.array([1.2, 2.1, 3.4])
    b_data = np.array([0.8, 1.3, 2.0])
    a = Tensor(a_data, requires_grad=True)
    b = Tensor(b_data, requires_grad=True)
    (a / b).sum().backward()
    expected_a = numerical_gradient(lambda v: np.sum(v / b_data), a_data.copy())
    expected_b = numerical_gradient(lambda v: np.sum(a_data / v), b_data.copy())
    np.testing.assert_allclose(a.grad, expected_a, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(b.grad, expected_b, rtol=1e-5, atol=1e-6)


def test_softmax_cross_entropy_gradient():
    logits_data = np.array([[1.0, 0.2, -0.4], [-0.1, 0.8, 0.3]])
    labels = np.array([0, 2])
    logits = Tensor(logits_data, requires_grad=True)
    loss = softmax_cross_entropy(logits, labels)
    loss.backward()
    expected = numerical_gradient(
        lambda v: float(softmax_cross_entropy(Tensor(v), labels).data),
        logits_data.copy(),
    )
    np.testing.assert_allclose(logits.grad, expected, rtol=1e-5, atol=1e-6)


def test_backward_requires_seed_for_non_scalar():
    x = Tensor([1.0, 2.0], requires_grad=True)
    with pytest.raises(ValueError):
        x.backward()
