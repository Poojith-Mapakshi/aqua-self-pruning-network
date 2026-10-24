"""Minimal NumPy reverse-mode autodiff engine.

This is an intentionally small educational engine. Gradients are accumulated
through a dynamically constructed graph during Tensor.backward().
"""

from __future__ import annotations

from typing import Callable, Iterable
import numpy as np


def _as_array(value) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.data
    return np.asarray(value, dtype=np.float64)


def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Sum a broadcasted gradient back to the original operand shape."""
    grad = np.asarray(grad, dtype=np.float64)
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    for axis, size in enumerate(target_shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)


class Tensor:
    """An ndarray plus gradient metadata and a reverse-mode computation graph."""

    def __init__(
        self,
        data,
        requires_grad: bool = False,
        _children: Iterable["Tensor"] = (),
        _backward: Callable[[], None] | None = None,
        _op: str = "",
    ):
        self.data = _as_array(data).copy()
        self.requires_grad = bool(requires_grad)
        self.grad = np.zeros_like(self.data) if self.requires_grad else None
        self._prev = set(_children)
        self._backward_fn = _backward or (lambda: None)
        self._op = _op

    @property
    def shape(self):
        return self.data.shape

    def __repr__(self):
        return f"Tensor(data={self.data!r}, requires_grad={self.requires_grad})"

    def _accumulate_grad(self, grad):
        if not self.requires_grad:
            return
        grad = np.asarray(grad, dtype=np.float64)
        if grad.shape != self.data.shape:
            raise ValueError(f"Gradient shape {grad.shape} != tensor shape {self.data.shape}")
        self.grad += grad

    def backward(self, grad=None):
        """Backpropagate from this tensor, accumulating gradients on leaf tensors.

        If this tensor is non-scalar, the caller must supply the vector-Jacobian
        product seed `grad`, which must match this tensor's shape.
        """
        if not self.requires_grad:
            raise RuntimeError("backward() requires a tensor with requires_grad=True")
        if grad is None:
            if self.data.size != 1:
                raise ValueError("grad must be supplied when backward() starts from a non-scalar")
            grad = np.ones_like(self.data)
        else:
            grad = np.asarray(grad, dtype=np.float64)
            if grad.shape != self.data.shape:
                raise ValueError(f"Seed gradient shape {grad.shape} != output shape {self.data.shape}")

        topo = []
        visited = set()

        def build(node):
            if node not in visited:
                visited.add(node)
                for parent in node._prev:
                    build(parent)
                topo.append(node)

        build(self)
        # Reset intermediate gradients to make repeated backward calls represent
        # a fresh traversal. Leaf gradients are also reset intentionally.
        for node in topo:
            if node.requires_grad:
                node.grad = np.zeros_like(node.data)
        self.grad = grad.copy()

        for node in reversed(topo):
            node._backward_fn()

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, self.requires_grad or other.requires_grad, (self, other), _op="add")

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad, other.shape))
        out._backward_fn = _backward
        return out

    __radd__ = __add__

    def __neg__(self):
        out = Tensor(-self.data, self.requires_grad, (self,), _op="neg")

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(-out.grad)
        out._backward_fn = _backward
        return out

    def __sub__(self, other):
        return self + (-other if isinstance(other, Tensor) else -np.asarray(other))

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, self.requires_grad or other.requires_grad, (self, other), _op="mul")

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad * other.data, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(out.grad * self.data, other.shape))
        out._backward_fn = _backward
        return out

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data / other.data, self.requires_grad or other.requires_grad, (self, other), _op="div")

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(_unbroadcast(out.grad / other.data, self.shape))
            if other.requires_grad:
                other._accumulate_grad(_unbroadcast(-out.grad * self.data / (other.data ** 2), other.shape))
        out._backward_fn = _backward
        return out

    def __rtruediv__(self, other):
        return (other if isinstance(other, Tensor) else Tensor(other)) / self

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad, (self, other), _op="matmul")

        def _backward():
            # The initial implementation intentionally supports the 2-D matrix
            # multiplication used by Linear layers.
            if self.data.ndim != 2 or other.data.ndim != 2:
                raise NotImplementedError("matmul backward currently supports 2-D matrices only")
            if self.requires_grad:
                self._accumulate_grad(out.grad @ other.data.T)
            if other.requires_grad:
                other._accumulate_grad(self.data.T @ out.grad)
        out._backward_fn = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,), _op="sum")

        def _backward():
            if not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                grad = np.broadcast_to(grad, self.shape)
            else:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                axes = tuple(a if a >= 0 else a + self.data.ndim for a in axes)
                if not keepdims:
                    for a in sorted(axes):
                        grad = np.expand_dims(grad, axis=a)
                grad = np.broadcast_to(grad, self.shape)
            self._accumulate_grad(grad)
        out._backward_fn = _backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else tuple(axis)
            count = int(np.prod([self.shape[a] for a in axes]))
        return self.sum(axis=axis, keepdims=keepdims) / count

    def relu(self):
        out = Tensor(np.maximum(self.data, 0.0), self.requires_grad, (self,), _op="relu")

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (self.data > 0))
        out._backward_fn = _backward
        return out

    def tanh(self):
        values = np.tanh(self.data)
        out = Tensor(values, self.requires_grad, (self,), _op="tanh")

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * (1.0 - values ** 2))
        out._backward_fn = _backward
        return out

    def sigmoid(self):
        # Stable sigmoid: separate branches avoid overflow for large magnitudes.
        values = np.empty_like(self.data, dtype=np.float64)
        positive = self.data >= 0
        values[positive] = 1.0 / (1.0 + np.exp(-self.data[positive]))
        exp_x = np.exp(self.data[~positive])
        values[~positive] = exp_x / (1.0 + exp_x)
        out = Tensor(values, self.requires_grad, (self,), _op="sigmoid")

        def _backward():
            if self.requires_grad:
                self._accumulate_grad(out.grad * values * (1.0 - values))
        out._backward_fn = _backward
        return out


def softmax_cross_entropy(logits: Tensor, labels: np.ndarray) -> Tensor:
    """Mean softmax cross-entropy for logits (N, C) and integer labels (N,)."""
    if logits.data.ndim != 2:
        raise ValueError("logits must have shape (batch, classes)")
    labels = np.asarray(labels, dtype=np.int64)
    if labels.shape != (logits.data.shape[0],):
        raise ValueError("labels must have shape (batch,)")
    if np.any(labels < 0) or np.any(labels >= logits.data.shape[1]):
        raise ValueError("labels contain an invalid class index")

    shifted = logits.data - np.max(logits.data, axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    n = logits.data.shape[0]
    loss_value = -np.mean(np.log(np.maximum(probabilities[np.arange(n), labels], 1e-300)))
    out = Tensor(loss_value, logits.requires_grad, (logits,), _op="softmax_cross_entropy")

    def _backward():
        if logits.requires_grad:
            grad = probabilities.copy()
            grad[np.arange(n), labels] -= 1.0
            logits._accumulate_grad(out.grad * grad / n)
    out._backward_fn = _backward
    return out
