# Design Notes

This document will be expanded alongside the implementation. Each design decision should be supported by derivation, tests, or measured evidence.

## Dataset

We begin with a deterministic, synthetic three-class spiral classification dataset generated with NumPy. Each sample has two features and an integer class label in `{0, 1, 2}`. This keeps iteration fast while we validate the autodiff engine. The dataset is a development benchmark, not evidence that the method generalizes to all datasets.

## Planned technical decisions to document

- Reverse-mode autodiff and gradient accumulation.
- Broadcasting-aware gradient reduction.
- The meaning of gradients for masked weights.
- Adam state behavior for pruned and revived connections.
- The pruning importance criterion and its approximation assumptions.
- Sparse-aware inference and how compute savings are measured.
- Limitations, seed variance, and reproducibility.
