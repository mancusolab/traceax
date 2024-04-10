# Stochastic Trace Estimators

Given a square linear operator $\mathbf{A}$, the `trace` of $\mathbf{A}$ is defined as,

$$\text{trace}(\mathbf{A}) = \sum_i \mathbf{A}_{ii}.$$

When $\mathbf{A}$ is represented in memory as an $n \times n$ matrix, computing
the `trace` is straightforward, only requiring $O(n)$ time to sum along the
diagonal. However, in practice, $\mathbf{A}$ can be the result of many operations
and explicit calculation and represention of $\mathbf{A}$ may be prohibitive.

Given this, we may represent $\mathbf{A}$ as a linear operator, which can be viewed
as a lazy representation of $\mathbf{A}$ that only tracks the underlying operations
to calculate its final result. As such, matrix vector products between $\mathbf{A}$ and
vector $\omega$ can be obtained by lazily evaluating the chain of underlying
composition operations with intermediate matrix-vector products
(e.g., [lineax](https://github.com/patrick-kidger/lineax)).

There is a rich history of *stochastic* trace estimation for matrices where
one can estimate the `trace` of $\mathbf{A}$ using multiple matrix-vector products
followed by averaging. To see this, observe that

$$\mathbb{E}[\omega^T \mathbf{A} \omega] = \text{trace}(\mathbf{A}),$$

where $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.
The above is known as the Girard-Hutchinson estimator. There have been multiple
advancements in stochastic `trace` estimation. Here, `traceax` aims to provide an
easy-to-use API for stochastic trace estimation that leverages the flexibility of
[lineax](https://github.com/patrick-kidger/lineax) linear operators together with
differentiable and performant [JAX](https://github.com/google/jax) based numerics.

??? abstract "`traceax.AbstractTraceEstimator`"

    ::: traceax.AbstractTraceEstimator
        options:
            show_bases: false
            members:
            - estimate
            - __call__

::: traceax.HutchinsonEstimator
    options:
        members:
        - __init__

---

::: traceax.HutchPlusPlusEstimator
    options:
        members:
        - __init__

---

::: traceax.XTraceEstimator
    options:
        members:
        - __init__
