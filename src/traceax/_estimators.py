#  Copyright (c) 2024 MancusoLab.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from abc import abstractmethod
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import jax.scipy as jsp

from equinox import AbstractVar
from jax.numpy.linalg import norm
from jaxtyping import Array, PRNGKeyArray
from lineax import AbstractLinearOperator, is_negative_semidefinite, is_positive_semidefinite

from ._samplers import AbstractSampler, RademacherSampler, SphereSampler


def _check_shapes(operator: AbstractLinearOperator, k: int) -> tuple[int, int]:
    n_in = operator.in_size()
    n_out = operator.out_size()
    if n_in != n_out:
        raise ValueError(f"Trace estimation requires square linear operator. Found {(n_out, n_in)}.")

    if k < 1:
        raise ValueError(f"Trace estimation requires positive number of matvecs. Found {k}.")

    return n_in, k


def _get_scale(W: Array, D: Array, n: int, k: int) -> Array:
    return (n - k + 1) / (n - norm(W, axis=0) ** 2 + jnp.abs(D) ** 2)


class AbstractTraceEstimator(eqx.Module, strict=True):
    r"""Abstract base class for all trace estimators."""

    sampler: AbstractVar[AbstractSampler]

    @abstractmethod
    def estimate(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        """Estimate the trace of `operator`.

        !!! Example

            ```python
            key = jax.random.PRNGKey(...)
            operator = lx.MatrixLinearOperator(...)
            result = hutch.compute(key, operator, k=10)
            #  or
            result = hutch(key, operator, k=10)
            ```

        **Arguments:**

        - `key`: the PRNG key used as the random key for sampling.
        - `operator`: the (square) linear operator for which the trace is to be estimated.
        - `k`: the number of matrix vector operations to perform for trace estimation.

        **Returns:**

        A two-tuple of:

        - The trace estimate.
        - A dictionary of any extra statistics above the trace, e.g., the standard error.
        """
        ...

    def __call__(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        """An alias for `estimate`."""
        return self.estimate(key, operator, k)


class HutchinsonEstimator(AbstractTraceEstimator):
    r"""Girard-Hutchinson Trace Estimator:

    $\mathbb{E}[\omega^T \mathbf{A} \omega] = \text{trace}(\mathbf{A})$,
    where $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = RademacherSampler()

    def estimate(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        n, k = _check_shapes(operator, k)
        # sample from proposed distribution
        samples = self.sampler(key, n, k)

        # project to k-dim space
        projected = operator.mv(samples)

        # take the mean across estimates
        trace_est = jnp.sum(projected * samples) / k

        return trace_est, {}


HutchinsonEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: The sampling distribution for $\omega$. Default is [`traceax.RademacherSampler`][].
"""


class HutchPlusPlusEstimator(AbstractTraceEstimator):
    r"""Hutch++ Trace Estimator:

    Let $\hat{\mathbf{A}} := \mathbf{Q}\mathbf{Q}^* \mathbf{A}$ be the a _low-rank approximation_
    to $\mathbf{A}$, where $\mathbf{Q}$ is the orthonormal basis of $\mathbf{A} \Omega$, for
    $\Omega = [\omega_1, \dotsc, \omega_k]$.

    Hutch++ improves upon Girard-Hutchinson estimator by including the trace of the residuals. Namely,
    Hutch++ estimates $\text{trace}(\mathbf{A})$ as
    $\text{trace}(\hat{\mathbf{A}}) - \text{trace}(\mathbf{A} - \hat{\mathbf{A}})$.

    As with the Girard-Hutchinson estimator, it requires
    $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = RademacherSampler()

    def estimate(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        # generate an n, k matrix X
        n, k = _check_shapes(operator, k)
        m = k // 3

        # split X into 2 Xs; X1 and X2, where X1 has shape 2m, where m = k/3
        samples = self.sampler(key, n, 2 * m)
        X1 = samples[:, :m]
        X2 = samples[:, m:]

        Y = operator.mv(X1)

        # compute Q, _ = QR(Y) (orthogonal matrix)
        Q, _ = jnp.linalg.qr(Y)

        # compute G = X2 - Q @ (Q.T @ X2)
        G = X2 - Q @ (Q.T @ X2)

        # estimate trace = tr(Q.T @ A @ Q) + tr(G.T @ A @ G) / k
        AQ = operator.mv(Q)
        AG = operator.mv(G)
        trace_est = jnp.sum(AQ * Q) + jnp.sum(AG * G) / (G.shape[1])

        return trace_est, {}


HutchPlusPlusEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: The sampling distribution for $\omega$. Default is [`traceax.RademacherSampler`][].
"""


class XTraceEstimator(AbstractTraceEstimator):
    r"""XTrace Trace Estimator:

    Let $\hat{\mathbf{A}} := \mathbf{Q}\mathbf{Q}^* \mathbf{A}$ be the the _low-rank approximation_
    to $\mathbf{A}$, where $\mathbf{Q}$ is the orthonormal basis of $\mathbf{A} \Omega$, for
    $\Omega = [\omega_1, \dotsc, \omega_k]$.

    XTrace improves upon Hutch++ estimator by enforcing *exchangeability* of sampled test-vectors,
    to construct a symmetric estimation function with lower variance.

    Additionally, the *improved* XTrace algorithm (i.e. `improved = True`), ensures that test-vectors
    are orthogonalized against the low rank approximation $\mathbf{Q}\mathbf{Q}^* \mathbf{A}$ and
    renormalized. This improved XTrace approach may provide better empirical results compared with
    the non-orthogonalized version.

    As with the Girard-Hutchinson estimator, it requires
    $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = SphereSampler()
    improved: bool = True

    def estimate(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        n, k = _check_shapes(operator, k)
        m = k // 2

        samples = self.sampler(key, n, m)
        Y = operator.mv(samples)
        Q, R = jnp.linalg.qr(Y)

        # solve and rescale
        S = jnp.linalg.inv(R).T
        s = norm(S, axis=0)
        S = S / s

        # working variables
        Z = operator.mv(Q)
        H = Q.T @ Z
        W = Q.T @ samples
        T = Z.T @ samples
        HW = H @ W

        SW_d = jnp.sum(S * W, axis=0)
        TW_d = jnp.sum(T * W, axis=0)
        SHS_d = jnp.sum(S * (H @ S), axis=0)
        WHW_d = jnp.sum(W * HW, axis=0)

        term1 = SW_d * jnp.sum((T - H.T @ W) * S, axis=0)
        term2 = (jnp.abs(SW_d) ** 2) * SHS_d
        term3 = jnp.conjugate(SW_d) * jnp.sum(S * (R - HW), axis=0)

        if self.improved:
            scale = _get_scale(W, SW_d, n, k)
        else:
            scale = 1

        estimates = jnp.trace(H) * jnp.ones(m) - SHS_d + (WHW_d - TW_d + term1 + term2 + term3) * scale
        trace_est = jnp.mean(estimates)
        std_err = jnp.std(estimates) / jnp.sqrt(m)

        return trace_est, {"std.err": std_err}


XTraceEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: the sampling distribution for $\omega$. Default is [`traceax.SphereSampler`][].
- `improved`: whether to use the *improved* XTrace estimator, which rescales predicted samples.
    Default is `True` (see Notes).
"""


class XNysTraceEstimator(AbstractTraceEstimator):
    r"""XNysTrace Trace Estimator:

    XNysTrace improves upon XTrace estimator when $\mathbf{A}$ is (negative-) positive-semidefinite, by
    performing a [Nyström approximation](https://en.wikipedia.org/wiki/Low-rank_matrix_approximations#Nystr%C3%B6m_approximation),
    rather than a randomized SVD (i.e., random projection followed by QR decomposition).

    Like, [`traceax.XTraceEstimator`][], the *improved* XNysTrace algorithm (i.e. `improved = True`), ensures
    that test-vectors are orthogonalized against the low rank approximation and renormalized.
    This improved XNysTrace approach may provide better empirical results compared with the non-orthogonalized version.

    As with the Girard-Hutchinson estimator, it requires
    $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = SphereSampler()
    improved: bool = True

    def estimate(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        is_nsd = is_negative_semidefinite(operator)
        if not (is_positive_semidefinite(operator) | is_nsd):
            raise ValueError("`XNysTraceEstimator` may only be used for positive or negative definite linear operators")
        if is_nsd:
            operator = -operator

        n, k = _check_shapes(operator, k)
        samples = self.sampler(key, n, k)
        Y = operator.mv(samples)

        # shift for numerical issues
        nu = jnp.finfo(Y.dtype).eps * norm(Y, "fro") / jnp.sqrt(n)
        Y = Y + samples * nu
        Q, R = jnp.linalg.qr(Y)

        # compute and symmetrize H, then take cholesky factor
        H = samples.T @ Y
        C = jnp.linalg.cholesky(0.5 * (H + H.T)).T
        B = jsp.linalg.solve_triangular(C.T, R.T, lower=True).T

        # if improved == True
        Qs, Rs = jnp.linalg.qr(samples)
        Ws = Qs.T @ samples

        # solve and rescale
        if self.improved:
            S = jnp.linalg.inv(Rs).T
            s = norm(S, axis=0)
            S = S / s
            scale = _get_scale(Ws, jnp.sum(S * Ws, axis=0), n, k)
        else:
            scale = 1

        W = Q.T @ samples
        S = jsp.linalg.solve_triangular(C, B.T).T / jnp.sqrt(jnp.diag(jnp.linalg.inv(H)))
        dSW = jnp.sum(S * W, axis=0)

        estimates = norm(B, "fro") ** 2 - norm(S, axis=0) ** 2 + (jnp.abs(dSW) ** 2) * scale - nu * n
        trace_est = jnp.mean(estimates)
        std_err = jnp.std(estimates) / jnp.sqrt(k)
        trace_est = jnp.where(is_nsd, -trace_est, trace_est)

        return trace_est, {"std.err": std_err}


XNysTraceEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: the sampling distribution for $\omega$. Default is [`traceax.SphereSampler`][].
- `improved`: whether to use the *improved* XNysTrace estimator, which rescales predicted samples.
    Default is `True` (see Notes).
"""
