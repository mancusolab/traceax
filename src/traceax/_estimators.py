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

from equinox import AbstractVar
from jaxtyping import Array, PRNGKeyArray
from lineax import AbstractLinearOperator

from ._samplers import AbstractSampler, RademacherSampler, SphereSampler


def _get_shape(operator: AbstractLinearOperator) -> int:
    n_in = operator.in_size()
    n_out = operator.out_size()
    if n_in != n_out:
        raise ValueError(f"Trace estimation requires square linear operator. Found {(n_out, n_in)}")

    return n_in


class AbstractTraceEstimator(eqx.Module, strict=True):
    r"""Abstract base class for all trace estimators."""

    sampler: AbstractVar[AbstractSampler]

    @abstractmethod
    def compute(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        ...

    def __call__(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        return self.compute(key, operator, k)


class HutchinsonEstimator(AbstractTraceEstimator):
    r"""Girard-Hutchinson Trace Estimator:

    $\mathbb{E}[\omega^T \mathbf{A} \omega] = \text{trace}(\mathbf{A})$,
    where $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    !!! info


    """

    sampler: AbstractSampler = RademacherSampler()

    def compute(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        n = _get_shape(operator)
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

    Let $\hat{\mathbf{A}} := \mathbf{Q}\mathbf{Q}^* \mathbf{A}$ be the the _low-rank approximation_
    to $\mathbf{A}$, where $\mathbf{Q}$ is the orthonormal basis of $\mathbf{A} \Omega$, for
    $\Omega = [\omega_1, \dotsc, \omega_k]$.

    Hutch++ improves upon Girard-Hutchinson estimator by including the trace of the residuals. Namely,
    Hutch++ estimates $\text{trace}(\mathbf{A})$ as
    $\text{trace}(\hat{\mathbf{A}}) - \text{trace}(\mathbf{A} - \hat{\mathbf{A}})$.

    As with the Girard-Hutchinson estimator, it requires
    $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    !!! info

    """

    sampler: AbstractSampler = RademacherSampler()

    def compute(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        # generate an n, k matrix X
        n = _get_shape(operator)
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
    r""" """

    sampler: AbstractSampler = SphereSampler()
    rescale: bool = True

    def compute(self, key: PRNGKeyArray, operator: AbstractLinearOperator, k: int) -> tuple[Array, dict[str, Any]]:
        n = _get_shape(operator)
        m = k // 2

        samples = self.sampler(key, n, m)
        Y = operator.mv(samples)
        Q, R = jnp.linalg.qr(Y)

        # solve and rescale
        S = jnp.linalg.inv(R).T
        s = jnp.sqrt(jnp.sum(S**2, axis=0))
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

        re_vals = n - jnp.linalg.norm(W, axis=0) ** 2 + jnp.abs(SW_d * jnp.linalg.norm(S, axis=0)) ** 2
        scale = jnp.where(self.rescale, (n - m + 1) / re_vals, 1.0)

        estimates = jnp.trace(H) * jnp.ones(m) - SHS_d + (WHW_d - TW_d + term1 + term2 + term3) * scale
        trace_est = jnp.mean(estimates)
        std_err = jnp.std(estimates) / jnp.sqrt(m)

        return trace_est, {"std.err": std_err}


XTraceEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: The sampling distribution for $\omega$. Default is [`traceax.SphereSampler`][].
- `rescale`: Whether to rescale samples for _improved_ XTrace estimator (see Notes).
"""
