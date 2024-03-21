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
from typing_extensions import TypeAlias

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr

from jaxtyping import Array, PRNGKeyArray
from lineax import AbstractLinearOperator

from ._samplers import AbstractSampler


TraceEstimatorState: TypeAlias = tuple[AbstractLinearOperator, AbstractSampler]


class AbstractTraceEstimator(eqx.Module, strict=True):
    """Abstract base class for all trace estimators."""

    @abstractmethod
    def compute(
        self,
        key: PRNGKeyArray,
        k: int,
        operator: AbstractLinearOperator,
        sampler: AbstractSampler,
    ) -> tuple[Array, dict[str, Any]]:
        pass


class HutchinsonEstimator(AbstractTraceEstimator):
    def compute(
        self,
        key: PRNGKeyArray,
        k: int,
        operator: AbstractLinearOperator,
        sampler: AbstractSampler,
    ) -> tuple[Array, dict[str, Any]]:
        # sample from proposed distribution
        n = operator.in_size()
        samples = sampler(key, n, k)

        # project to k-dim space
        projected = operator.mv(samples)

        # take the mean across estimates
        trace_est = jnp.sum(projected * samples) / k

        return trace_est, {}


class HutchPlusPlusEstimator(AbstractTraceEstimator):
    def compute(
        self,
        key: PRNGKeyArray,
        k: int,
        operator: AbstractLinearOperator,
        sampler: AbstractSampler,
    ) -> tuple[Array, dict[str, Any]]:
        # generate an n, k matrix X
        n = operator.in_size()
        m = k // 3

        # split X into 2 Xs; X1 and X2, where X1 has shape 2m, where m = k/3
        x1_key, x2_key = jr.split(key)
        X1 = sampler(x1_key, n, m)
        X2 = sampler(x2_key, n, m)

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


class XTraceEstimator(AbstractTraceEstimator):
    scale: bool

    def compute(
        self,
        key: PRNGKeyArray,
        k: int,
        operator: AbstractLinearOperator,
        sampler: AbstractSampler,
    ) -> tuple[Array, dict[str, Any]]:
        n = operator.in_size()
        m = k // 2

        Omega = sampler(key, n, m)
        Y = operator.mv(Omega)
        Q, R = jnp.linalg.qr(Y)

        # solve and rescale
        S = jnp.linalg.inv(R).T
        s = jnp.sqrt(jnp.sum(S**2, axis=0))
        S = S / s
        scale = 1.0

        # working variables
        Z = operator.mv(Q)
        H = Q.T @ Z
        W = Q.T @ Omega
        T = Z.T @ Omega
        HW = H @ W

        SW_d = jnp.sum(S * W, axis=0)
        TW_d = jnp.sum(T * W, axis=0)
        SHS_d = jnp.sum(S * (H @ S), axis=0)
        WHW_d = jnp.sum(W * HW, axis=0)

        term1 = SW_d * jnp.sum((T - H.T @ W) * S, axis=0)
        term2 = (jnp.abs(SW_d) ** 2) * SHS_d
        term3 = jnp.conjugate(SW_d) * jnp.sum(S * (R - HW), axis=0)

        estimates = (
            jnp.trace(H) * jnp.ones(m)
            - SHS_d
            + (WHW_d - TW_d + term1 + term2 + term3) * scale
        )
        trace_est = jnp.mean(estimates)
        std_err = jnp.std(estimates) / jnp.sqrt(m)

        return trace_est, {"std.err": std_err}
