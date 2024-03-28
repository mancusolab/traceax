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

import equinox as eqx
import jax.numpy as jnp
import jax.random as rdm

from jaxtyping import Array, Inexact, PRNGKeyArray


class AbstractSampler(eqx.Module, strict=True):
    """Abstract base class for all samplers."""

    @abstractmethod
    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        r"""Sample random variates from the underlying distribution as an $n \times k$
        matrix.

        !!! Example

            ```python
            sampler = tr.RademacherSampler()
            samples = sampler(key, n, k)
            ```
        **Arguments:**

        - `key`: a jax PRNG key used as the random key.
        - `n`: the size of the leading dimension.
        - `k`: the size of the trailing dimension.

        **Returns**:

        An Array of random samples.
        """
        ...


class NormalSampler(AbstractSampler, strict=True):
    r"""Standard normal distribution sampler.

    Generates samples $X_{ij} \sim N(0, 1)$ for $i \in [n]$ and $j \in [k]$.
    """

    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        return rdm.normal(key, (n, k))


class SphereSampler(AbstractSampler, strict=True):
    r"""Sphere distribution sampler.

    Generates samples $X_1, \dotsc, X_n$ uniformly distributed on the surface of a
    $k$ dimensional sphere (i.e. $k-1$-sphere) with radius $\sqrt{n}$. Internally,
    this operates by sampling standard normal variates, and then rescaling such that
    each $k$-vector $X_i$ has $\lVert X_i \rVert = \sqrt{n}$.
    """

    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        samples = rdm.normal(key, (n, k))
        return jnp.sqrt(n) * (samples / jnp.linalg.norm(samples, axis=0))


class ComplexNormalSampler(AbstractSampler, strict=True):
    r"""Standard complex normal distribution sampler.

    Generates complex-valued samples $X_{ij} = A_{ij} + i B_{ij}$ where
    $A_{ij} \sim N(0, 1)$ and $B_{ij} \sim N(0, 1)$ for $i \in [n]$ and $j \in [k]$.
    """

    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        samples = rdm.normal(key, (n, k)) + 1j * rdm.normal(key, (n, k))
        return samples / jnp.sqrt(2)


class ComplexSphereSampler(AbstractSampler, strict=True):
    r"""Complex sphere distribution sampler.

    Generates complex-valued samples $X_1, \dotsc, X_n$ uniformly distributed on the
    surface of a $k$ dimensional complex-valued sphere with radius $\sqrt{n}$. Internally,
    this operates by sampling standard complex normal variates, and then rescaling such
    that each complex-valued $k$-vector $X_i$ has $\lVert X_i \rVert = \sqrt{n}$.
    """

    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        samples = rdm.normal(key, (n, k)) + 1j * rdm.normal(key, (n, k))
        return jnp.sqrt(n) * (samples / jnp.linalg.norm(samples, axis=0))


class RademacherSampler(AbstractSampler, strict=True):
    r"""Rademacher distribution sampler.

    Generates samples $X_{ij} \sim \mathcal{U}(-1, +1)$ for $i \in [n]$ and $j \in [k]$.
    """

    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        return rdm.rademacher(key, (n, k))
