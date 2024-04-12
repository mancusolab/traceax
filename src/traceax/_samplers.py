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

from equinox import AbstractVar
from jax.dtypes import canonicalize_dtype
from jax.numpy import issubdtype
from jaxtyping import Array, DTypeLike, Inexact, Num, PRNGKeyArray


class AbstractSampler(eqx.Module, strict=True):
    """Abstract base class for all samplers."""

    dtype: AbstractVar[DTypeLike]

    @abstractmethod
    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Num[Array, "n k"]:
        r"""Sample random variates from the underlying distribution as an $n \times k$
        matrix.

        !!! Example

            ```python
            sampler = tr.RademacherSampler()
            samples = sampler(key, n, k)
            ```

        Each sampler accepts a `dtype` (i.e. `float`, `complex`, `int`) argument upon initialization,
        with sensible default values. This makes it possible to sample from more general spaces (e.g.,
        complex Normal test-vectors).

        !!! Example

            ```python
            sampler = tr.NormalSampler(complex)
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

    !!! Note
        Supports float and complex-valued types.
    """

    dtype: DTypeLike = eqx.field(converter=canonicalize_dtype, default=float)

    def __check_init__(self):
        if not issubdtype(self.dtype, jnp.inexact):
            raise ValueError(f"NormalSampler requires float or complex dtype. Found {self.dtype}.")

    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        return rdm.normal(key, (n, k), self.dtype)


NormalSampler.__init__.__doc__ = r"""**Arguments:**

- `dtype`: numeric representation for sampled test-vectors. Default is `float`.
"""


class SphereSampler(AbstractSampler, strict=True):
    r"""Sphere distribution sampler.

    Generates samples $X_1, \dotsc, X_n$ uniformly distributed on the surface of a
    $k$ dimensional sphere (i.e. $k-1$-sphere) with radius $\sqrt{n}$. Internally,
    this operates by sampling standard normal variates, and then rescaling such that
    each $k$-vector $X_i$ has $\lVert X_i \rVert = \sqrt{n}$.

    !!! Note
        Supports float and complex-valued types.
    """

    dtype: DTypeLike = eqx.field(converter=canonicalize_dtype, default=float)

    def __check_init__(self):
        if not issubdtype(self.dtype, jnp.inexact):
            raise ValueError(f"SphereSampler requires float or complex dtype. Found {self.dtype}.")

    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        samples = rdm.normal(key, (n, k), self.dtype)
        return jnp.sqrt(n) * (samples / jnp.linalg.norm(samples, axis=0))


SphereSampler.__init__.__doc__ = r"""**Arguments:**

- `dtype`: numeric representation for sampled test-vectors. Default is `float`.
"""


class RademacherSampler(AbstractSampler, strict=True):
    r"""Rademacher distribution sampler.

    Generates samples $X_{ij} \sim \mathcal{U}(-1, +1)$ for $i \in [n]$ and $j \in [k]$.

    !!! Note
        Supports integer, float, and complex-valued types.
    """

    dtype: DTypeLike = eqx.field(converter=canonicalize_dtype, default=int)

    def __check_init__(self):
        if not issubdtype(self.dtype, jnp.number):
            raise ValueError(f"RademacherSampler requires numeric dtype. Found {self.dtype}.")

    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Num[Array, "n k"]:
        return rdm.rademacher(key, (n, k), self.dtype)


RademacherSampler.__init__.__doc__ = r"""**Arguments:**

- `dtype`: numeric representation for sampled test-vectors. Default is `int`.
"""
