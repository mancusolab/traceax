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
    @abstractmethod
    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        ...


class NormalSampler(AbstractSampler, strict=True):
    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        return rdm.normal(key, (n, k))


class SphereSampler(AbstractSampler, strict=True):
    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        samples = rdm.normal(key, (n, k))
        return jnp.sqrt(n) * (samples / jnp.linalg.norm(samples, axis=0))


class ComplexNormalSampler(AbstractSampler, strict=True):
    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        samples = rdm.normal(key, (n, k)) + 1j * rdm.normal(key, (n, k))
        return samples / jnp.sqrt(2)


class ComplexSphereSampler(AbstractSampler, strict=True):
    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        samples = rdm.normal(key, (n, k)) + 1j * rdm.normal(key, (n, k))
        return jnp.sqrt(n) * (samples / jnp.linalg.norm(samples, axis=0))


class RademacherSampler(AbstractSampler, strict=True):
    def __call__(self, key: PRNGKeyArray, n: int, k: int) -> Inexact[Array, "n k"]:
        return rdm.normal(key, (n, k))
