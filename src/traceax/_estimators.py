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
from typing import Any, Generic, TypeVar

import equinox as eqx

from equinox import AbstractVar
from jaxtyping import Array, PRNGKeyArray, PyTree
from lineax import AbstractLinearOperator

from ._samplers import AbstractSampler


_EstimatorState = TypeVar("_EstimatorState")


class AbstractEstimator(eqx.Module, Generic[_EstimatorState], strict=True):
    r"""Abstract base class for all trace estimators."""

    sampler: AbstractVar[AbstractSampler]

    @abstractmethod
    def init(self, key: PRNGKeyArray, operator: AbstractLinearOperator) -> _EstimatorState:
        """ """
        ...

    @abstractmethod
    def estimate(self, state: _EstimatorState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        """Estimate the trace of `operator`.

        !!! Example

            ```python
            key = jax.random.PRNGKey(...)
            operator = lx.MatrixLinearOperator(...)
            hutch = tx.HutchinsonEstimator()
            result = hutch.estimate(key, operator, k=10)
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

    def __call__(self, state: _EstimatorState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        """An alias for `estimate`."""
        return self.estimate(state, k)

    @abstractmethod
    def transpose(self, state: _EstimatorState) -> _EstimatorState: ...
