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

from typing import Any

import equinox as eqx

from jaxtyping import ArrayLike, PyTree


class Solution(eqx.Module, strict=True):
    """The solution to a stochastic estimation problem.

    **Attributes:**

    - `value`: The estimated value.
    - `stats`: A dictionary containing statistics about the solution (e.g., standard error).
        This may be empty if individual estimators cannot provide this information (i.e. `{}`)
    - `state`: The internal state for the estimator.
    """

    value: PyTree[Any]
    stats: dict[str, PyTree[ArrayLike]]
    state: PyTree[Any]
