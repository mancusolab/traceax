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

import equinox.internal as eqxi
import jax
import jax.core

from jax import vmap
from jax.interpreters import ad as ad
from lineax import AbstractLinearOperator


sentinel: Any = eqxi.doc_repr(object(), "sentinel")


def _check_operator(operator: AbstractLinearOperator) -> int:
    n_in = operator.in_size()
    n_out = operator.out_size()
    if n_in != n_out:
        raise ValueError(f"Estimation requires square linear operator. Found {(n_out, n_in)}.")

    return n_in


def _clip_k(k: int, n: int) -> int:
    return min(max(k, 1), n)


def _vmap_mv(operator: AbstractLinearOperator):
    return vmap(operator.mv, (1,), 1)


def _is_none(x):
    return x is None


def _to_shapedarray(x):
    if isinstance(x, jax.ShapeDtypeStruct):
        return jax.core.ShapedArray(x.shape, x.dtype)
    else:
        return x


def _to_struct(x, name):
    if isinstance(x, jax.core.ShapedArray):
        return jax.ShapeDtypeStruct(x.shape, x.dtype)
    elif isinstance(x, jax.core.AbstractValue):
        raise NotImplementedError(
            f"`{name}` only supports working with JAX arrays; not " f"other abstract values. Got abstract value {x}."
        )
    else:
        return x


def _assert_false(x):
    assert False


def _is_undefined(x):
    return isinstance(x, ad.UndefinedPrimal)


def _assert_defined(x):
    assert not _is_undefined(x)


def _keep_undefined(v, ct):
    if _is_undefined(v):
        return ct
    else:
        return None
