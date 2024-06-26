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

import pytest

import jax.numpy as jnp
import lineax as lx

import traceax as tr

from .helpers import (
    construct_matrix,
)


@pytest.mark.parametrize("estimator", (tr.HutchinsonEstimator(), tr.HutchPlusPlusEstimator(), tr.XTraceEstimator()))
@pytest.mark.parametrize("k", (5, 10, 50))
@pytest.mark.parametrize(
    "tags",
    (
        None,
        lx.diagonal_tag,
        lx.symmetric_tag,
        lx.lower_triangular_tag,
        lx.upper_triangular_tag,
        lx.tridiagonal_tag,
        lx.unit_diagonal_tag,
    ),
)
@pytest.mark.parametrize("size", (5, 50, 500))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_matrix_linop(getkey, estimator, k, tags, size, dtype):
    k = min(k, size)
    matrix = construct_matrix(getkey, tags, size, dtype)
    operator = lx.MatrixLinearOperator(matrix, tags=tags)
    result = estimator.estimate(getkey(), operator, k)

    assert result is not None
    assert result[0] is not None
    assert jnp.isfinite(result[0])


@pytest.mark.parametrize("estimator", (tr.HutchinsonEstimator(), tr.HutchPlusPlusEstimator(), tr.XTraceEstimator()))
@pytest.mark.parametrize("k", (5, 10, 50))
@pytest.mark.parametrize("size", (5, 50, 500))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_diag_linop(getkey, estimator, k, size, dtype):
    k = min(k, size)
    matrix = construct_matrix(getkey, lx.diagonal_tag, size, dtype)
    operator = lx.DiagonalLinearOperator(jnp.diag(matrix))
    result = estimator.estimate(getkey(), operator, k)

    assert result is not None
    assert result[0] is not None
    assert jnp.isfinite(result[0])


@pytest.mark.parametrize(
    "estimator", (tr.HutchinsonEstimator(), tr.HutchPlusPlusEstimator(), tr.XTraceEstimator(), tr.XNysTraceEstimator())
)
@pytest.mark.parametrize("k", (5, 10, 50))
@pytest.mark.parametrize("tags", (lx.positive_semidefinite_tag, lx.negative_semidefinite_tag))
@pytest.mark.parametrize("size", (5, 50, 500))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_nsd_psd_matrix_linop(getkey, estimator, k, tags, size, dtype):
    k = min(k, size)
    matrix = construct_matrix(getkey, tags, size, dtype)
    operator = lx.MatrixLinearOperator(matrix, tags=tags)
    result = estimator.estimate(getkey(), operator, k)

    assert result is not None
    assert result[0] is not None
    assert jnp.isfinite(result[0])
