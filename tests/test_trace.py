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

import jax
import jax.numpy as jnp
import lineax as lx

import traceax as tx

from .helpers import (
    construct_matrix,
)


@pytest.mark.parametrize(
    "estimator",
    (tx.HutchinsonEstimator(), tx.HutchPlusPlusEstimator(), tx.XTraceEstimator()),
)
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
    result = tx.trace(getkey(), operator, k, estimator)

    assert result is not None
    assert result.value is not None
    assert jnp.isfinite(result.value)


@pytest.mark.parametrize(
    "estimator",
    (tx.HutchinsonEstimator(), tx.HutchPlusPlusEstimator(), tx.XTraceEstimator()),
)
@pytest.mark.parametrize("k", (5, 10, 50))
@pytest.mark.parametrize("size", (5, 50, 500))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_diag_linop(getkey, estimator, k, size, dtype):
    k = min(k, size)
    matrix = construct_matrix(getkey, lx.diagonal_tag, size, dtype)
    operator = lx.DiagonalLinearOperator(jnp.diag(matrix))
    result = tx.trace(getkey(), operator, k, estimator)

    assert result is not None
    assert result.value is not None
    assert jnp.isfinite(result.value)


@pytest.mark.parametrize(
    "estimator",
    (
        tx.HutchinsonEstimator(),
        tx.HutchPlusPlusEstimator(),
        tx.XTraceEstimator(),
        tx.XNysTraceEstimator(),
    ),
)
@pytest.mark.parametrize("k", (5, 10, 50))
@pytest.mark.parametrize("tags", (lx.positive_semidefinite_tag, lx.negative_semidefinite_tag))
@pytest.mark.parametrize("size", (5, 50, 500))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_nsd_psd_matrix_linop(getkey, estimator, k, tags, size, dtype):
    k = min(k, size)
    matrix = construct_matrix(getkey, tags, size, dtype)
    operator = lx.MatrixLinearOperator(matrix, tags=tags)
    result = tx.trace(getkey(), operator, k, estimator)

    assert result is not None
    assert result.value is not None
    assert jnp.isfinite(result.value)


@pytest.mark.parametrize(
    "estimator",
    (tx.HutchinsonEstimator(), tx.HutchPlusPlusEstimator(), tx.XTraceEstimator()),
)
@pytest.mark.parametrize("k", (5, 10, 50))
@pytest.mark.parametrize("size", (5, 50, 500))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_tridiagonal_linop(getkey, estimator, k, size, dtype):
    k = min(k, size)
    matrix = construct_matrix(getkey, lx.tridiagonal_tag, size, dtype)
    main_diag = jnp.diag(matrix)
    lower_diag = jnp.diag(matrix, k=-1)
    upper_diag = jnp.diag(matrix, k=1)
    operator = lx.TridiagonalLinearOperator(main_diag, lower_diag, upper_diag)
    result = tx.trace(getkey(), operator, k, estimator)
    
    assert result is not None
    assert result.value is not None
    assert jnp.isfinite(result.value)


@pytest.mark.parametrize(
    "estimator",
    (tx.HutchinsonEstimator(), tx.HutchPlusPlusEstimator(), tx.XTraceEstimator()),
)
@pytest.mark.parametrize("k", (5, 10, 50))
@pytest.mark.parametrize("size", (5, 50, 500))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_identity_linop(getkey, estimator, k, size, dtype):
    k = min(k, size)
    input_structure = jax.ShapeDtypeStruct((size,), dtype)
    operator = lx.IdentityLinearOperator(input_structure)
    result = tx.trace(getkey(), operator, k, estimator)
    
    assert result is not None
    assert result.value is not None
    assert jnp.isfinite(result.value)


@pytest.mark.parametrize(
    "estimator",
    (tx.HutchinsonEstimator(), tx.HutchPlusPlusEstimator(), tx.XTraceEstimator()),
)
@pytest.mark.parametrize("k", (5, 10, 50))
@pytest.mark.parametrize(
    "tags",
    (
        lx.diagonal_tag,
        lx.symmetric_tag,
        lx.lower_triangular_tag,
        lx.upper_triangular_tag,
        lx.tridiagonal_tag,
        lx.unit_diagonal_tag,
        lx.positive_semidefinite_tag,
        lx.negative_semidefinite_tag,
    ),
)
@pytest.mark.parametrize("size", (5, 50, 500))
@pytest.mark.parametrize("dtype", (jnp.float32, jnp.float64))
def test_tagged_linear_operator(getkey, estimator, k, tags, size, dtype):
    k = min(k, size)
    matrix = construct_matrix(getkey, tags, size, dtype)
    operator = lx.MatrixLinearOperator(matrix, tags=tags)
    tagged_operator = lx.TaggedLinearOperator(operator, tags=tags)
    result = tx.trace(getkey(), tagged_operator, k, estimator)

    assert result is not None
    assert result.value is not None
    assert jnp.isfinite(result.value)