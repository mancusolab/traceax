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

from typing import Any, cast

import pytest

import jax
import jax.numpy as jnp
import lineax as lx

import traceax as tx

from .helpers import (
    construct_matrix,
)


TRACE_ESTIMATORS = (
    tx.HutchinsonEstimator(),
    tx.HutchPlusPlusEstimator(),
    tx.XTraceEstimator(),
)
LINEAR_PROBE_FUNCTIONAL_ESTIMATORS = (
    tx.HutchinsonEstimator(),
    tx.XTraceEstimator(),
)
PSD_TRACE_ESTIMATORS = (*TRACE_ESTIMATORS, tx.XNysTraceEstimator())

FAST_SIZE = 25
FAST_K = 10
FAST_DTYPE = jnp.float32

TRACE_SIZES = (5, 25, 100)
TRACE_KS = (5, 10, 50)
TRACE_DTYPES = (jnp.float32, jnp.float64)

MATRIX_TAGS = (
    None,
    lx.diagonal_tag,
    lx.symmetric_tag,
    lx.lower_triangular_tag,
    lx.upper_triangular_tag,
    lx.tridiagonal_tag,
    lx.unit_diagonal_tag,
)
STRUCTURED_TAGS = (
    lx.diagonal_tag,
    lx.symmetric_tag,
    lx.lower_triangular_tag,
    lx.upper_triangular_tag,
    lx.tridiagonal_tag,
    lx.unit_diagonal_tag,
    lx.positive_semidefinite_tag,
    lx.negative_semidefinite_tag,
)
PSD_TAGS = (lx.positive_semidefinite_tag, lx.negative_semidefinite_tag)
COMPOUND_TAGS = (None, *STRUCTURED_TAGS)
LINEAX_OPERATOR_CASES = (
    "matrix",
    "matrix_transpose",
    "diagonal_plus_matrix",
    "tridiagonal_transpose",
    "identity_composed_matrix",
    "function",
    "function_transpose",
    "function_plus_matrix",
    "jacobian_fwd",
    "jacobian_bwd",
    "linearised_jacobian",
    "jacobian_plus_diagonal",
    "pytree",
    "pytree_transpose",
    "pytree_symmetric_sum",
    "nested_compound",
    "tagged_symmetric_sum",
)
PSD_OPERATOR_CASES = (
    "tagged_gram_psd",
    "tagged_negative_gram_nsd",
)
LINEAX_GRAD_OPERATOR_CASES = (
    "matrix_transpose",
    "function_plus_matrix",
    "jacobian_plus_diagonal",
    "pytree_symmetric_sum",
    "nested_compound",
    "tagged_symmetric_sum",
)


def _assert_finite_solution(result):
    assert result is not None
    assert result.value is not None
    assert jnp.isfinite(result.value)


def _matrix_operator(getkey, tags, size, dtype):
    matrix = construct_matrix(getkey, tags, size, dtype)
    return lx.MatrixLinearOperator(matrix, tags=tags)


def _trace(getkey, operator, k=FAST_K, estimator=tx.HutchinsonEstimator()):
    k = min(k, operator.in_size())
    return tx.trace(getkey(), operator, k, estimator)


def _function_operator(matrix):
    input_structure = jax.ShapeDtypeStruct((matrix.shape[1],), matrix.dtype)

    def fn(vector):
        return matrix @ vector

    return lx.FunctionLinearOperator(fn, input_structure)


def _jacobian_operator(matrix, *, jac=None):
    x = jnp.linspace(-1, 1, matrix.shape[1], dtype=matrix.dtype)

    def fn(vector, args):
        return args @ vector

    return lx.JacobianLinearOperator(fn, x, args=matrix, jac=jac)


def _pytree_operator(getkey):
    x_size = 2
    y_size = 3
    output_structure = {
        "x": jax.ShapeDtypeStruct((x_size,), FAST_DTYPE),
        "y": jax.ShapeDtypeStruct((y_size,), FAST_DTYPE),
    }
    return lx.PyTreeLinearOperator(
        {
            "x": {
                "x": jax.random.normal(getkey(), (x_size, x_size), dtype=FAST_DTYPE),
                "y": jax.random.normal(getkey(), (x_size, y_size), dtype=FAST_DTYPE),
            },
            "y": {
                "x": jax.random.normal(getkey(), (y_size, x_size), dtype=FAST_DTYPE),
                "y": jax.random.normal(getkey(), (y_size, y_size), dtype=FAST_DTYPE),
            },
        },
        output_structure,
    )


def _lineax_operator_case(getkey, case_name):
    matrix_a = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)
    matrix_b = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)
    matrix_c = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)
    op_a = lx.MatrixLinearOperator(matrix_a)
    op_b = lx.MatrixLinearOperator(matrix_b)
    op_c = lx.MatrixLinearOperator(matrix_c)
    diagonal = jnp.linspace(1, 2, FAST_SIZE, dtype=FAST_DTYPE)
    diagonal_op = lx.DiagonalLinearOperator(diagonal)
    identity_op = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((FAST_SIZE,), FAST_DTYPE))
    function_op = _function_operator(matrix_a)
    jacobian_fwd_op = _jacobian_operator(matrix_a, jac="fwd")
    jacobian_bwd_op = _jacobian_operator(matrix_a, jac="bwd")

    if case_name == "matrix":
        return op_a
    if case_name == "matrix_transpose":
        return op_a.T
    if case_name == "diagonal_plus_matrix":
        return diagonal_op + op_a
    if case_name == "tridiagonal_transpose":
        tridiagonal = construct_matrix(getkey, lx.tridiagonal_tag, FAST_SIZE, FAST_DTYPE)
        return lx.TridiagonalLinearOperator(
            jnp.diag(tridiagonal),
            jnp.diag(tridiagonal, k=-1),
            jnp.diag(tridiagonal, k=1),
        ).T
    if case_name == "identity_composed_matrix":
        return identity_op @ op_a @ identity_op
    if case_name == "function":
        return function_op
    if case_name == "function_transpose":
        return function_op.T
    if case_name == "function_plus_matrix":
        return function_op + op_b
    if case_name == "jacobian_fwd":
        return jacobian_fwd_op
    if case_name == "jacobian_bwd":
        return jacobian_bwd_op
    if case_name == "linearised_jacobian":
        return lx.linearise(jacobian_fwd_op)
    if case_name == "jacobian_plus_diagonal":
        return jacobian_fwd_op + diagonal_op
    if case_name == "pytree":
        return _pytree_operator(getkey)
    if case_name == "pytree_transpose":
        return _pytree_operator(getkey).T
    if case_name == "pytree_symmetric_sum":
        pytree_op = _pytree_operator(getkey)
        return pytree_op + pytree_op.T
    if case_name == "nested_compound":
        quarter = jnp.asarray(0.25, dtype=FAST_DTYPE)
        denom = jnp.asarray(1.5, dtype=FAST_DTYPE)
        return ((quarter * op_a.T) @ (op_b + diagonal_op) - op_c) / denom
    if case_name == "tagged_symmetric_sum":
        return lx.TaggedLinearOperator(op_a + op_a.T, lx.symmetric_tag)
    if case_name == "tagged_gram_psd":
        return lx.TaggedLinearOperator(op_a.T @ op_a, lx.positive_semidefinite_tag)
    if case_name == "tagged_negative_gram_nsd":
        return lx.TaggedLinearOperator(-(op_a.T @ op_a), lx.negative_semidefinite_tag)
    raise AssertionError(f"Unhandled Lineax operator case {case_name}")


@pytest.mark.parametrize("tags", MATRIX_TAGS)
def test_matrix_linop_tag_smoke(getkey, tags):
    operator = _matrix_operator(getkey, tags, FAST_SIZE, FAST_DTYPE)
    result = _trace(getkey, operator)

    _assert_finite_solution(result)


@pytest.mark.parametrize("tags", PSD_TAGS)
@pytest.mark.parametrize("estimator", PSD_TRACE_ESTIMATORS)
def test_psd_nsd_matrix_linop_smoke(getkey, tags, estimator):
    operator = _matrix_operator(getkey, tags, FAST_SIZE, FAST_DTYPE)
    result = _trace(getkey, operator, estimator=estimator)

    _assert_finite_solution(result)


@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
def test_tridiagonal_linop_smoke(getkey, estimator):
    matrix = construct_matrix(getkey, lx.tridiagonal_tag, FAST_SIZE, FAST_DTYPE)
    operator = lx.TridiagonalLinearOperator(jnp.diag(matrix), jnp.diag(matrix, k=-1), jnp.diag(matrix, k=1))
    result = _trace(getkey, operator, estimator=estimator)

    _assert_finite_solution(result)


@pytest.mark.parametrize("dtype", TRACE_DTYPES)
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
def test_identity_linop_exact(getkey, dtype, estimator):
    operator = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((FAST_SIZE,), dtype))
    result = _trace(getkey, operator, estimator=estimator)

    assert result.stats == {}
    assert result.value == jnp.asarray(FAST_SIZE, dtype=float)


@pytest.mark.parametrize("dtype", TRACE_DTYPES)
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
def test_diagonal_linop_exact(getkey, dtype, estimator):
    diagonal = jnp.arange(FAST_SIZE, dtype=dtype)
    operator = lx.DiagonalLinearOperator(diagonal)
    result = _trace(getkey, operator, estimator=estimator)

    assert result.stats == {}
    assert result.value == jnp.sum(diagonal)


@pytest.mark.parametrize("tags", STRUCTURED_TAGS)
def test_tagged_linop_smoke(getkey, tags):
    operator = _matrix_operator(getkey, tags, FAST_SIZE, FAST_DTYPE)
    tagged_operator = lx.TaggedLinearOperator(operator, tags=tags)
    result = _trace(getkey, tagged_operator)

    _assert_finite_solution(result)


@pytest.mark.parametrize("operator_name", ("add", "composed", "mul", "neg", "div"))
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
def test_compound_operator_smoke(getkey, operator_name, estimator):
    matrix_a = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)
    matrix_b = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)
    op_a = lx.MatrixLinearOperator(matrix_a)
    op_b = lx.MatrixLinearOperator(matrix_b)
    scalar = jnp.asarray(0.5, dtype=FAST_DTYPE)

    if operator_name == "add":
        operator = op_a + op_b
    elif operator_name == "composed":
        operator = op_a @ op_b
    elif operator_name == "mul":
        operator = scalar * op_a
    elif operator_name == "neg":
        operator = -op_a
    elif operator_name == "div":
        operator = op_b / scalar
    else:
        raise AssertionError(f"Unhandled operator case {operator_name}")

    result = _trace(getkey, operator, estimator=estimator)

    _assert_finite_solution(result)


@pytest.mark.parametrize("case_name", LINEAX_OPERATOR_CASES)
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
def test_lineax_operator_zoo_smoke(getkey, case_name, estimator):
    operator = _lineax_operator_case(getkey, case_name)
    result = _trace(getkey, operator, estimator=estimator)

    _assert_finite_solution(result)


@pytest.mark.parametrize("case_name", PSD_OPERATOR_CASES)
@pytest.mark.parametrize("estimator", PSD_TRACE_ESTIMATORS)
def test_lineax_psd_operator_zoo_smoke(getkey, case_name, estimator):
    operator = _lineax_operator_case(getkey, case_name)
    result = _trace(getkey, operator, estimator=estimator)

    _assert_finite_solution(result)


@pytest.mark.parametrize("case_name", LINEAX_GRAD_OPERATOR_CASES)
def test_lineax_operator_zoo_hutchinson_grad_modes_match(getkey, case_name):
    key = getkey()
    operator = _lineax_operator_case(getkey, case_name)
    estimator = tx.HutchinsonEstimator()
    k = min(FAST_K, operator.in_size())

    def trace_value(scalar, grad_mode):
        scaled_operator = scalar * operator
        return tx.trace(key, scaled_operator, k, estimator, grad_mode=grad_mode).value

    scalar = jnp.asarray(0.5, dtype=FAST_DTYPE)
    functional_value, functional_grad = jax.value_and_grad(trace_value)(scalar, "functional")
    estimator_value, estimator_grad = jax.value_and_grad(trace_value)(scalar, "estimator")

    assert jnp.allclose(functional_value, estimator_value)
    assert jnp.allclose(functional_grad, estimator_grad)


@pytest.mark.parametrize("estimator", PSD_TRACE_ESTIMATORS)
def test_estimator_state_api_returns_finite_estimate(getkey, estimator):
    operator = _matrix_operator(getkey, lx.positive_semidefinite_tag, FAST_SIZE, FAST_DTYPE)
    state = estimator.init(getkey(), operator)
    value, _stats = estimator.estimate(state, FAST_K)

    assert jnp.isfinite(value)


def test_trace_solution_state_can_be_reused(getkey):
    estimator = tx.HutchinsonEstimator()
    operator = _matrix_operator(getkey, None, FAST_SIZE, FAST_DTYPE)

    initial = _trace(getkey, operator, estimator=estimator)
    reused = tx.trace(getkey(), operator, FAST_K, estimator, state=initial.state)

    _assert_finite_solution(reused)


def test_trace_reused_state_uses_current_operator(getkey):
    key = getkey()
    estimator = tx.HutchinsonEstimator()
    initial_operator = lx.MatrixLinearOperator(jnp.eye(FAST_SIZE, dtype=FAST_DTYPE))
    current_operator = lx.MatrixLinearOperator(2 * jnp.eye(FAST_SIZE, dtype=FAST_DTYPE))

    initial = tx.trace(key, initial_operator, FAST_K, estimator)
    reused = tx.trace(getkey(), current_operator, FAST_K, estimator, state=initial.state)

    assert reused.value == jnp.asarray(2 * FAST_SIZE, dtype=FAST_DTYPE)


@pytest.mark.parametrize("estimator", LINEAR_PROBE_FUNCTIONAL_ESTIMATORS)
def test_trace_grad_matrix_operator_matches_linear_probe_pullback(getkey, estimator):
    key = getkey()
    matrix = jnp.arange(FAST_SIZE * FAST_SIZE, dtype=FAST_DTYPE).reshape(FAST_SIZE, FAST_SIZE)
    matrix = matrix / jnp.asarray(FAST_SIZE, dtype=FAST_DTYPE)

    def trace_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        return tx.trace(key, operator, FAST_K, estimator).value

    def expected_probe_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        samples = tx.RademacherSampler()(key, FAST_SIZE, FAST_K)
        projected = jax.vmap(operator.mv, (1,), 1)(samples)
        return jnp.sum(projected * samples) / FAST_K

    actual = jax.grad(trace_value)(matrix)
    expected = jax.grad(expected_probe_value)(matrix)

    assert actual.shape == matrix.shape
    assert jnp.allclose(actual, expected)


@pytest.mark.parametrize("estimator", LINEAR_PROBE_FUNCTIONAL_ESTIMATORS)
def test_trace_grad_scalar_scaled_operator_matches_linear_probe_pullback(getkey, estimator):
    key = getkey()
    matrix = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)
    operator = lx.MatrixLinearOperator(matrix)

    def trace_value(scalar):
        return tx.trace(key, scalar * operator, FAST_K, estimator).value

    def expected_probe_value(scalar):
        scaled_operator = scalar * operator
        samples = tx.RademacherSampler()(key, FAST_SIZE, FAST_K)
        projected = jax.vmap(scaled_operator.mv, (1,), 1)(samples)
        return jnp.sum(projected * samples) / FAST_K

    actual = jax.grad(trace_value)(jnp.asarray(0.5, dtype=FAST_DTYPE))
    expected = jax.grad(expected_probe_value)(jnp.asarray(0.5, dtype=FAST_DTYPE))

    assert jnp.allclose(actual, expected)


def test_trace_functional_jvp_matches_linear_probe_jvp(getkey):
    key = getkey()
    matrix = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)
    direction = jnp.eye(FAST_SIZE, dtype=FAST_DTYPE)

    def trace_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        return tx.trace(key, operator, FAST_K, tx.XTraceEstimator()).value

    def expected_probe_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        samples = tx.RademacherSampler()(key, FAST_SIZE, FAST_K)
        projected = jax.vmap(operator.mv, (1,), 1)(samples)
        return jnp.sum(projected * samples) / FAST_K

    _value, actual_tangent = jax.jvp(trace_value, (matrix,), (direction,))
    _expected_value, expected_tangent = jax.jvp(expected_probe_value, (matrix,), (direction,))

    assert jnp.allclose(actual_tangent, expected_tangent)


def test_trace_grad_estimator_mode_matches_estimator_program(getkey):
    key = getkey()
    estimator = tx.XTraceEstimator()
    matrix = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)

    def trace_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        return tx.trace(key, operator, FAST_K, estimator, grad_mode="estimator").value

    def expected_estimator_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        state = estimator.init(key, operator)
        value, _stats = estimator.estimate(state, FAST_K)
        return value

    actual = jax.grad(trace_value)(matrix)
    expected = jax.grad(expected_estimator_value)(matrix)

    assert actual.shape == matrix.shape
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_hutchplusplus_functional_grad_matches_frozen_sketch_pullback(getkey):
    key = getkey()
    matrix = construct_matrix(getkey, None, FAST_SIZE, FAST_DTYPE)
    estimator = tx.HutchPlusPlusEstimator()
    k = FAST_K

    def trace_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        return tx.trace(key, operator, k, estimator, grad_mode="functional").value

    def frozen_sketch_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        samples = tx.RademacherSampler()(key, FAST_SIZE, 2 * (k // 3))
        X1 = samples[:, : k // 3]
        X2 = samples[:, k // 3 :]

        Y = jax.vmap(operator.mv, in_axes=1, out_axes=1)(X1)
        Q, _ = jnp.linalg.qr(Y)
        Q = jax.lax.stop_gradient(Q)
        G = X2 - Q @ (Q.T @ X2)
        G = jax.lax.stop_gradient(G)

        AQ = jax.vmap(operator.mv, in_axes=1, out_axes=1)(Q)
        AG = jax.vmap(operator.mv, in_axes=1, out_axes=1)(G)
        return jnp.sum(AQ * Q) + jnp.sum(AG * G) / G.shape[1]

    def hutchinson_value(matrix):
        operator = lx.MatrixLinearOperator(matrix)
        samples = tx.RademacherSampler()(key, FAST_SIZE, k)
        projected = jax.vmap(operator.mv, in_axes=1, out_axes=1)(samples)
        return jnp.sum(projected * samples) / k

    actual = jax.grad(trace_value)(matrix)
    expected = jax.grad(frozen_sketch_value)(matrix)
    hutchinson_expected = jax.grad(hutchinson_value)(matrix)

    assert actual.shape == matrix.shape
    assert jnp.allclose(actual, expected, rtol=1e-5, atol=1e-5)
    assert not jnp.allclose(expected, hutchinson_expected, rtol=1e-5, atol=1e-5)


def test_trace_rejects_unknown_grad_mode(getkey):
    operator = _matrix_operator(getkey, None, FAST_SIZE, FAST_DTYPE)

    with pytest.raises(ValueError, match="grad_mode"):
        tx.trace(getkey(), operator, FAST_K, grad_mode=cast(Any, "unknown"))


@pytest.mark.parametrize(
    ("estimator", "k"),
    (
        (tx.HutchPlusPlusEstimator(), 2),
        (tx.XTraceEstimator(), 1),
    ),
)
def test_trace_rejects_estimator_k_too_small(getkey, estimator, k):
    operator = _matrix_operator(getkey, None, FAST_SIZE, FAST_DTYPE)

    with pytest.raises(ValueError, match="requires.*k"):
        tx.trace(getkey(), operator, k, estimator)


def test_trace_rejects_non_static_k(getkey):
    operator = _matrix_operator(getkey, None, FAST_SIZE, FAST_DTYPE)

    with pytest.raises(TypeError, match="Python int"):
        tx.trace(getkey(), operator, cast(int, jnp.asarray(FAST_K)), tx.HutchinsonEstimator())


def test_trace_rejects_complex_operator(getkey):
    matrix = jnp.eye(FAST_SIZE, dtype=jnp.complex64)
    operator = lx.MatrixLinearOperator(matrix)

    with pytest.raises(ValueError, match="complex"):
        tx.trace(getkey(), operator, FAST_K, tx.HutchinsonEstimator())


def test_pytree_linear_operator_trace_and_grad(getkey):
    key = getkey()
    diagonal = {
        "x": jnp.arange(1, 4, dtype=FAST_DTYPE),
        "y": jnp.arange(4, 6, dtype=FAST_DTYPE),
    }

    def trace_value(scale):
        operator = lx.DiagonalLinearOperator(jax.tree.map(lambda leaf: scale * leaf, diagonal))
        return tx.trace(key, operator, FAST_K, tx.HutchinsonEstimator()).value

    scale = jnp.asarray(2.0, dtype=FAST_DTYPE)
    actual_value, actual_grad = jax.value_and_grad(trace_value)(scale)
    expected_grad = sum(jnp.sum(leaf) for leaf in jax.tree.leaves(diagonal))

    assert jnp.allclose(actual_value, scale * expected_grad)
    assert jnp.allclose(actual_grad, expected_grad)


@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
def test_function_linear_operator_trace_and_grad(getkey, estimator):
    key = getkey()
    input_structure = {
        "x": jax.ShapeDtypeStruct((3,), FAST_DTYPE),
        "y": jax.ShapeDtypeStruct((2,), FAST_DTYPE),
    }

    def trace_value(scale):
        def fn(vector):
            return {
                "x": scale * jnp.array([1.0, 2.0, 3.0], dtype=FAST_DTYPE) * vector["x"],
                "y": scale * jnp.array([4.0, 5.0], dtype=FAST_DTYPE) * vector["y"],
            }

        operator = lx.FunctionLinearOperator(fn, input_structure)
        return tx.trace(key, operator, FAST_K, estimator).value

    scale = jnp.asarray(2.0, dtype=FAST_DTYPE)
    actual_value, actual_grad = jax.value_and_grad(trace_value)(scale)
    expected_grad = jnp.asarray(15.0, dtype=FAST_DTYPE)

    assert jnp.isfinite(actual_value)
    assert jnp.isfinite(actual_grad)
    if not isinstance(estimator, tx.HutchPlusPlusEstimator):
        assert jnp.allclose(actual_grad, expected_grad)


@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
def test_jacobian_linear_operator_trace_and_grad(getkey, estimator):
    key = getkey()
    x = {
        "x": jnp.arange(1, 4, dtype=FAST_DTYPE),
        "y": jnp.arange(4, 6, dtype=FAST_DTYPE),
    }

    def trace_value(scale):
        def fn(vector, args):
            del args
            return {
                "x": scale * jnp.array([1.0, 2.0, 3.0], dtype=FAST_DTYPE) * vector["x"],
                "y": scale * jnp.array([4.0, 5.0], dtype=FAST_DTYPE) * vector["y"],
            }

        operator = lx.JacobianLinearOperator(fn, x)
        return tx.trace(key, operator, FAST_K, estimator).value

    scale = jnp.asarray(2.0, dtype=FAST_DTYPE)
    actual_value, actual_grad = jax.value_and_grad(trace_value)(scale)
    expected_grad = jnp.asarray(15.0, dtype=FAST_DTYPE)

    assert jnp.isfinite(actual_value)
    assert jnp.isfinite(actual_grad)
    if not isinstance(estimator, tx.HutchPlusPlusEstimator):
        assert jnp.allclose(actual_grad, expected_grad)


def test_hutchinson_grad_modes_match_for_function_linear_operator(getkey):
    key = getkey()
    estimator = tx.HutchinsonEstimator()
    input_structure = {
        "x": jax.ShapeDtypeStruct((3,), FAST_DTYPE),
        "y": jax.ShapeDtypeStruct((2,), FAST_DTYPE),
    }

    def trace_value(scale, grad_mode):
        def fn(vector):
            return {
                "x": scale * jnp.array([1.0, 2.0, 3.0], dtype=FAST_DTYPE) * vector["x"],
                "y": scale * jnp.array([4.0, 5.0], dtype=FAST_DTYPE) * vector["y"],
            }

        operator = lx.FunctionLinearOperator(fn, input_structure)
        return tx.trace(key, operator, FAST_K, estimator, grad_mode=grad_mode).value

    scale = jnp.asarray(2.0, dtype=FAST_DTYPE)
    functional_value, functional_grad = jax.value_and_grad(trace_value)(scale, "functional")
    estimator_value, estimator_grad = jax.value_and_grad(trace_value)(scale, "estimator")

    assert jnp.allclose(functional_value, estimator_value)
    assert jnp.allclose(functional_grad, estimator_grad)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
@pytest.mark.parametrize("k", TRACE_KS)
@pytest.mark.parametrize("tags", MATRIX_TAGS)
@pytest.mark.parametrize("size", TRACE_SIZES)
@pytest.mark.parametrize("dtype", TRACE_DTYPES)
def test_slow_matrix_linop(getkey, estimator, k, tags, size, dtype):
    operator = _matrix_operator(getkey, tags, size, dtype)
    result = _trace(getkey, operator, k, estimator)

    _assert_finite_solution(result)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
@pytest.mark.parametrize("k", TRACE_KS)
@pytest.mark.parametrize("size", TRACE_SIZES)
@pytest.mark.parametrize("dtype", TRACE_DTYPES)
def test_slow_diag_linop(getkey, estimator, k, size, dtype):
    matrix = construct_matrix(getkey, lx.diagonal_tag, size, dtype)
    operator = lx.DiagonalLinearOperator(jnp.diag(matrix))
    result = _trace(getkey, operator, k, estimator)

    _assert_finite_solution(result)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", PSD_TRACE_ESTIMATORS)
@pytest.mark.parametrize("k", TRACE_KS)
@pytest.mark.parametrize("tags", PSD_TAGS)
@pytest.mark.parametrize("size", TRACE_SIZES)
@pytest.mark.parametrize("dtype", TRACE_DTYPES)
def test_slow_nsd_psd_matrix_linop(getkey, estimator, k, tags, size, dtype):
    operator = _matrix_operator(getkey, tags, size, dtype)
    result = _trace(getkey, operator, k, estimator)

    _assert_finite_solution(result)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
@pytest.mark.parametrize("k", TRACE_KS)
@pytest.mark.parametrize("size", TRACE_SIZES)
@pytest.mark.parametrize("dtype", TRACE_DTYPES)
def test_slow_tridiagonal_linop(getkey, estimator, k, size, dtype):
    matrix = construct_matrix(getkey, lx.tridiagonal_tag, size, dtype)
    operator = lx.TridiagonalLinearOperator(jnp.diag(matrix), jnp.diag(matrix, k=-1), jnp.diag(matrix, k=1))
    result = _trace(getkey, operator, k, estimator)

    _assert_finite_solution(result)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
@pytest.mark.parametrize("k", TRACE_KS)
@pytest.mark.parametrize("size", TRACE_SIZES)
@pytest.mark.parametrize("dtype", TRACE_DTYPES)
def test_slow_identity_linop(getkey, estimator, k, size, dtype):
    operator = lx.IdentityLinearOperator(jax.ShapeDtypeStruct((size,), dtype))
    result = _trace(getkey, operator, k, estimator)

    _assert_finite_solution(result)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
@pytest.mark.parametrize("k", TRACE_KS)
@pytest.mark.parametrize("tags", STRUCTURED_TAGS)
@pytest.mark.parametrize("size", TRACE_SIZES)
@pytest.mark.parametrize("dtype", TRACE_DTYPES)
def test_slow_tagged_linop(getkey, estimator, k, tags, size, dtype):
    operator = _matrix_operator(getkey, tags, size, dtype)
    tagged_operator = lx.TaggedLinearOperator(operator, tags=tags)
    result = _trace(getkey, tagged_operator, k, estimator)

    _assert_finite_solution(result)

    sym_operator = operator + operator.T
    sym_operator = lx.TaggedLinearOperator(sym_operator, lx.symmetric_tag)
    sym_result = _trace(getkey, sym_operator, k, estimator)

    assert lx.is_symmetric(sym_operator)
    _assert_finite_solution(sym_result)


@pytest.mark.slow
@pytest.mark.parametrize("estimator", TRACE_ESTIMATORS)
@pytest.mark.parametrize("tags", COMPOUND_TAGS)
@pytest.mark.parametrize("k", TRACE_KS)
@pytest.mark.parametrize("size", TRACE_SIZES)
@pytest.mark.parametrize("dtype", TRACE_DTYPES)
def test_slow_compound_op(getkey, estimator, tags, k, size, dtype):
    matrix_a = construct_matrix(getkey, tags, size, dtype)
    matrix_b = construct_matrix(getkey, tags, size, dtype)
    op_a = lx.MatrixLinearOperator(matrix_a, tags=tags)
    op_b = lx.MatrixLinearOperator(matrix_b, tags=tags)
    scalar = jnp.asarray(0.5, dtype=dtype)

    operators = (
        op_a + op_b,
        op_a @ op_b,
        scalar * op_a,
        -op_a,
        op_b / scalar,
    )

    for operator in operators:
        result = _trace(getkey, operator, k, estimator)
        _assert_finite_solution(result)
