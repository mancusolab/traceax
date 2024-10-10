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
import functools as ft

from typing import Any
from typing_extensions import TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import lineax as lx

from jax.interpreters import ad as ad
from jax.numpy.linalg import norm
from jaxtyping import Array, PRNGKeyArray, PyTree

from ._estimators import AbstractEstimator
from ._samplers import AbstractSampler, RademacherSampler, SphereSampler
from ._solution import Solution
from ._utils import (
    _check_operator,
    _clip_k,
    _vmap_mv,
    sentinel,
)


def _get_scale(W: Array, D: Array, n: int, k: int) -> Array:
    return (n - k + 1) / (n - norm(W, axis=0) ** 2 + jnp.abs(D) ** 2)


_BasicTraceState: TypeAlias = tuple[PRNGKeyArray, lx.AbstractLinearOperator, int]
_PSDTraceState: TypeAlias = tuple[PRNGKeyArray, lx.AbstractLinearOperator, int, bool]


class HutchinsonEstimator(AbstractEstimator[_BasicTraceState], strict=True):
    r"""Girard-Hutchinson Trace Estimator:

    $\mathbb{E}[\omega^T \mathbf{A} \omega] = \text{trace}(\mathbf{A})$,
    where $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = RademacherSampler()

    def init(self, key: PRNGKeyArray, operator: lx.AbstractLinearOperator) -> _BasicTraceState:
        n = _check_operator(operator)
        return (key, operator, n)

    def estimate(self, state: _BasicTraceState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        key, operator, n = state

        k = _clip_k(k, n)

        # sample from proposed distribution
        samples = self.sampler(key, n, k)

        # project to k-dim space
        projected = _vmap_mv(operator)(samples)

        # take the mean across estimates
        trace_est = jnp.sum(projected * samples) / k

        return trace_est, {}

    def transpose(self, state: _BasicTraceState) -> _BasicTraceState:
        key, operator, n = state
        return key, operator.transpose(), n


HutchinsonEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: The sampling distribution for $\omega$. Default is [`traceax.RademacherSampler`][].
"""


class HutchPlusPlusEstimator(AbstractEstimator[_BasicTraceState], strict=True):
    r"""Hutch++ Trace Estimator:

    Let $\hat{\mathbf{A}} := \mathbf{Q}\mathbf{Q}^* \mathbf{A}$ be the a _low-rank approximation_
    to $\mathbf{A}$, where $\mathbf{Q}$ is the orthonormal basis of $\mathbf{A} \Omega$, for
    $\Omega = [\omega_1, \dotsc, \omega_k]$.

    Hutch++ improves upon Girard-Hutchinson estimator by including the trace of the residuals. Namely,
    Hutch++ estimates $\text{trace}(\mathbf{A})$ as
    $\text{trace}(\hat{\mathbf{A}}) - \text{trace}(\mathbf{A} - \hat{\mathbf{A}})$.

    As with the Girard-Hutchinson estimator, it requires
    $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = RademacherSampler()

    def init(self, key: PRNGKeyArray, operator: lx.AbstractLinearOperator) -> _BasicTraceState:
        n = _check_operator(operator)
        return (key, operator, n)

    def estimate(self, state: _BasicTraceState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        key, operator, n = state

        k = _clip_k(k, n)
        m = k // 3

        # some operators work fine with matrices in mv, some dont; this ensures they all do
        mv = _vmap_mv(operator)

        # split X into 2 Xs; X1 and X2, where X1 has shape 2m, where m = k/3
        samples = self.sampler(key, n, 2 * m)
        X1 = samples[:, :m]
        X2 = samples[:, m:]

        Y = mv(X1)

        # compute Q, _ = QR(Y) (orthogonal matrix)
        Q, _ = jnp.linalg.qr(Y)

        # compute G = X2 - Q @ (Q.T @ X2)
        G = X2 - Q @ (Q.T @ X2)

        # estimate trace = tr(Q.T @ A @ Q) + tr(G.T @ A @ G) / k
        AQ = mv(Q)
        AG = mv(G)
        trace_est = jnp.sum(AQ * Q) + jnp.sum(AG * G) / (G.shape[1])

        return trace_est, {}

    def transpose(self, state: _BasicTraceState) -> _BasicTraceState:
        key, operator, n = state
        return key, operator.transpose(), n


HutchPlusPlusEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: The sampling distribution for $\omega$. Default is [`traceax.RademacherSampler`][].
"""


class XTraceEstimator(AbstractEstimator[_BasicTraceState], strict=True):
    r"""XTrace Trace Estimator:

    Let $\hat{\mathbf{A}} := \mathbf{Q}\mathbf{Q}^* \mathbf{A}$ be the the _low-rank approximation_
    to $\mathbf{A}$, where $\mathbf{Q}$ is the orthonormal basis of $\mathbf{A} \Omega$, for
    $\Omega = [\omega_1, \dotsc, \omega_k]$.

    XTrace improves upon Hutch++ estimator by enforcing *exchangeability* of sampled test-vectors,
    to construct a symmetric estimation function with lower variance.

    Additionally, the *improved* XTrace algorithm (i.e. `improved = True`), ensures that test-vectors
    are orthogonalized against the low rank approximation $\mathbf{Q}\mathbf{Q}^* \mathbf{A}$ and
    renormalized. This improved XTrace approach may provide better empirical results compared with
    the non-orthogonalized version.

    As with the Girard-Hutchinson estimator, it requires
    $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = SphereSampler()
    improved: bool = True

    def init(self, key: PRNGKeyArray, operator: lx.AbstractLinearOperator) -> _BasicTraceState:
        n = _check_operator(operator)
        return (key, operator, n)

    def estimate(self, state: _BasicTraceState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        key, operator, n = state

        k = _clip_k(k, n)
        m = k // 2

        # some operators work fine with matrices in mv, some dont; this ensures they all do
        mv = _vmap_mv(operator)

        samples = self.sampler(key, n, m)
        Y = mv(samples)
        Q, R = jnp.linalg.qr(Y)

        # solve and rescale
        S = jnp.linalg.inv(R).T
        s = norm(S, axis=0)
        S = S / s

        # working variables
        Z = mv(Q)
        H = Q.T @ Z
        W = Q.T @ samples
        T = Z.T @ samples
        HW = H @ W

        SW_d = jnp.sum(S * W, axis=0)
        TW_d = jnp.sum(T * W, axis=0)
        SHS_d = jnp.sum(S * (H @ S), axis=0)
        WHW_d = jnp.sum(W * HW, axis=0)

        term1 = SW_d * jnp.sum((T - H.T @ W) * S, axis=0)
        term2 = (jnp.abs(SW_d) ** 2) * SHS_d
        term3 = jnp.conjugate(SW_d) * jnp.sum(S * (R - HW), axis=0)

        if self.improved:
            scale = _get_scale(W, SW_d, n, k)
        else:
            scale = 1

        estimates = jnp.trace(H) * jnp.ones(m) - SHS_d + (WHW_d - TW_d + term1 + term2 + term3) * scale
        trace_est = jnp.mean(estimates)
        std_err = jnp.std(estimates) / jnp.sqrt(m)

        return trace_est, {"std.err": std_err}

    def transpose(self, state: _BasicTraceState) -> _BasicTraceState:
        key, operator, n = state
        return key, operator.transpose(), n


XTraceEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: the sampling distribution for $\omega$. Default is [`traceax.SphereSampler`][].
- `improved`: whether to use the *improved* XTrace estimator, which rescales predicted samples.
    Default is `True` (see Notes).
"""


class XNysTraceEstimator(AbstractEstimator[_PSDTraceState], strict=True):
    r"""XNysTrace Trace Estimator:

    XNysTrace improves upon XTrace estimator when $\mathbf{A}$ is (negative-) positive-semidefinite, by
    performing a [Nyström approximation](https://en.wikipedia.org/wiki/Low-rank_matrix_approximations#Nystr%C3%B6m_approximation),
    rather than a randomized SVD (i.e., random projection followed by QR decomposition).

    Like, [`traceax.XTraceEstimator`][], the *improved* XNysTrace algorithm (i.e. `improved = True`), ensures
    that test-vectors are orthogonalized against the low rank approximation and renormalized.
    This improved XNysTrace approach may provide better empirical results compared with the non-orthogonalized version.

    As with the Girard-Hutchinson estimator, it requires
    $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = SphereSampler()
    improved: bool = True

    def init(self, key: PRNGKeyArray, operator: lx.AbstractLinearOperator) -> _PSDTraceState:
        n = _check_operator(operator)
        is_nsd = lx.is_negative_semidefinite(operator)
        if not (lx.is_positive_semidefinite(operator) | is_nsd):
            raise ValueError("`XNysTraceEstimator` may only be used for positive or negative definite linear operators")
        if is_nsd:
            operator = -operator

        return (key, operator, n, is_nsd)

    def estimate(self, state: _PSDTraceState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        key, operator, n, is_nsd = state

        k = _clip_k(k, n)

        # some operators work fine with matrices in mv, some dont; this ensures they all do
        mv = _vmap_mv(operator)

        samples = self.sampler(key, n, k)
        Y = mv(samples)

        # shift for numerical issues
        nu = jnp.finfo(Y.dtype).eps * norm(Y, "fro") / jnp.sqrt(n)
        Y = Y + samples * nu
        Q, R = jnp.linalg.qr(Y)

        # compute and symmetrize H, then take cholesky factor
        H = samples.T @ Y
        C = jnp.linalg.cholesky(0.5 * (H + H.T)).T
        B = jsp.linalg.solve_triangular(C.T, R.T, lower=True).T

        # if improved == True
        Qs, Rs = jnp.linalg.qr(samples)
        Ws = Qs.T @ samples

        # solve and rescale
        if self.improved:
            S = jnp.linalg.inv(Rs).T
            s = norm(S, axis=0)
            S = S / s
            scale = _get_scale(Ws, jnp.sum(S * Ws, axis=0), n, k)
        else:
            scale = 1

        W = Q.T @ samples
        S = jsp.linalg.solve_triangular(C, B.T).T / jnp.sqrt(jnp.diag(jnp.linalg.inv(H)))
        dSW = jnp.sum(S * W, axis=0)

        estimates = norm(B, "fro") ** 2 - norm(S, axis=0) ** 2 + (jnp.abs(dSW) ** 2) * scale - nu * n
        trace_est = jnp.mean(estimates)
        std_err = jnp.std(estimates) / jnp.sqrt(k)
        trace_est = jnp.where(is_nsd, -trace_est, trace_est)

        return trace_est, {"std.err": std_err}

    def transpose(self, state: _PSDTraceState) -> _PSDTraceState:
        key, operator, n, is_nsd = state
        return key, operator.transpose(), n, is_nsd


XNysTraceEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: the sampling distribution for $\omega$. Default is [`traceax.SphereSampler`][].
- `improved`: whether to use the *improved* XNysTrace estimator, which rescales predicted samples.
    Default is `True` (see Notes).
"""


def _is_undefined(x):
    return isinstance(x, ad.UndefinedPrimal)


def _assert_defined(x):
    assert not _is_undefined(x)


def _remove_undefined_primal(x):
    if _is_undefined(x):
        return x.aval
    else:
        return


@ft.singledispatch
def _make_identity(op: lx.AbstractLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    raise ValueError("Unsupported type!")


@_make_identity.register
def _(op: lx.MatrixLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    operator_struct = jtu.tree_map(_remove_undefined_primal, op, is_leaf=_is_undefined)
    in_size = eqx.filter_eval_shape(lambda o: o.in_size(), operator_struct)
    diag = jnp.full(in_size, ct_result)
    return lx.MatrixLinearOperator(jnp.diag(diag), tags=operator_struct.tags)


@_make_identity.register
def _(op: lx.DiagonalLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    operator_struct = jtu.tree_map(_remove_undefined_primal, op, is_leaf=_is_undefined)
    in_size = eqx.filter_eval_shape(lambda o: o.in_size(), operator_struct)
    diag = jnp.full(in_size, ct_result)
    return lx.DiagonalLinearOperator(diag)


@_make_identity.register
def _(op: lx.MulLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    inner_op = _make_identity(op.operator, ct_result)
    scalar = jnp.array(1.0)
    return lx.MulLinearOperator(inner_op, scalar)


@_make_identity.register
def _(op: lx.TridiagonalLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    operator_struct = jtu.tree_map(_remove_undefined_primal, op, is_leaf=_is_undefined)
    in_size = eqx.filter_eval_shape(lambda o: o.in_size(), operator_struct)
    diag = jnp.full(in_size, ct_result)
    off_diag = jnp.zeros(in_size - 1)
    return lx.TridiagonalLinearOperator(diag, off_diag, off_diag)


@_make_identity.register
def _(op: lx.AddLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    inner_op1 = _make_identity(op.operator1, ct_result)
    inner_op2 = _make_identity(op.operator2, ct_result)
    return lx.AddLinearOperator(inner_op1, inner_op2)


@_make_identity.register
def _(op: lx.NegLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    inner_op = _make_identity(op.operator, ct_result)
    return lx.NegLinearOperator(inner_op)


@_make_identity.register
def _(op: lx.DivLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    inner_op = _make_identity(op.operator, ct_result)
    scalar = op.scalar
    return lx.DivLinearOperator(inner_op, scalar)


@_make_identity.register
def _(op: lx.ComposedLinearOperator, ct_result: float) -> lx.AbstractLinearOperator:
    inner_op1 = _make_identity(op.operator1, ct_result)
    inner_op2 = _make_identity(op.operator2, ct_result)
    return lx.ComposedLinearOperator(inner_op1, inner_op2)


@eqxi.filter_primitive_transpose(materialise_zeros=True)  # pyright: ignore
def _estimate_trace_transpose(inputs, cts_out):
    # the jacobian, for the trace is just the identity matrix, i.e. J = I
    # so J'v = I v = v

    # primal inputs; operator should have UndefinedPrimal leaves
    key, operator, state, _, estimator = inputs

    # co-tangent of the trace approximation and the stats (None)
    cts_result, _ = cts_out

    # the internals of the operator are UndefinedPrimal leaves so
    # we need to rely on abstract values to pull structure info
    op_t = _make_identity(operator, cts_result)

    key_none = jtu.tree_map(lambda _: None, key)
    state_none = (None, op_t, None)
    k_none = None
    estimator_none = jtu.tree_map(lambda _: None, estimator)

    return key_none, op_t, state_none, k_none, estimator_none


@eqx.filter_jit
def trace(
    operator: lx.AbstractLinearOperator,
    k: int,
    key: PRNGKeyArray,
    estimator: AbstractEstimator = XTraceEstimator(),
    *,
    state: PyTree[Any] = sentinel,
) -> Solution:
    r""" """
    if eqx.is_array(operator):
        raise ValueError(
            "`traceax.trace(..., operator=...)` should be an "
            "`lineax.AbstractLinearOperator`, not a raw JAX array. If you are trying to pass "
            "a matrix then this should be passed as "
            "`lineax.MatrixLinearOperator(matrix)`."
        )

    in_size = operator.in_size()
    out_size = operator.out_size()
    if in_size != out_size:
        raise ValueError(
            "`traceax.trace(..., operator=...)` should be a square `lineax.AbstractLinearOperator`. "
            f"Found shape {out_size}x{in_size}."
        )

    # if identity op, then just shortcircuit and return dimension size
    if isinstance(operator, lx.IdentityLinearOperator):
        return Solution(
            value=jnp.asarray(in_size, dtype=float),
            stats={},
            state=state,
        )
    # if diagonal op, then just shortcircuit and sum diagonal
    if isinstance(operator, lx.DiagonalLinearOperator):
        return Solution(
            value=jnp.sum(operator.diagonal),
            stats={},
            state=state,
        )

    # cannot differentiate through key, state, or estimator
    key = eqxi.nondifferentiable(key, name="`trace(key, ...)`")
    state = eqxi.nondifferentiable(state, name="`trace(..., state=...)`")
    estimator = eqxi.nondifferentiable(estimator, name="`trace(..., estimator=...)`")

    # set up state if necessary
    if state == sentinel:
        state = estimator.init(key, operator)
        # we don't want to allow differntiate through trace-alg state, which likely contains the operator
        # or by-products of the operator
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)
        dynamic_state = lax.stop_gradient(dynamic_state)
        state = eqx.combine(dynamic_state, static_state)

    # estimate trace and compute stats if any
    result, stats = _trace(state, estimator, k)

    # cannot differentiate backwards through stats
    stats = eqxi.nondifferentiable_backward(stats, name="_, stats = trace(...)")

    return Solution(value=result, stats=stats, state=state)


@eqx.filter_custom_vjp
def _trace(
    state,
    estimator,
    k,
):
    return estimator.estimate(state, k)


@_trace.def_fwd
def _trace_fwd(
    t_state,
    state,
    estimator,
    k: int,
): ...


@_trace.def_bwd
def _trace_bwd(
    residuals,
    grad_obj,
    t_state,
    state,
    estimator,
    k: int,
): ...
