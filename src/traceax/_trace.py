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

from typing import Any, Literal, TypeAlias

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.lax as lax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.tree_util as jtu
import lineax as lx

from jax.flatten_util import ravel_pytree
from jax.interpreters import ad as ad, mlir as mlir
from jax.numpy.linalg import norm
from jaxtyping import Array, PRNGKeyArray, PyTree

from ._estimators import AbstractEstimator
from ._samplers import AbstractSampler, RademacherSampler, SphereSampler
from ._solution import Solution
from ._utils import (
    _assert_false,
    _check_operator,
    _clip_k,
    _is_none,
    _to_shapedarray,
    _to_struct,
    sentinel,
)


def _get_scale(W: Array, D: Array, n: int, k: int) -> Array:
    """Compute the XTrace/XNysTrace finite-population correction."""
    return (n - k + 1) / (n - norm(W, axis=0) ** 2 + jnp.abs(D) ** 2)


def _check_k(k: int, n: int, estimator_name: str, *, min_k: int = 1) -> int:
    """Validate and clip the probe count before entering traced estimator code.

    `k` controls sampled array shapes, so it must be a Python integer at the
    public boundary. Letting a JAX scalar through would fail later with a tracer
    boolean-conversion error from Python `min`/`max`, which is much harder for
    users to interpret.
    """
    if type(k) is not int:
        raise TypeError(f"`k` must be a Python int. Found {type(k).__name__}.")

    k = _clip_k(k, n)
    if k < min_k:
        raise ValueError(f"`{estimator_name}` requires `k >= {min_k}` after clipping to operator size. Found {k}.")

    return k


def _structure_is_complex(structure: PyTree[Any]) -> bool:
    """Return whether any ShapeDtypeStruct-like leaf carries a complex dtype."""
    for leaf in jtu.tree_leaves(structure):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None and jnp.issubdtype(dtype, jnp.complexfloating):
            return True
    return False


def _check_real_operator(operator: lx.AbstractLinearOperator) -> None:
    """Reject complex operators until estimator formulas use Hermitian products."""
    if _structure_is_complex(operator.in_structure()) or _structure_is_complex(operator.out_structure()):
        raise ValueError("`traceax.trace` does not currently support complex-valued linear operators.")


def _unravel_vector(vector: Array, structure: PyTree[jax.ShapeDtypeStruct]) -> PyTree[Array]:
    """Convert one flat probe vector into an operator-compatible PyTree."""
    prototype = jtu.tree_map(lambda leaf: jnp.zeros(leaf.shape, leaf.dtype), structure)
    flat_prototype, unravel = ravel_pytree(prototype)
    # Samplers may produce integer Rademacher probes or higher-precision sphere
    # probes. Cast before unravelling so Lineax function/Jacobian operators see
    # vectors matching their declared input structure.
    return unravel(vector.astype(flat_prototype.dtype))


def _ravel_vector(pytree: PyTree[Array]) -> Array:
    """Flatten one operator output PyTree into trace-estimator coordinates."""
    flat, _ = ravel_pytree(pytree)
    return flat


def _sum_leaves(pytree: PyTree[Array]) -> Array:
    """Sum every array leaf in a PyTree."""
    leaf_sums = eqxi.ω(pytree).call(jnp.sum).ω
    return jtu.tree_reduce(lambda left, right: left + right, leaf_sums)


def _project_probes(operator: lx.AbstractLinearOperator, samples: Array) -> Array:
    """Apply an abstract operator to flat probe columns.

    Estimators such as Hutch++ and XTrace need dense probe matrices for QR and
    correction formulas. Lineax operators, however, may act on PyTrees. This
    helper keeps estimator algebra in flat coordinates while respecting each
    operator's input/output PyTree structure at the `mv` boundary.
    """
    input_structure = operator.in_structure()

    def mv_flat(sample):
        vector = _unravel_vector(sample, input_structure)
        return _ravel_vector(operator.mv(vector))

    return jax.vmap(mv_flat, in_axes=1, out_axes=1)(samples)


def _linear_probe_trace(key: PRNGKeyArray, operator: lx.AbstractLinearOperator, k: int) -> PyTree[Array]:
    """Estimate `trace(operator)` with the universal functional-mode fallback.

    Functional AD needs a linear stochastic approximation to `trace(dA)`. The
    Hutchinson rule is the least-assumptive fallback because it only requires
    fixed probes and operator-vector products.
    """
    estimator = HutchinsonEstimator()
    state = estimator.init(key, operator)
    value, _stats = estimator.estimate(state, k)
    return value


_BasicTraceState: TypeAlias = tuple[PRNGKeyArray, lx.AbstractLinearOperator, int]
_PSDTraceState: TypeAlias = tuple[PRNGKeyArray, lx.AbstractLinearOperator, int, bool]
TraceGradMode: TypeAlias = Literal["functional", "estimator"]
_TRACE_GRAD_MODES = ("functional", "estimator")


class HutchinsonEstimator(AbstractEstimator[_BasicTraceState], strict=True):
    r"""Girard-Hutchinson Trace Estimator:

    $\mathbb{E}[\omega^T \mathbf{A} \omega] = \text{trace}(\mathbf{A})$,
    where $\mathbb{E}[\omega] = 0$ and $\mathbb{E}[\omega \omega^T] = \mathbf{I}$.

    """

    sampler: AbstractSampler = RademacherSampler()

    def init(self, key: PRNGKeyArray, operator: lx.AbstractLinearOperator) -> _BasicTraceState:
        """Create reusable state for a fixed key/operator pair."""
        n = _check_operator(operator)
        return (key, operator, n)

    def estimate(self, state: _BasicTraceState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        """Estimate the trace with the Girard-Hutchinson identity."""
        key, operator, n = state

        k = _check_k(k, n, "HutchinsonEstimator")

        # The sampler contract enforces E[zz^T] = I, making z^T A z an
        # unbiased estimate of trace(A) for real-valued operators.
        samples = self.sampler(key, n, k)
        projected = _project_probes(operator, samples)
        trace_est = jnp.sum(projected * samples) / k

        return trace_est, {}

    def transpose(self, state: _BasicTraceState) -> _BasicTraceState:
        """Return estimator state for the transposed linear operator."""
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
        """Create reusable state for a fixed key/operator pair."""
        n = _check_operator(operator)
        return (key, operator, n)

    def estimate(self, state: _BasicTraceState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        """Estimate the trace with Hutch++ low-rank variance reduction."""
        key, operator, n = state

        k = _check_k(k, n, "HutchPlusPlusEstimator", min_k=3)
        m = k // 3

        # Probe matrices live in flattened coordinates. `_project_probes`
        # handles the conversion to and from the operator's input/output PyTrees.
        mv = ft.partial(_project_probes, operator)

        # Hutch++ spends one block on the range sketch and one independent block
        # on the residual trace estimate.
        samples = self.sampler(key, n, 2 * m)
        X1 = samples[:, :m]
        X2 = samples[:, m:]

        Y = mv(X1)

        # Q spans the sampled range of A. The residual term below estimates the
        # trace not captured by this low-rank subspace.
        Q, _ = jnp.linalg.qr(Y)
        G = X2 - Q @ (Q.T @ X2)

        AQ = mv(Q)
        AG = mv(G)
        trace_est = jnp.sum(AQ * Q) + jnp.sum(AG * G) / (G.shape[1])

        return trace_est, {}

    def transpose(self, state: _BasicTraceState) -> _BasicTraceState:
        """Return estimator state for the transposed linear operator."""
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
    improved: bool = eqx.field(static=True, default=True)

    def init(self, key: PRNGKeyArray, operator: lx.AbstractLinearOperator) -> _BasicTraceState:
        """Create reusable state for a fixed key/operator pair."""
        n = _check_operator(operator)
        return (key, operator, n)

    def estimate(self, state: _BasicTraceState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        """Estimate the trace with the exchangeable XTrace estimator."""
        key, operator, n = state

        k = _check_k(k, n, "XTraceEstimator", min_k=2)
        m = k // 2

        # Keep the estimator operator-based. Many Lineax operators are implicit
        # and can only promise a vector `mv`, not dense matrix multiplication.
        mv = ft.partial(_project_probes, operator)

        samples = self.sampler(key, n, m)
        Y = mv(samples)
        Q, R = jnp.linalg.qr(Y)

        # XTrace uses the inverse R factor to construct leave-one-out style
        # weights. Normalization keeps these weights comparable across probes.
        S = jnp.linalg.inv(R).T
        s = norm(S, axis=0)
        S = S / s

        # Cache all projected quantities once. This makes the exchangeable
        # correction formula below read directly in terms of its mathematical
        # components and avoids repeated operator evaluations.
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
            scale = _get_scale(W, SW_d, n, m)
        else:
            scale = 1

        estimates = jnp.trace(H) * jnp.ones(m) - SHS_d + (WHW_d - TW_d + term1 + term2 + term3) * scale
        trace_est = jnp.mean(estimates)
        std_err = jnp.std(estimates) / jnp.sqrt(m)

        return trace_est, {"std.err": std_err}

    def transpose(self, state: _BasicTraceState) -> _BasicTraceState:
        """Return estimator state for the transposed linear operator."""
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
    improved: bool = eqx.field(static=True, default=True)

    def init(self, key: PRNGKeyArray, operator: lx.AbstractLinearOperator) -> _PSDTraceState:
        """Create state for a positive- or negative-semidefinite operator."""
        n = _check_operator(operator)
        is_nsd = lx.is_negative_semidefinite(operator)
        if not (lx.is_positive_semidefinite(operator) | is_nsd):
            raise ValueError("`XNysTraceEstimator` may only be used for positive or negative definite linear operators")
        if is_nsd:
            operator = -operator

        return (key, operator, n, is_nsd)

    def estimate(self, state: _PSDTraceState, k: int) -> tuple[PyTree[Array], dict[str, Any]]:
        """Estimate the trace with the Nyström variant for semidefinite operators."""
        key, operator, n, is_nsd = state

        k = _check_k(k, n, "XNysTraceEstimator")

        # XNysTrace is still an operator algorithm: all dense blocks below are
        # probe matrices, not materializations of the linear operator.
        mv = ft.partial(_project_probes, operator)

        samples = self.sampler(key, n, k)
        Y = mv(samples)

        # The small diagonal shift stabilizes the Cholesky path for nearly
        # singular sketches without changing the trace estimate at leading order.
        nu = jnp.finfo(Y.dtype).eps * norm(Y, "fro") / jnp.sqrt(n)
        Y = Y + samples * nu
        Q, R = jnp.linalg.qr(Y)

        # H should be symmetric for PSD operators, but symmetrizing protects the
        # Cholesky factorization from floating-point asymmetry.
        H = samples.T @ Y
        C = jnp.linalg.cholesky(0.5 * (H + H.T)).T
        B = jsp.linalg.solve_triangular(C.T, R.T, lower=True).T

        Qs, Rs = jnp.linalg.qr(samples)
        Ws = Qs.T @ samples

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
        """Return estimator state for the transposed linear operator."""
        key, operator, n, is_nsd = state
        return key, operator.transpose(), n, is_nsd


XNysTraceEstimator.__init__.__doc__ = r"""**Arguments:**

- `sampler`: the sampling distribution for $\omega$. Default is [`traceax.SphereSampler`][].
- `improved`: whether to use the *improved* XNysTrace estimator, which rescales predicted samples.
    Default is `True` (see Notes).
"""


def _functional_trace_estimate(
    key: PRNGKeyArray,
    primal_operator: lx.AbstractLinearOperator,
    operator: lx.AbstractLinearOperator,
    k: int,
    estimator: AbstractEstimator,
) -> PyTree[Array]:
    """Estimate `trace(operator)` for the functional-mode AD rule.

    `primal_operator` may be used to build variance-reducing sketches, but the
    returned scalar must remain linear in `operator`. This keeps JVPs and VJPs
    tied to the trace functional, `trace(dA)`, instead of to the derivative of a
    nonlinear estimator program.
    """
    if isinstance(estimator, HutchPlusPlusEstimator):
        return _hutchplusplus_frozen_sketch_trace(key, primal_operator, operator, k, estimator)
    return _linear_probe_trace(key, operator, k)


def _hutchplusplus_frozen_sketch_trace(
    key: PRNGKeyArray,
    primal_operator: lx.AbstractLinearOperator,
    operator: lx.AbstractLinearOperator,
    k: int,
    estimator: HutchPlusPlusEstimator,
) -> PyTree[Array]:
    """Estimate `trace(operator)` with a frozen Hutch++ sketch.

    The low-rank basis is adapted to `primal_operator` for variance reduction,
    then stopped before applying `operator`. In functional AD, `operator` is the
    tangent operator in JVPs and the live primal operator in VJPs.
    """
    n = _check_operator(primal_operator)
    k = _check_k(k, n, "HutchPlusPlusEstimator", min_k=3)
    m = k // 3

    primal_mv = ft.partial(_project_probes, primal_operator)
    operator_mv = ft.partial(_project_probes, operator)

    samples = estimator.sampler(key, n, 2 * m)
    X1 = samples[:, :m]
    X2 = samples[:, m:]

    Y = primal_mv(X1)
    Q, _ = jnp.linalg.qr(Y)
    Q = lax.stop_gradient(Q)

    G = X2 - Q @ (Q.T @ X2)
    G = lax.stop_gradient(G)

    AQ = operator_mv(Q)
    AG = operator_mv(G)
    return jnp.sum(AQ * Q) + jnp.sum(AG * G) / G.shape[1]


class _TangentLinearOperator(lx.AbstractLinearOperator):
    """Represent an operator tangent as a Lineax operator.

    The functional JVP needs to evaluate `trace(V)` where `V` is the tangent of
    an arbitrary `AbstractLinearOperator` PyTree. Lineax does not require that an
    operator can materialize as a matrix, so this wrapper defines `mv` by pushing
    a JVP through the primal operator's `mv` method.
    """

    primal: lx.AbstractLinearOperator
    tangent: lx.AbstractLinearOperator
    scalar: Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        assert type(self.primal) is type(self.tangent)  # noqa: E721

    def mv(self, vector):
        """Apply the tangent operator to one vector."""
        mv = lambda operator: operator.mv(vector)
        out, t_out = eqx.filter_jvp(mv, (self.primal,), (self.tangent,))
        # `filter_jvp` returns `None` for nondifferentiable/static leaves. Lineax
        # expects an array-valued vector output, so missing tangents represent
        # zeros in the output structure.
        tangent_out = jtu.tree_map(eqxi.materialise_zeros, out, t_out, is_leaf=_is_none)
        return (eqxi.ω(tangent_out) * self.scalar).ω

    def as_matrix(self):
        """Materialize the tangent only when the primal operator supports it."""
        as_matrix = lambda operator: operator.as_matrix()
        out, t_out = eqx.filter_jvp(as_matrix, (self.primal,), (self.tangent,))
        return jtu.tree_map(eqxi.materialise_zeros, out, t_out, is_leaf=_is_none)

    def transpose(self):
        """Transpose both primal and tangent through Lineax's operator API."""
        transpose = lambda operator: operator.transpose()
        primal_out, tangent_out = eqx.filter_jvp(transpose, (self.primal,), (self.tangent,))
        return _TangentLinearOperator(primal_out, tangent_out, scalar=self.scalar)

    def in_structure(self):
        return self.primal.in_structure()

    def out_structure(self):
        return self.primal.out_structure()


@lx.is_symmetric.register(_TangentLinearOperator)
def _(op):
    return False


@lx.is_positive_semidefinite.register(_TangentLinearOperator)
def _(op):
    return False


@lx.is_negative_semidefinite.register(_TangentLinearOperator)
def _(op):
    return False


def _refresh_builtin_state_operator(
    state: PyTree[Any], operator: lx.AbstractLinearOperator, estimator: AbstractEstimator
) -> PyTree[Any]:
    """Rebind built-in estimator state to the operator from the current call.

    Built-in states store the operator so that users can reuse random probes.
    Without this refresh, a caller could pass stale state from a previous
    operator and receive a value for the old operator but a gradient for the new
    one. Custom estimator states are left untouched because traceax cannot know
    their layout.
    """
    if isinstance(estimator, HutchinsonEstimator | HutchPlusPlusEstimator | XTraceEstimator | XNysTraceEstimator):
        state_key = state[0]
        return estimator.init(state_key, operator)
    return state


def _estimate_trace_impl(key, operator, state, k, estimator, *, check_closure):
    """Implementation shared by abstract evaluation and primitive execution."""
    out = estimator.estimate(state, k)
    if check_closure:
        # Closed-over JAX arrays should be explicit PyTree leaves. Treating them
        # as Python closure data can produce invalid traces under JIT/AD.
        out = eqxi.nontraceable(out, name="`traceax.trace` with respect to a closed-over value")
    result, stats = out

    return result, stats


_to_struct_tr = ft.partial(_to_struct, name="traceax.trace")


@eqxi.filter_primitive_def
def _estimate_trace_abstract_eval(key, operator, state, k, estimator):
    """Infer the primitive output structure without running estimator numerics."""
    key, state, k, estimator = jtu.tree_map(_to_struct_tr, (key, state, k, estimator))
    out = eqx.filter_eval_shape(
        _estimate_trace_impl,
        key,
        operator,
        state,
        k,
        estimator,
        check_closure=False,
    )
    out = jtu.tree_map(_to_shapedarray, out)

    return out


@eqxi.filter_primitive_jvp
def _estimate_trace_jvp(primals, tangents):
    """Functional-mode JVP rule.

    The primal output remains the requested estimator's value. The tangent output
    estimates the derivative of the trace functional, `trace(dA)`, with the same
    probe key. This keeps JVP linear in the operator tangent even for nonlinear
    variance-reduced estimators such as Hutch++ and XTrace.
    """
    key, operator, state, k, estimator = primals

    t_key, t_operator, t_state, t_k, t_estimator = tangents
    # The stochastic key, estimator state, probe count, and estimator object are
    # configuration for the stochastic oracle. Differentiating them is undefined.
    jtu.tree_map(_assert_false, (t_key, t_state, t_k, t_estimator))
    del t_key, t_state, t_k, t_estimator

    # compute the primal
    result, stats = eqxi.filter_primitive_bind(_estimate_trace_p, key, operator, state, k, estimator)
    out = result, stats

    # The tangent problem is `trace(V)`, where `V` is the operator tangent. We
    # wrap `V` as a Lineax operator and estimate that trace with the
    # estimator-specific functional AD surrogate. Surrogates may use the primal
    # operator to build stopped sketches, but must remain linear in `V`.
    t_operator = _TangentLinearOperator(operator, t_operator, scalar=jnp.asarray(1.0))

    # Cast to the primal estimator dtype so JAX sees a dtype-consistent custom
    # JVP even when the primal estimator samples in float64 and the operator is
    # float32.
    t_result = jnp.asarray(_functional_trace_estimate(key, operator, t_operator, k, estimator), dtype=result.dtype)
    t_out = (
        t_result,
        jtu.tree_map(lambda _: None, stats),
    )

    return out, t_out


@eqxi.filter_primitive_transpose(materialise_zeros=True)  # pyright: ignore
def _estimate_trace_transpose(inputs, cts_out):
    """Functional-mode transpose rule.

    Reverse mode must return a cotangent PyTree matching the primal operator,
    not a matrix-like object. We get that by differentiating the same functional
    AD surrogate used by the JVP and asking Equinox for its pullback.
    """
    key, operator, state, k, estimator = inputs

    cts_result, _ = cts_out

    def probe_trace(primal_operator):
        # Match the incoming cotangent dtype for the same reason as the JVP cast:
        # JAX transpose rules are stricter about mixed dtypes than ordinary
        # `jnp` promotion.
        estimate = _functional_trace_estimate(key, primal_operator, primal_operator, k, estimator)
        return jnp.asarray(estimate, dtype=cts_result.dtype)

    _, pullback = eqx.filter_vjp(probe_trace, operator.primal)
    (op_ct,) = pullback(cts_result)
    # Only the operator is differentiable. The rest of the primitive inputs are
    # stochastic/configuration data and therefore receive `None` cotangents.
    key_none = jtu.tree_map(lambda _: None, key)
    state_none = jtu.tree_map(lambda _: None, state)
    k_none = None
    estimator_none = jtu.tree_map(lambda _: None, estimator)

    return key_none, op_ct, state_none, k_none, estimator_none


_estimate_trace_p = eqxi.create_vprim(
    "trace",
    eqxi.filter_primitive_def(ft.partial(_estimate_trace_impl, check_closure=False)),
    _estimate_trace_abstract_eval,
    _estimate_trace_jvp,
    _estimate_trace_transpose,
)
# Rebind the concrete implementation with closure checks after shape inference
# has been registered. Abstract evaluation cannot use `nontraceable` because it
# must operate on shape/dtype structures rather than concrete arrays.
_estimate_trace_p.def_impl(
    eqxi.filter_primitive_def(ft.partial(_estimate_trace_impl, check_closure=True)),
)
eqxi.register_impl_finalisation(_estimate_trace_p)


@eqx.filter_jit
def trace(
    key: PRNGKeyArray,
    operator: lx.AbstractLinearOperator,
    k: int,
    estimator: AbstractEstimator = XTraceEstimator(),
    *,
    state: PyTree[Any] = sentinel,
    grad_mode: TraceGradMode = "functional",
) -> Solution:
    r"""Estimate the trace of a linear operator.

    !!! warning "Automatic differentiation semantics"

        By default, `traceax.trace` defines AD for the trace functional, not for
        the realized estimator program. With `grad_mode="functional"`, the
        primal value is computed with `estimator`, while JVP/VJP rules estimate
        `trace(dA)` with an estimator-specific functional surrogate. Hutch++
        uses a frozen low-rank sketch: the sketch is built from the primal
        operator and treated as constant for AD, then the tangent operator is
        applied to the stopped sketch vectors. Estimators without a specialized
        functional rule use a fixed-probe Hutchinson fallback. This is useful as
        a stochastic gradient oracle for objectives involving `trace(A)`, but
        the returned gradient is not generally the gradient of the exact sampled
        estimator scalar.

        Use `grad_mode="estimator"` to differentiate through the estimator
        implementation itself, including QR, low-rank bases, correction terms,
        and other estimator-specific computations.

        Simple operators may bypass the estimator. Identity and diagonal
        operators return exact traces before estimator-specific logic or
        `grad_mode` handling, so their gradients follow the exact shortcut.

    **Arguments:**

    - `key`: PRNG key used by stochastic trace estimators.
    - `operator`: Square real-valued `lineax.AbstractLinearOperator`.
    - `k`: Python `int` controlling the number of probe vectors. Estimators may
        require a minimum `k`.
    - `estimator`: Trace estimator used for the primal estimate.
    - `state`: Optional reusable estimator state. For built-in estimators this
        preserves the state's PRNG key but refreshes the state around the current
        operator, so the returned value and AD rule refer to the same operator.
    - `grad_mode`: AD semantics. `"functional"` returns a stochastic gradient
        oracle for the trace functional, using specialized frozen-sketch rules
        where available and a Hutchinson probe fallback otherwise. `"estimator"`
        differentiates through the estimator program itself. Exact identity and
        diagonal shortcuts are used before estimator-specific logic.
    """
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
    _check_real_operator(operator)
    _check_k(k, in_size, type(estimator).__name__)

    if grad_mode not in _TRACE_GRAD_MODES:
        raise ValueError(f"`grad_mode` must be one of {_TRACE_GRAD_MODES}. Found {grad_mode!r}.")

    # These exact cases avoid stochastic noise and preserve the simplest AD
    # contract: the gradient is the exact trace derivative of the shortcut.
    if isinstance(operator, lx.IdentityLinearOperator):
        return Solution(
            value=jnp.asarray(in_size, dtype=float),
            stats={},
            state=state,
        )
    if isinstance(operator, lx.DiagonalLinearOperator):
        return Solution(
            value=_sum_leaves(operator.diagonal),
            stats={},
            state=state,
        )

    # set up state if necessary
    if state == sentinel:
        state = estimator.init(key, operator)
    else:
        # Built-in states carry the operator internally. Refreshing the state
        # preserves reproducibility from the state key while avoiding stale
        # operator values.
        state = _refresh_builtin_state_operator(state, operator, estimator)

    if grad_mode == "functional":
        # We don't differentiate through trace-algorithm state, which may contain
        # the operator or estimator by-products. The primitive owns the AD rule.
        dynamic_state, static_state = eqx.partition(state, eqx.is_array)
        dynamic_state = lax.stop_gradient(dynamic_state)
        state = eqx.combine(dynamic_state, static_state)

    # cannot differentiate through key or estimator
    key = eqxi.nondifferentiable(key, name="`trace(key, ...)`")
    estimator = eqxi.nondifferentiable(estimator, name="`trace(..., estimator=...)`")

    if grad_mode == "functional":
        # Functional mode owns its derivative rule through the primitive, so the
        # estimator state is treated as nondifferentiable bookkeeping.
        state = eqxi.nondifferentiable(state, name="`trace(..., state=...)`")
        result, stats = eqxi.filter_primitive_bind(_estimate_trace_p, key, operator, state, k, estimator)
    else:
        # Estimator mode deliberately exposes the estimator implementation to
        # JAX AD, including QR factorizations and correction terms.
        result, stats = estimator.estimate(state, k)

    # Stats are diagnostics rather than optimization variables. Blocking their
    # cotangents avoids accidental differentiation through metadata.
    stats = eqxi.nondifferentiable_backward(stats, name="_, stats = trace(...)")

    return Solution(value=result, stats=stats, state=state)
