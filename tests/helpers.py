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

import jax.numpy as jnp
import jax.random as rdm
import lineax as lx


def has_tag(tags, tag):
    return tag is tags or (isinstance(tags, tuple) and tag in tags)


def construct_matrix(getkey, tags, size, dtype):
    matrix = rdm.normal(getkey(), (size, size), dtype=dtype)
    if has_tag(tags, lx.diagonal_tag):
        matrix = jnp.diag(jnp.diag(matrix))
    if has_tag(tags, lx.symmetric_tag):
        matrix = matrix + matrix.T
    if has_tag(tags, lx.lower_triangular_tag):
        matrix = jnp.tril(matrix)
    if has_tag(tags, lx.upper_triangular_tag):
        matrix = jnp.triu(matrix)
    if has_tag(tags, lx.unit_diagonal_tag):
        matrix = matrix.at[jnp.arange(size), jnp.arange(size)].set(1)
    if has_tag(tags, lx.tridiagonal_tag):
        diagonal = jnp.diag(jnp.diag(matrix))
        upper_diagonal = jnp.diag(jnp.diag(matrix, k=1), k=1)
        lower_diagonal = jnp.diag(jnp.diag(matrix, k=-1), k=-1)
        matrix = lower_diagonal + diagonal + upper_diagonal
    if has_tag(tags, lx.positive_semidefinite_tag):
        matrix = matrix @ matrix.T.conj()
    if has_tag(tags, lx.negative_semidefinite_tag):
        matrix = -matrix @ matrix.T.conj()

    return matrix
