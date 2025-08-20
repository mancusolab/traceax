[![Documentation-webpage](https://img.shields.io/badge/Docs-Available-brightgreen)](https://mancusolab.github.io/traceax/)
[![PyPI-Server](https://img.shields.io/pypi/v/traceax.svg)](https://pypi.org/project/traceax/)
[![Github](https://img.shields.io/github/stars/mancusolab/traceax?style=social)](https://github.com/mancusolab/traceax)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Project generated with Hatch](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch)

# traceax
``traceax`` is a Python library to perform stochastic trace estimation for linear operators. Namely,
given a square linear operator <i>`A`</i>, ``traceax`` provides flexible routines that estimate,

$$\text{trace}(\mathbf{A}) = \sum_i \mathbf{A}_{ii},$$

using only matrix-vector products. ``traceax`` is heavily inspired by
[lineax](https://github.com/patrick-kidger/lineax) as well as
[XTrace](https://github.com/eepperly/XTrace).

> [!NOTE]
> For preprint, please see: <br/>
> Nahid, A.A., Serafin, L., Mancuso, N. (2025). <i>traceax</i>: a JAX-based framework for stochastic trace estimation. bioRxiv 
> (https://doi.org/10.1101/2025.07.14.662216)

> [!IMPORTANT]
> To reproduce the analyses in the preprint, please see: <br/>
> https://github.com/mancusolab/traceax_analysis

  [**Installation**](#installation)
  | [**Example**](#get-started-with-example)
  | [**Documentation**](#documentation)
  | [**Notes**](#notes)
  | [**Support**](#support)
  | [**Other Software**](#other-software)

------------------

## Installation

Users can directly install from `pip`:

``` bash
pip install traceax
```

Or, users can download the latest repository and then use `pip`:

```
git clone https://github.com/mancusolab/traceax.git
cd traceax
pip install .
```

## Get Started with Example

```python
import jax.numpy as jnp
import jax.random as rdm
import lineax as lx

import traceax as tx

# simulate simple symmetric matrix with exponential eigenvalue decay
seed = 0
N = 1000
key = rdm.PRNGKey(seed)
key, xkey = rdm.split(key)

X = rdm.normal(xkey, (N, N))
Q, R = jnp.linalg.qr(X)
U = jnp.power(0.7, jnp.arange(N))
A = (Q * U) @ Q.T

# should be numerically close
print(jnp.trace(A))  # 3.3333323
print(jnp.sum(U))  # 3.3333335

# setup linear operator
operator = lx.MatrixLinearOperator(A)

# number of matrix vector operators
k = 25

# split key for estimators
key, key1, key2, key3, key4 = rdm.split(key, 5)

# Hutchinson estimator; default samples Rademacher {-1,+1}
hutch = tx.HutchinsonEstimator()
print(hutch.estimate(key1, operator, k))  # (Array(3.4099615, dtype=float32), {})

# Hutch++ estimator; default samples Rademacher {-1,+1}
hpp = tx.HutchPlusPlusEstimator()
print(hpp.estimate(key2, operator, k))  # (Array(3.3033807, dtype=float32), {})

# XTrace estimator; default samples uniformly on n-Sphere
xt = tx.XTraceEstimator()
print(xt.estimate(key3, operator, k))  # (Array(3.3271673, dtype=float32), {'std.err': Array(0.01717775, dtype=float32)})

# XNysTrace estimator; Improved performance for NSD/PSD trace estimates
operator = lx.TaggedLinearOperator(operator, lx.positive_semidefinite_tag)
nt = tx.XNysTraceEstimator()
print(nt.estimate(key4, operator, k))  # (Array(3.3297246, dtype=float32), {'std.err': Array(0.00042093, dtype=float32)})
```

## Documentation
Documentation is available at [here](https://mancusolab.github.io/traceax/).

## Notes

-   `traceax` uses [JAX](https://github.com/google/jax) with [Just In
    Time](https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html)
    compilation to achieve high-speed computation. However, there are
    some [issues](https://github.com/google/jax/issues/5501) for JAX
    with Mac M1 chip. To solve this, users need to initiate conda using
    [miniforge](https://github.com/conda-forge/miniforge), and then
    install `traceax` using `pip` in the desired environment.


## Support

Please report any bugs or feature requests in the [Issue
Tracker](https://github.com/mancusolab/traceax/issues). If users have
any questions or comments, please contact Abdullah Al Nahid (<alnahid@usc.edu>) or
Nicholas Mancuso (<nmancuso@usc.edu>).

## Other Software

Feel free to use other software developed by [Mancuso
Lab](https://www.mancusolab.com/):

-   [SuShiE](https://github.com/mancusolab/sushie): a Bayesian
    fine-mapping framework for molecular QTL data across multiple
    ancestries.
-   [MA-FOCUS](https://github.com/mancusolab/ma-focus): a Bayesian
    fine-mapping framework using
    [TWAS](https://www.nature.com/articles/ng.3506) statistics across
    multiple ancestries to identify the causal genes for complex traits.
-   [SuSiE-PCA](https://github.com/mancusolab/susiepca): a scalable
    Bayesian variable selection technique for sparse principal component
    analysis
-   [twas_sim](https://github.com/mancusolab/twas_sim): a Python
    software to simulate [TWAS](https://www.nature.com/articles/ng.3506)
    statistics.
-   [FactorGo](https://github.com/mancusolab/factorgo): a scalable
    variational factor analysis model that learns pleiotropic factors
    from GWAS summary statistics.
-   [HAMSTA](https://github.com/tszfungc/hamsta): a Python software to
    estimate heritability explained by local ancestry data from
    admixture mapping summary statistics.

------------------------------------------------------------------------

``traceax`` is distributed under the terms of the
[Apache-2.0 license](https://spdx.org/licenses/Apache-2.0.html).


------------------------------------------------------------------------

This project has been set up using Hatch. For details and usage
information on Hatch see <https://github.com/pypa/hatch>.
