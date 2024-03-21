.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

.. image:: https://img.shields.io/badge/Docs-Available-brightgreen
    :alt: Documentation-webpage
    :target: https://mancusolab.github.io/traceax/

.. image:: https://img.shields.io/pypi/v/traceax.svg
    :alt: PyPI-Server
    :target: https://pypi.org/project/traceax/

.. image:: https://img.shields.io/github/stars/mancusolab/traceax?style=social
    :alt: Github
    :target: https://github.com/mancusolab/traceax

.. image:: https://img.shields.io/badge/License-MIT-yellow.svg
    :alt: License
    :target: https://opensource.org/licenses/MIT

.. image:: https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg
    :alt: Project generated with Hatch
    :target: https://github.com/pypa/hatch

======
GiddyUp
======
``traceax`` is a Python library to perform stochastic trace estimation of linear operators.


|Installation|_ | |Example|_ | |Notes|_ | |Version|_ | |Support|_ | |Other Software|_

=================

.. _Installation:
.. |Installation| replace:: **Installation**

Installation
============
Users can download the latest repository and then use ``pip``:

.. code:: bash

    git clone https://github.com/mancusolab/traceax.git
    cd traceax
    pip install .

.. _Example:
.. |Example| replace:: **Example**

Quick Example
========================
.. code:: python

   import jax.numpy as jnp
   import jax.random as rdm
   import lineax as lx

   import traceax as tr

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
   print(jnp.trace(A)) # 3.3333323
   print(jnp.sum(U)) # 3.3333335

   # setup linear operator
   Aop = lx.MatrixLinearOperator(A)

   # number of matrix vector operators
   k = 10

   # split key for estimators
   key, key1, key2, key3 = rdm.split(key, 4)

   # sampler
   sampler = tr.NormalSampler()

   # hutch estimator
   hutch = tr.HutchinsonEstimator()
   print(hutch.compute(key1, k, Aop, sampler)) # (Array(3.4798508, dtype=float32), {})

   # hutch++ estimator
   hpp = tr.HutchPlusPlusEstimator()
   print(hpp.compute(key2, k, Aop, sampler)) # (Array(3.671408, dtype=float32), {})

   # XTrace estimator
   xt = tr.XTraceEstimator(scale=False)
   print(xt.compute(key3, k, Aop, sampler)) # (Array(3.1899667, dtype=float32), {'std.err': Array(0.2524434, dtype=float32)})


.. _Notes:
.. |Notes| replace:: **Notes**

Notes
=====
* ``traceax`` uses `JAX <https://github.com/google/jax>`_ with `Just In Time  <https://jax.readthedocs.io/en/latest/jax-101/02-jitting.html>`_ compilation to achieve high-speed computation. However, there are some `issues <https://github.com/google/jax/issues/5501>`_ for JAX with Mac M1 chip. To solve this, users need to initiate conda using `miniforge <https://github.com/conda-forge/miniforge>`_, and then install ``traceax`` using ``pip`` in the desired environment.

.. _Version:
.. |Version| replace:: **Version**

Version History
===============
TBD

.. _Support:
.. |Support| replace:: **Support**


Support
=======

Please report any bugs or feature requests in the `Issue Tracker <https://github.com/mancusolab/traceax/issues>`_.
If users have any questions or comments, please contact Nicholas Mancuso (nmancuso@usc.edu).

.. _OtherSoftware:
.. |Other Software| replace:: **Other Software**

Other Software
==============

Feel free to use other software developed by `Mancuso Lab <https://www.mancusolab.com/>`_:

* `SuShiE <https://github.com/mancusolab/sushie>`_: a Bayesian fine-mapping framework for molecular QTL data across multiple ancestries.

* `MA-FOCUS <https://github.com/mancusolab/ma-focus>`_: a Bayesian fine-mapping framework using `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics across multiple ancestries to identify the causal genes for complex traits.

* `SuSiE-PCA <https://github.com/mancusolab/susiepca>`_: a scalable Bayesian variable selection technique for sparse principal component analysis

* `twas_sim <https://github.com/mancusolab/twas_sim>`_: a Python software to simulate `TWAS <https://www.nature.com/articles/ng.3506>`_ statistics.

* `FactorGo <https://github.com/mancusolab/factorgo>`_: a scalable variational factor analysis model that learns pleiotropic factors from GWAS summary statistics.

* `HAMSTA <https://github.com/tszfungc/hamsta>`_: a Python software to  estimate heritability explained by local ancestry data from admixture mapping summary statistics.

---------------------

.. _license:

``traceax`` is distributed under the terms of the `Apache-2.0 license <https://spdx.org/licenses/Apache-2.0.html>`_ license.


---------------------

.. _hatch-notes:

This project has been set up using Hatch. For details and usage
information on Hatch see https://github.com/pypa/hatch.
