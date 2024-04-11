# Stochastic Samplers

`traceax` uses a flexible approach to define how random samples are generated within
[`traceax.AbstractTraceEstimator`][] instances. While this typically wraps a single
jax random call, the varied interfaces for each randomization procedure may differ,
which makes uniformly interfacing with it a bit annoying. As such, we provide a
simple abstract class definition, [`traceax.AbstractSampler`][] using that subclasses
[`Equinox`](https://docs.kidger.site/equinox/) modules.

??? abstract "`traceax.AbstractSampler`"
    ::: traceax.AbstractSampler
        options:
            show_bases: false
            members:
            - __call__


::: traceax.NormalSampler
    options:
        members:
        - __init__

---

::: traceax.SphereSampler
    options:
        members:
        - __init__

---

::: traceax.RademacherSampler
    options:
        members:
        - __init__
