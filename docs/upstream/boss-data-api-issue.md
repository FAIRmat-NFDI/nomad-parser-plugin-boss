# Draft issue for https://gitlab.com/cest-group/boss/-/issues

**Title:** Programmatic (non-plotting) API for extracting model slice data

---

Hi, and thanks for maintaining BOSS!

## Context

We are integrating BOSS results into [NOMAD](https://nomad-lab.eu), a FAIR data
platform for materials science. Our parser plugin
([nomad-parser-plugin-boss](https://github.com/ndaelman-hu/nomad-parser-plugin-boss))
reads a finished run (`boss.rst` + `boss.out`) and extracts, for every 2D slice of
the parameter space and every iteration, the PES fit mean and uncertainty on a
regular grid. The arrays are stored as HDF5 and rendered interactively in the
browser (H5Web) — so we need **data arrays, not matplotlib figures**.

## What we currently have to do

The only way we found to get these arrays goes through postprocessing internals:

```python
res = BOResults.from_file('boss.rst', 'boss.out')
for iteration in range(iterpts, 0, -1):
    pp = PPMain(
        res,
        pp_models=True,
        pp_iters=[iteration],
        pp_model_slice=[i + 1, j + 1, n_grid],
    )
    X = build_query_points(pp.settings, res.select('x_glmin', iterpts))
    mu, var = res.reconstruct_model(iteration).predict(X)
```

Pain points:

1. `BOResults.reconstruct_model` and `boss.io.dump.build_query_points` are (as far
   as we can tell) undocumented internals, so we cannot rely on them being stable.
2. We instantiate a full `PPMain` per slice per iteration *only* to obtain a
   consistent `settings` object for `build_query_points` — with the side effect
   risk of `PPMain`'s file outputs, and O(n_slices × n_iterations) model
   reconstructions.
3. It is hard to tell from the outside which combination of settings
   (`pp_model_slice`, `pp_iters`, defaults) actually defines the grid we get back,
   i.e. the API is opaque for data-extraction use cases.

## Request

The 1.14.0 postprocessing refactor (Plotter / Graphics / Dumper) is a great step,
but it targets figure creation. Could BOSS expose a documented **data-level** API,
for example:

```python
X, mu, var = res.predict_slice(iteration, dims=(i, j), n_points=50)
```

or a `Dumper`-like object that returns the arrays it would write, without file
I/O? That would let downstream tools (NOMAD, custom dashboards, notebooks) consume
BOSS models robustly, and plotting could remain a thin layer on top.

## Secondary question: GPy dependency

GPy currently pins `scipy<=1.12`, which propagates into downstream environments —
in our NOMAD deployment we had to pin `pymatgen<2025.10.7` to stay compatible with
aalto-boss. Are there plans to migrate to a maintained GP backend (GPyTorch,
GPJax, ...)? Even a rough roadmap statement would help us plan.

Happy to provide more details about our use case, or to test a draft API.
