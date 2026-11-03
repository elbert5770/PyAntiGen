# Multi-start and the start sampler

Nelder-Mead is a local optimizer, so a fit lands in whichever basin it starts
in. Multi-start runs it from several points and keeps the best. Four fields on
the optimization spec control it:

| field | default | meaning |
|---|---|---|
| `n_starts` | `1` | number of starts. `1` is a single fit and never touches the random generator |
| `search_decades` | `None` | radius around `x0` that starts 2..n are drawn from, in log10 units for a log-scaled parameter. Unset, the radius is half the declared bound range |
| `start_seed` | `None` | seed for the draw. Set it in the same edit that raises `n_starts`, or runs are not reproducible and cached fits cannot resume |
| `start_sampler` | `"lhs"` | how starts 2..n are drawn: `"lhs"` (Latin hypercube) or `"sobol"` (scrambled Sobol sequence) |

Start 1 is always the spec's own `x0`, so a multi-start fit cannot come back
worse than the single fit it replaces. Cost is linear in `n_starts`.

## Choosing a sampler

Both are low-discrepancy samplers from `scipy.stats.qmc`, so for a handful of
starts in a few parameters they behave alike. Sobol fills the box more evenly
as the number of parameters grows.

Sobol's balance holds only when the number of points drawn is a power of two.
Starts 2..n are `n_starts - 1` points, so use `n_starts` of 3, 5, 9, 17, ...
Other counts still run, with a warning from SciPy. Latin hypercube has no such
restriction.

```python
Optimization(
    ...,
    n_starts=9,              # 1 + 8 Sobol points
    start_sampler="sobol",
    search_decades=0.5,
    start_seed=20260909,
)
```

## Keep the radius small

Sample around `x0`, not across the whole bound box. Drawing log-uniformly over
bounds such as (1e-9, 1) puts starts many decades from the answer, and they
converge to far worse basins. A `search_decades` of 0.5 to 1.0 is a reasonable
first try. `Example6` in the template uses 1.0, enough to reach the true mode of
the flip-flop problem from the wrong basin.

## Reading the result

The spread of the final NLLs across starts is the report. If they agree, the
objective is unimodal in that region and one start will do. If they scatter,
the fit depends on where it began, and a profile or confidence interval
anchored on any one of them may be measuring the wrong basin.

## Caching

`start_sampler` is part of the fit-cache key only when it is not `"lhs"`, so
fits cached before the field existed keep their hash. Switching to `"sobol"`
runs a fresh fit.

## What it does not cover

Multi-start applies to local methods. `differential_evolution`,
`basin_hopping`, `dual_annealing` and `shgo` search the whole domain
themselves and ignore `n_starts` and `start_sampler`.

This is distinct from `sobol_analysis`, the variance-based sensitivity
diagnostic (`_SOBOL_ONLY`), which samples the bounds to apportion variance in
the loss and does not search for a minimum.

## Multi-start on the AMICI route (Slurm array)

With `method="LM"` and `--backend amici` each start is one array task (`PYANTIGEN_MULTISTART=<i>`), followed by a
`collect` (`PYANTIGEN_MULTISTART=collect`) that writes `best.json`, `multistart_report.json` and `waterfall.png`
(`python -m pyantigen.engine.Multistart_plot DIR` redraws it). Start 0 is the spec's x0.

**The starts share one set of noise floors.** With `PYANTIGEN_SIGMA_FLOOR=fixed` the floors are the sigmas, and each is
calibrated against the model's shape at the first point a process evaluates. Tasks are separate processes, so without sharing,
each would calibrate at its own random start and the f values would belong to different objectives (no ranking would mean
anything). The first task therefore calibrates at start 0 and writes `floors.pkl` into the start directory; every task loads
it before its first fit evaluation, each `start_<i>.json` carries its stamp, and `collect` ranks only starts that carry the
current stamp (others are listed in `unfinished.txt` and are fitted again from their own start point on relaunch). This is
the AMICI profile's mechanism (`Profile_amici.use_floors`); the Nelder-Mead multi-start gets the same effect by scoring x0
first and seeding its pool from that calibration.

A fit that starts elsewhere calibrates its floors at its own start, so its f is not comparable with a multi-start's. To compare
a promoted optimum with the multi-start, evaluate it under `floors.pkl` (`Floors.load`).
