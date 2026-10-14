# Fitting with a schedule: the bounds check and the AMICI gradient

Two options of `optimizer_kwargs` for a spec whose simulations take their clock-driven inputs from
an `InputSchedule` (see `pyantigen/generate/input_schedule.py`). Both leave a spec without a
schedule exactly as it was.

## `schedule_bounds`: keep the box inside the region the schedule can be integrated in

A schedule can only be integrated while its switches keep their order (`InputSchedule.validate`):
in the SILK hour `0 < t_CSFdraw < V_LP/Q_CSF < 1`. A parameter set where they cross is a failed
evaluation, and a gradient method reads the failure as a cliff. Before the first evaluation the
optimizer checks the bounds of every fitted parameter that moves a switch time against each
simulation's schedule (`InputSchedule.check_box`: the corners of the box, the starting point, and
64 pseudo-random points inside it).

```python
optimizer_kwargs = {"schedule_bounds": "warn"}      # "warn" (default) | "error" | "clip" | "ignore"
```

* `"warn"` prints which corner is infeasible and goes on. A box wider than the region is not
  necessarily wrong, but those evaluations will fail.
* `"error"` raises `ScheduleError` with the same message.
* `"clip"` cuts the bounds of the switch-moving parameters toward x0, by the largest common factor
  for which the box is feasible (`InputSchedule.shrink_box`), and prints what it did.
* A starting point that is itself infeasible stops the fit in every mode except `"ignore"`.

Only simulations whose solver settings carry a `schedule` are checked, and the fitted names are taken
to be the schedule's parameters: a parameter a hook derives from them is not followed.

## `analytic_gradient`: the gradient from AMICI's forward sensitivities

```python
optimizer_kwargs = {
    "method": "L-BFGS-B",
    "analytic_gradient": True,                       # or a dict of the settings below
}
```

Used by every gradient method (`L-BFGS-B`, `BFGS`, `CG`, `TNC`, `SLSQP`): in a single fit and in the
local fits of the multi-start stage, in place of the finite differences. With a gradient-free method it
is ignored (with a note). Needs AMICI >= 1.1 (on Windows, under WSL) and the simulations' inputs as an
`InputSchedule`, or as `at (time >= ...)` event lines `InputSchedule.from_antimony_events` can read.
A problem it cannot handle stops the run before any fit (`Multistart.GradientSetupError`).

| key | default | meaning |
| --- | --- | --- |
| `rtol`, `atol` | 1e-10, 1e-12 | AMICI's tolerances (state and sensitivities) |
| `max_steps` | 2 000 000 | AMICI step cap; the 10 000 h pre-equilibration of SILK needs it |
| `rel_step` | 1e-5 | step of the loss differentiation (below) |
| `cache_dir` | `<plot_path>/amici_models` | compiled models; one directory per distinct model |
| `seed` | none | `{state: amount}` given to AMICI at the start of every run, on top of the model's own initial state (see "A model that starts at zero") |

### What it computes

The objective is the engine's concentrated NLL. Its derivative is not written out again; it is split by
the chain rule into the part AMICI knows and the part the loss knows:

1. AMICI integrates the event-free model of `generate/scaled_time.py` and differentiates every
   observed column (concentrations as `species / compartment`, assigned variables, parameters) with
   respect to the model parameters the fitted ones reach. That table is mapped to the fitted parameters by
   the Jacobian of the engine's own parameter updates (`Update_parameters`, `parameter_hooks`,
   `Update_opt_parameters`), obtained by replaying them on the RoadRunner model.
2. `LossGradient` differentiates the engine's own loss (`collect_loss_blocks`, `concentrated_nll`:
   groups, pooled sigma blocks, sigma floors, observable expressions such as
   `np.log10(np.maximum(predicted_B, 1e-12))`) along that table, by a Richardson central difference of the
   loss alone. It re-does arithmetic on tables in memory: no integration, no noise. A parameter the loss
   names (`SF*predicted_B`) gets its explicit partial derivative too.

The value in the fit is still the engine's (`Multistart.AnalyticObjective`: value from the objective,
gradient from the provider), so the best-so-far tracking, caches and failure handling are unchanged.
If AMICI fails at one point (a solver error, a non-finite sensitivity) that gradient is taken by forward
differences of the objective, counted, and reported with the fit; a zero gradient there would read as
convergence.

### It differentiates the engine's objective, not the model's formulas

Where the two differ, the gradient follows the engine:

* **RoadRunner's `reset()` re-derives a species' initial assignment from the current parameters but not a
  parameter's.** A parameter defined by an initial assignment from a fitted one (SILK's `Q_glymph = Q_CSF`,
  `Q_SN = Q_CSF*f_SN`) keeps the value it had when the model was built while the optimizer moves `Q_CSF`.
  The AMICI model gets those parameters as plain numbers at the engine's values
  (`freeze_stale_assignments`); a schedule input among them keeps its stale starting value when it
  differs from the formula. The stale value is a property of the in-place simulation, and the gradient
  matches it.
* **Every parameter with an initial assignment is frozen**, not only those that read a fitted one. A model
  whose constants are written as expressions (`k = 9.56e-7*70`, `V = V_base`) has them as initial
  assignments, and a project's `Update_parameters` or hooks may overwrite them on the RoadRunner model.
  Each becomes a plain parameter at the value RoadRunner holds, so AMICI integrates what the engine
  integrates, and a hook that moves it is seen by the replay. (Before, only those reading a fitted parameter
  were frozen; the others stayed NaN placeholders that reached AMICI's parameter vector, and a value a hook
  had overwritten was not the one AMICI used.) Variables the schedule assigns, and what their initial values
  are built from, are left alone.
* A species a hook sets directly is handed to AMICI as an initial value (without sensitivities; a hook
  whose value depends on the fitted parameters is refused).

### A model that starts at zero

AMICI differentiates the right-hand side symbolically and evaluates it at the first instant. A rate law
with a fractional power of an amount that is exactly zero there, `(Total/V)^0.8`, has an infinite derivative
at 0, and `0 * inf` is NaN: the run fails at t = 0 with status -42 ("NaN value for sxdot"). RoadRunner does
not differentiate, so the same model integrates without complaint. Give the states in question a negligible
amount, in AMICI only:

```python
optimizer_kwargs = {"analytic_gradient": {"seed": {"AB40_BrainISF": 1e-12, "AB42_BrainISF": 1e-12}}}
```

Read the AMICI log for the names: it lists the first states and rates that were NaN. The effect on the
trajectory is of the order of the seed, forgotten within the first steps of a stable system; check it for
yours by comparing the AMICI values with the engine's, as `Checks/wsl/` does. Unguarded ratios of two
zero amounts (`A / B` as an output) also print a NaN warning at t = 0 but do not stop the run unless the
dynamics read them.

### Limits

* Time points: the loss is differentiated at the data times and the ends of the tracked window; the
  engine interpolates a dense grid linearly between them. The two differ by the interpolation error
  (a few 1e-4 of the gradient's scale on SILK).
* The fitted parameters must reach the model through plain parameters (constants without rule or
  initial assignment) or hooks onto them. A fitted species initial value, or a parameter with an initial
  assignment, is refused with a message naming it.
* Columns that are not species, parameters or compartments (reaction rates) are not available to the
  loss; the gradient is then zero for anything that depends only on them.
* The profile-likelihood stage keeps its own optimizers.
  The gradient can differentiate what a profile point minimizes: `provider.freeze_sigmas(block_sigmas)` pins the floored blocks'
  sigmas as `evaluate_nll_fixed(frozen_sigmas=...)` does (without it the gradient is off by 10 to 30% away from the fit);
  `freeze_sigmas(None)` releases them. Nothing calls it yet.
* Cost: the gradient costs one AMICI forward-sensitivity run, which grows with the number of
  parameters it differentiates. On SILK, 8 parameters: about 1.2 s a gradient against 9 objective
  evaluations (about 1.4 s) for forward differences; with all 48 parameters AMICI needs about 40 s a
  gradient against about 8 s for 49 evaluations. The case for it is a gradient without the noise of the
  integrator tolerance and without a step to choose, not speed at every size. Fit only the parameters
  that need fitting: the model is compiled for exactly those.

Checks on the real SILK model are in the examples repository,
`Projects/Elbert_2022_PyAntiGen_v4/Checks/wsl/fit_gradient.py`.
