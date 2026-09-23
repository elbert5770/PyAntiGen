# PyAntiGen 2 — Study-centred design

Branch `v2`. Breaking release.

## Decisions

| Question | Decision |
| --- | --- |
| Base | PyAntiGen history, branch `v2` |
| Engine | One copy, `pyantigen.engine`, imported from the installed package; projects never carry their own |
| Authoring | Python only, so designs can be generated with loops and functions |
| Design record | One JSON per study (`studies/<name>.json`), never CSV tables |
| Bloat | Factored: each factor level's attributes written once; occasion *generators* stored, never their product |
| Reasoning | Per-project `remarks.json`, time-stamped, append-only, linked by id; rendered to `remarks.html#<id>` |
| Pairing | Every occasion names its subject; contrasts pair within subject by default |
| Forcing | Data forcings become knot *parameters* of a shared piecewise template, not per-subject Antimony text |

## One Engine, installed

1.x copied `Engine/` into every project at `pyantigen-create` time, so each
project ran whatever copy it was born with, and copies drifted apart. In 2.0
the Engine is `pyantigen.engine`, part of the installed package:

* **Projects hold no Engine.** A project imports `pyantigen.engine`; nothing
  in the Engine imports project code (`Modules`, `AntiGen_paths`). The
  project runner passes `MODEL_NAME` and `REPO_ROOT` in `settings`.
* **One environment per project repository.** Install PyAntiGen into the
  project's own virtual environment, editable (`pip install -e
  <PyAntiGen checkout>`) while PyAntiGen is being developed alongside it, or
  pinned to a released version once it is not. Two projects that need
  different Engines get different environments, never different copies.
* **Guard against a second copy.** A project's `AntiGen_paths.py` should
  refuse to run if `pyantigen.engine` is missing or a local `Engine/`
  folder has reappeared.
* **Every result records its Engine.** The run snapshot JSON carries
  `metadata.pyantigen_version` (the git tag and commit), the only record of
  which code produced a result now that projects carry none.

## The design model

```
Factor   name -> {level: attributes}; the reserved attribute "params" holds
         model parameter values that level fixes
Subject  an individual or a cohort; covariates; between-subject factor levels
Protocol events / solver / observed functions, stored as "module:qualname";
         optionally a data loader (inputs for the events, prepared tables)
         and update hooks run after resolve()
Occasion subject x protocol x remaining factor levels (x period): the unit
         that is simulated. An "arm" is only a selection of occasions.
Assay    Measured observables: model Obs + DataSource + Noise (+ only-filter);
         Obs is an expression or a sum of columns as % of its own baseline
Contrast numerator vs denominator occasions, ratio_pct | diff,
         pairing = "subject" (default) | "between"
Param    global, or by=<factor> (one fitted value per level), or fixed values
Rule     applied last: "k_oligo1 = 0 when amyloid_positive is False"
```

`resolve(study, occasion, theta)` gives every parameter value the design sets
for an occasion, in the order level params, fixed Params, fitted theta,
rules. It reads nothing from a RoadRunner instance.

### Why this avoids PEtab's bloat

Froehlich 2018's `experimentalCondition` table is 9,570 rows x 147 columns,
18,060,637 bytes. Every row repeats its cell line's 122 expression values.
The information is 290 cell-line profiles x 122 values plus a treatment table.
`tests/study/test_design.py::test_froehlich_sized_design_stays_small` builds a
Froehlich-sized full cross (290 x 33 = 9,570 occasions) and writes it as
537 KB of readable JSON, 34 times smaller. Three mechanisms:

1. factor attributes are stored on the level, once;
2. `Study.cross` is stored as one record, not 9,570;
3. a factor with 8 or more levels sharing the same attribute names is written
   as a table: column names once, then one row of values per level.

There are no dates or hashes in the design. A run records
`fingerprint(study)` (SHA-256 of the canonical JSON) with its results.

### Pairing, from the start

A crossover is subjects x periods/doses with each subject its own control.
Because every occasion carries its subject, `Contrast(pairing="subject")`
pairs M1-drug with M1-vehicle, M2-drug with M2-vehicle, and raises if any
numerator has no control or more than one. Cohort-mean data are the same
code with the cohort as the subject, so moving from published composite
means to per-animal data changes the subjects, not the contrast. Parallel-
group designs use `pairing="between"` and must name the factors to match on.
Each pair lowers to its own composite loss element, so per-subject random
effects later need no change here. Both simulations of a pair are predicted
at the numerator's data times: the pair's data table is stored on both
sides under a key unique to the pair, because the Engine evaluates each
simulation of a composite at its own table's times and drops one that has
none. Inter-occasion variability (a random
effect per subject x period) has a place to live: the occasion.

### Forcing: parameterized template, not time-course input

RoadRunner compiles per model text. NfL writes each subject's plasma Leu curve
into the Antimony text as a literal piecewise, so every subject is a separate
compile. Measured (40-knot curve, 10 subjects, toy model): literal text 3.58 s
(one compile each); one template with the knots as parameters `t_i`, `v_i`
compiled once in 0.62 s, then 0.11 s for all 10 subjects. Both reproduce
`np.interp` exactly; the two trajectories agree to 1.4e-6 (integrator
tolerance). Compile cost grows with model size, so the gap is larger for
large QSP models.

"Time-course input" (feeding a data series to the integrator) is not
preferred: RoadRunner has no input-function API, so it would mean stopping and
restarting the integrator at every data point, which is slower and, with a
held value between points, less accurate than linear interpolation. Subjects
with different knot counts share a template by padding to the largest count.
`pyantigen.engine.Event_times` currently finds piecewise breakpoints in the
event text; it has to read them from the knot parameters instead (milestone 7).

## Status

| # | Milestone | State |
| --- | --- | --- |
| 0 | `pyantigen.generate` + `pyantigen.engine` | done: `tests/engine` 191 passed, 1 failed (pre-existing, see below) |
| 1 | Engine reads a `Plan` directly | not started; `pyantigen.study.lower` targets today's Engine inputs instead |
| 2 | `pyantigen.study`: design, assays, contrasts, JSON, remarks, validate | done: template Example, and a private four-arm crossover study with a shared vehicle arm, both exact at x0 |
| 3 | `Param` + `resolve()`; arm de-duplication | `resolve()` and by-level Params done; `validate` warns on duplicate occasions; Engine-side de-duplication not started |
| 4-8 | NfL port, stats, shared compilation, NLME | not started |

Acceptance so far: the template Example written as Studies scores exactly as
its 1.x specs at x0: Example1 ADpos 147.4032137918934, Example1 ADneg
128.4185979052745, Example3 229.37082380066985, Example4 -78.76684327376287.
The same holds for a Study reloaded from its committed JSON
(`tests/study/test_example_equivalence.py`).

## Known issues carried over

* `tests/engine/test_profile_quadratic.py::test_checkpoint_wrapper_wiring`
  fails identically in the Engine as it stood before the v2 restructure.
* Example4's profile likelihood does not match `Flipflop_reference.py`: max
  |dNLL error| ~2.1, the second mode is never reached, and CIs assume
  unimodality. The 1.x template Engine gives the same numbers (to 1e-12), so
  this predates v2.
