# PyAntiGen model-development pipeline: review and catalogue of improvements

Written 2026-10-28, at the end of the Elbert_Esguerra_20260923 calibration round (PK, silk_appfull, aggregation_silk,
microglia_activation, APOE multipliers, Plasma). Every item below says **what was seen** (a job, a file, a commit) and its
**status**: *built* (in the repo now), *not built*, or *unverified* (a hypothesis nobody has tested). Where I did not check
something, it says so.

## 1. The split: RoadRunner to develop, AMICI to characterize

**Why it works.** RoadRunner takes any Antimony model with no compile step, so a reaction edit is testable in minutes. AMICI
gives exact forward sensitivities, so the things that need many evaluations (FIM, a 36-side profile, a Levenberg-Marquardt fit)
cost 30 to 160 s per evaluation instead of days. On every problem compared this round the two routes landed on the same optimum
when they scored the same objective:

| Problem | AMICI route | Older / other route | Difference |
|---|---|---|---|
| silk_appfull | f -3007.41689 (engine LM) | -3007.41696 (pypesto LM) | 7e-5 |
| silk_appfull profile | 32 limits | old script's profile | all within 0.67 % (the largest, `Q_osc_SP` down) |
| engine AMICI vs RoadRunner at one point | -3005.856 / -3007.4169 | -3005.895 / -3007.461 | about 0.04 |

**What can go wrong with the split** (the substance of this document, section 2 lists the fixes):

1. **The two routes score different objectives unless told otherwise.** AMICI needs every sigma known (`PYANTIGEN_SIGMA_FLOOR=fixed`);
   RoadRunner defaults to the *cap* mode (`min(sigma_hat, floor)`, re-estimated at every point). The same parameters give
   silk_appfull -3028.67 (cap) and -3005.9 (fixed): a 22.8-nat offset that looks like a model change and is not (a cap-mode
   evaluation on the current model reproduced the Oct 25 value exactly, job 42473838). A model developed in cap mode on
   RoadRunner and handed to AMICI is a different problem.
2. **The handoff is discovered late.** Three specs (PK, microglia, silk) failed only when first run on AMICI, with `AMICI:NaN sxdot[48]`,
   because `AB40_BrainISF` starts at exactly zero and feeds a rate law with exponent 0.8 (`n_oligo1_AB40`), whose derivative is infinite at zero.
   RoadRunner never sees this. The workaround (a 1e-12 seed) is per spec.
3. **AMICI cannot represent everything RoadRunner can.** Events whose time depends on a fitted parameter, parameters that are not model
   parameters (the APOE multipliers are applied by `Update_opt_parameters`, outside the model), and any future construct without a
   sensitivity (delays, algebraic rules) work on RoadRunner and not on AMICI. Nothing checks this when the model is written.
4. **Compile cost scales with model edits.** The model hash changes with any reaction edit. A first microglia compile took about 2 h
   for 12 models, silk about 2 h, aggregation under 30 minutes (its whole fit job took 36 min). `models_fit_mic` has roughly 120 hash directories, most stale.
5. **Numerical differences are real but small and undocumented.** AMICI vs RoadRunner differ by 0.04 nats on silk_appfull and 0.001 on
   aggregation_silk at the same point. Nothing states an expected tolerance or checks it.
6. **Version drift.** The repository's CI installs an unpinned `tellurium`; it failed on
   `test_a_loosened_scalar_scales_the_reference_not_the_failing_state` while the same file passed on the cluster (roadrunner 2.10.0, tellurium
   2.2.13.1). Cause *unverified* (a newer RoadRunner no longer re-derives the tolerance vector after `simulate`). AMICI is not
   obviously covered by CI at all (the CI log showed 4 skipped tests; which ones is *unverified*).

## 2. Catalogue

Priority: **A** correctness or reproducibility risk, **B** workflow cost or a trap, **C** polish.

### 2.1 Making the RoadRunner to AMICI handoff a gate (A)

| # | Item | Seen | Status |
|---|---|---|---|
| 1 | **Route-agreement check per spec.** One evaluation of every spec at its registry x0 on both routes, fixed sigma, flagging a difference above a stated tolerance (say 0.1 nat, tune from data). Would have caught the three NaN cases before any fit. Run it after every `Model_generate` that changes a tracked file. | PK, microglia, silk NaN; 0.04-nat offset | not built |
| 2 | **Preflight scan for the zero-state singularity.** Find states with initial value exactly 0 that feed a non-integer power; seed them automatically (or refuse with the name) instead of hand-written `analytic_gradient.seed` blocks per spec. A model-side floor on the base removes it for every route. | three specs hand-seeded | not built |
| 3 | **State the sigma mode in every result and panel.** Make `fixed` the default for new specs, or write `sigma_floor` into the result JSON. Panel titles already keep `n log sigma` (built, 1437e96). | the 22.8-nat offset | partly built |
| 4 | **Compatibility report on a spec**: which parameters are not model parameters, whether events depend on fitted parameters, which route(s) can run it. | APOE multipliers cannot go through AMICI | not built |

### 2.2 Provenance, drift and the dependency chain between fits (A)

The calibration is a chain: silk to aggregation to microglia, with APOE and Plasma hanging off aggregation. Nothing records that chain.

| # | Item | Seen | Status |
|---|---|---|---|
| 5 | **Upstream fingerprint on every fit.** Hash the values of the parameters a fit does not optimize but reads (`Human_registry` layers), store it in the fit and the profile directory, and warn on mismatch. The aggregation fit ran against stale silk values until the user caught it by reasoning, not by a tool; the APOE multipliers went stale when `k_c_AB42` doubled. | aggregation fit; APOE refit moved 0.636 to 0.517 and 0.382 to 0.272 | not built |
| 6 | **A `promote` command**: write a fit into the registry, refresh the fit cache, record the date, f and fit id. `tools/registry_update.py` does the first step with `--check` for drift (built, 11 tests); the cache refresh and provenance line are *not built*. | manual paste, refit, re-check | partly built |
| 7 | **One source of truth for starting values.** The Plasma spec had hard-coded x0 literals that had drifted from the registry (0.31268 / 0.036444 / 5.7548 against 0.31013 / 0.038069 / 5.6494), and the model's `parameters.csv` holds a third set (`k0_AB40_Plasma` 0.5503, marked "Fitted") that the registry silently overrides. A lint that compares CSV, registry and every spec literal would catch this. Plasma now reads the registry (uncommitted). | Plasma; CSV vs registry | partly built |
| 8 | **Objective regression baseline.** Store f at registry x0 per spec and route, and fail when a regenerated model moves it. The aggregation_silk objective at the same parameters shifted by 0.31 nats (-288.3178 on Oct 18, -288.005 on Oct 28) with the same floors stamp. The cause is *unverified*; the model edits of Oct 24 and Oct 26 are the suspects, and nothing tells you which edit moved which spec. `tests/spec_snapshot.json` is a place to start. | aggregation_silk | not built |
| 9 | **Versions in results**: git hash of PyAntiGen and the project, registry hash, roadrunner / amici / numpy versions, solver tolerances, floors stamp (exists for AMICI only). | CI failure from version drift | not built |

### 2.3 Fingerprints, caches and hidden state (A)

| # | Item | Seen | Status |
|---|---|---|---|
| 10 | **Fingerprints must not contain memory addresses.** The sigma-dictionary key contained a function repr, so every process hashed the same problem differently and each array task got its own profile directory (36 stray directories, a wasted array launch). Fixed with `_stable_key` (not yet committed). **Audit every `str()` / `repr()` that feeds a hash**, and add one test that hashes in two subprocesses with different `PYTHONHASHSEED`. | silk array 42488215 | built for sigma keys; audit not done |
| 11 | **Nondeterministic ordering across processes.** Replicate order differs between processes (set ordering), which changes accumulation order. Sort replicates, or fix `PYTHONHASHSEED` in the engine. | event lines in different order per task | not built |
| 12 | **`--refit` disables the cache write as well as the read.** A converged 50-minute fit existed only in the results snapshot (job 42508472). It should always write. | aggregation refit | not built |
| 13 | **The profile directory is keyed by the optimum to 6 significant figures.** A refit that lands slightly differently orphans a profile. Store profiles under a stable id and verify the anchor, as the checkpoint already half does. | worry, not seen to fail (silk resumed into the same directory) | not built |
| 14 | **Garbage collection** for `models_fit_*`, `profiles/` and `fits/` (stale hashes, partial directories, a pile of `optimization_progress_*.json/png`). | many directories | not built |
| 15 | **Model generation is a hidden step of every run.** `Model_run.py` rewrites the four model files in place at start. Lock ordering protected writers only; a reader (a task loading its model) saw a half-written file (`Global parameter ... is missing a value`, task 6, job 42462228). Also leaves tracked files dirty. Make writes atomic (temp file and rename), skip when the content is unchanged, and make generation an explicit step. `ESG_SKIP_GENERATE` and `hyak_model_generate.sbatch` are a workaround (built, committed in 4b06d51). | array failure | workaround built |

### 2.4 Fitting stage (B)

| # | Item | Seen | Status |
|---|---|---|---|
| 16 | **Nelder-Mead silently ignores `analytic_gradient`**; L-BFGS-B's first trial point landed on a corner of the box and an evaluation ran 30 minutes without returning. Bounded LM (`Fit_lm`, `method="LM"`) fixed it for fixed-sigma AMICI problems (built, tested; used for silk, aggregation, microglia). Document which method pairs with which route. | silk fit job 42473280 | built |
| 17 | **No multistart on AMICI.** LM runs one local fit from x0. The old pypesto route had Sobol triage and multistart; aggregation_silk's weak direction (FIM eigenvalue 0.22, condition 5e7) is where a second basin could hide. | aggregation FIM | not built |
| 18 | **Convergence criteria differ** between the pypesto LM (projected gradient 0.01) and `Fit_lm` / the profile (Newton decrement 1e-4 with a loose gradient limit 0.2). The "largest projected-gradient gain" warning printed at every profile start (0.079 nats on silk) overstates along ridges (the Newton decrement was 3.6e-5). One stated criterion, reported the same way. | silk stage warning | not built |
| 19 | **Method choice per route is not enforced**: a spec can ask for AMICI and Nelder-Mead; the engine only prints a message. | silk_appfull before `_amici` spec | not built |

### 2.5 Profile stage (B)

| # | Item | Seen | Status |
|---|---|---|---|
| 20 | **First walk step overshoots badly.** The first point often lands at dNLL 100 to 350 (silk `GSI_KI` down 116, `Vm_C99_Abeta_base` down 352, aggregation `k_c_AB42` down 283,582 after a four-decade jump). Costly points that are then bisected. A smaller first step from the FIM SD, or a backtracking walk, would save evaluations. | profile logs | not built |
| 21 | **Fill degenerates on a flat-then-cliff profile.** Linear interpolation toward dNLL 2.8e5 moved 7e-6 of the way per round; 12 h gave three rounds. `propose` now keeps its point inside 10 to 90 % of the bracket and bisects when the far end is over 100 times the level (built, 4 tests; uncommitted). `fill_rounds = 6` may still leave a coarse bracket on a four-decade span: *unverified*. | aggregation side 2_down | built |
| 22 | **The wall-time stop is mid-stage.** Two silk sides stopped inside fill after 11.6 h and needed a manual relaunch; a collect run also resumes unfinished sides *serially* on one CPU. Add an automatic requeue of unfinished sides and a collect that refuses to walk. | silk sides 5_down, 14_up | not built |
| 23 | **Physical hard limits** (`f_SP` is a fraction; real ceiling `1 - (Q_Leak + Q_Leak_CM)/Q_CSF`). The relax pass widens a bound by a decade and could pass 1. Proposal: `hard_lo` / `hard_hi` in the parameter definition, honoured by relax, extension and the pin grid, reported as "at the physical limit". Discussed, not built. | f_SP | not built |
| 24 | **Open sides are results.** `Vm_C99_Abeta_base` up and `GSI_KI` down are one-sided open along a ridge (accepted and explained). The report should say "open along ridge with GSI_KI" instead of "no crossing". | silk | partly built (side report) |
| 25 | **Wald step in `PROFILE_ONLY`.** It cost about 1.5 h per task and was unused by AMICI, and an empty profile fell back to a RoadRunner profile that forks one process per parameter (18 on a 1-CPU job). `PYANTIGEN_SKIP_WALD` and a guard in `Model_optimize` fix both (built, uncommitted guard). Presets that start different algorithms depending on the backend are a trap. | job 42481477 | built |
| 26 | **Profile non-identifiability by reparameterization.** FIM weak directions: silk `Vm_C99_Abeta_base`/`GSI_KI` (-0.70/+0.69), aggregation `k_denseplaque`/`k_c_AB42` (-0.76/+0.64). Detect them from the FIM and offer the product or ratio as the parameter. | both FIMs | not built |

### 2.6 Orchestration on Slurm (B)

| # | Item | Seen | Status |
|---|---|---|---|
| 27 | **The pipeline is a hand-run chain**: fit, FIM, probe two sides, check the hash, then the rest, then a collect. I did this four times. A driver that submits the DAG with dependencies and checks the probe's hash would remove the babysitting. | all four problems | not built |
| 28 | **`sbatch` timeouts created the job anyway** (twice: "Socket timed out", the job appeared about 20 s later). Retrying would have doubled the array. An idempotent submit (job name plus a lookup before resubmitting). | arrays 42488587 and 42488634 | not built |
| 29 | **Sizing is guesswork**: 1 CPU and 8 to 16 GB per side worked (0.9 GB RSS silk), compile wants 2 to 4 CPUs. Record peak RSS and wall time per stage and derive the request. | silk, microglia | not built |
| 30 | **Per-evaluation time is the budget**: silk 158 s, microglia 130 s, aggregation 30 s. Any fit or profile plan should print an estimate before submission. | all | not built |

### 2.7 Testing and CI (B)

| # | Item | Status |
|---|---|---|
| 31 | Pin `tellurium` / `libroadrunner` in CI or run a version matrix; investigate the failing tolerance-pinning test (section 1, item 6). | not built |
| 32 | A small AMICI test in CI (a two-species model with a zero initial state raised to 0.8), so the NaN path and the seed logic are covered somewhere other than on the cluster. | not built |
| 33 | A cross-process fingerprint test (item 10), a RoadRunner memory-growth test per evaluation (a 0.13 GB per evaluation leak, an SBML re-read, was found and fixed earlier), and an end-to-end test of `Fit_lm` wiring (the unit tests cover the solver, not `Optimize.py`). | not built |

### 2.8 Smaller things (C)

- The single-side message "INCOMPLETE: the clock stopped before every side finished" is printed even when the cause is not the clock (fim mode, a single task).
- `hyak_*.sbatch` files: about 50 older scripts point at the stale `PyAntiGen_fixedsigma` worktree, which still contains the SBML-reread memory leak.
- A handoff document per milestone is accumulating (about 30 `*.md` files in the project); an index would help.

## 3. Looming issues, ordered by how much they could cost

1. **Silent staleness down the chain** (items 5, 7, 8). The most damaging failure mode is not a crash, it is a fit made against yesterday's upstream values that looks fine. It already happened once.
2. **The handoff gate** (items 1, 2, 4). The more the model grows (the microglia clearance of capped fibrils, the next mechanism), the more likely a construct works in RoadRunner and not in AMICI, or works with a shifted objective.
3. **Hash and cache fragility** (items 10 to 13). One repr in a hash cost a full 36-task array. There are probably other places.
4. **Environment drift** (items 9, 31): unpinned roadrunner already broke CI.
5. **Weakly identified parameters** (item 26): reported as open or box-limited, correctly, but the downstream use of such parameters (simulation of new arms) needs the ridge to be carried, not a point value.

## 4. Suggested order if only a few things get done

1. Route-agreement gate (1) plus the cross-process fingerprint test (10, 33): cheapest, prevent the surprises seen this round.
2. Upstream-fingerprint and the `promote` command (5, 6): turns the manual chain into a checkable one.
3. Atomic model generation (15) and always-write fit cache (12).
4. Driver for the Slurm chain (27, 28).
5. Hard limits (23) and the first-step cap for the profile walk (20).
