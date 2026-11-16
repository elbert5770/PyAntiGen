# Prediction bands from a stored profile: `--predict ensemble`

A fit gives curves; a profile likelihood gives each parameter's confidence interval. `--predict ensemble` turns the second into a band on the first,
without a new optimization: every point of a stored profile already carries the full parameter vector that was fitted at that point.

```
Model_run.py --simulate KEY --predict ensemble --predict-stage <stage> ...
```

`--predict` is `none` by default and changes nothing then. `--predict exact` (a prediction profile likelihood that re-optimizes with the prediction held
at each value) is reserved and not implemented.

## Method

For a model output z = g(theta) at one time, the pointwise 95% interval of the prediction profile likelihood is the range of z over the parameter
vectors whose negative log-likelihood is within chi2(1, 0.95) / 2 = 1.9207 of its minimum (Kreutz, Raue and Timmer, BMC Systems Biology 2012;
check the reference before you cite it). The ensemble approximates that set by the **members**: the optimum and every profile point with
dnll <= 1.9207. The band is the minimum and maximum of the model curve over the members, at each time.

What the band is, and is not:

- **An inner approximation.** The members lie on the one-dimensional profile paths, not throughout the set, so the exact band can be wider where a
  prediction is extreme in a direction between them. `band` prints how the band grows as members are added; a band that has stopped growing shows that
  the profile paths are exhausted, not that the set is covered. The delta-method band (`fdmembers`, `delta`) is a cross-check: symmetric, quadratic,
  built from the Fisher information.
- **Pointwise**, each time on its own, not a simultaneous band over time or over curves.
- **Conditional** on the model structure, on the noise the likelihood declared (the floors, if the profile used fixed sigma floors), and on every
  parameter that was held rather than fitted. It is a band for the model curve, not a prediction interval for new measurements.
- **As independent as the residuals were assumed to be.** Strongly autocorrelated residuals make the likelihood, and so the set, too tight.

## Which solver

The members come from an **AMICI** profile (`--backend amici` on the profile stage). Their curves are computed on **RoadRunner**, the solver that
`--simulate` draws figures with. That is deliberate: the band is built the way the line it surrounds is built, every line a plot function draws is
available (reaction rates and derived quantities included, which AMICI does not return), and no AMICI install is needed to draw a figure.
The cost is that the members were selected under the AMICI likelihood and are drawn with RoadRunner's integration; the two agree closely for a
model the fit trusts, but the difference is not part of the band. Measure it once for your model with `--simulate KEY --backend amici --compare` at the optimum.

A member's parameter vector is applied exactly as the fit applied it (`pyantigen.engine.Parameter_replay`): the model is reset, the replicate's
`Update_parameters` runs, the fitted values go over it, then the replicate's `parameter_hooks` and `Update_opt_parameters`. A fitted name that only a
hook understands is therefore fine. A name that changes **nothing** in the model is an error, because the fit's own setter ignores unknown names and a
member would silently equal the optimum.

## Stages

Options carry the prefix `--predict-`. Lists use `:` or `;`, because a `,` is split by job systems that pass options through an environment list.

| Stage | Needs | What it does |
|---|---|---|
| `members` | `--predict-store`, `--predict-out` | reads the profile store (`meta.json` and the `<kind>_<k>_<up\|down>` files), writes the members file; the optimum is member 0 |
| `run` | `--simulate`, `--predict-members`, `--predict-out` | simulates KEY once per member and records the lines its plot function draws as `member_<id>.npz`; `--predict-task I --predict-ntasks N` take members I, I+N, ...; a member whose file exists is skipped, so a requeued task continues; `--predict-only 0:1` is a trial |
| `band` | `--predict-out`, `--predict-band` | the envelope over the members' lines, the growth of the band with members, `<band>.notes.json` (the range of any number the plot function annotates) |
| `plot` | `--simulate`, `--predict-band` | simulates KEY once and draws the bands behind its lines, in a private results tree |
| `fdmembers`, `delta` | store; fd files, `--predict-fim`, `--predict-band` | the delta-method cross-check |
| `opensides` | `--predict-members`, `--predict-out`, `--predict-band` | on a profile side that ends open, do the curves still move toward its end? |

`--predict-store` is the profile store of the AMICI profile stage (the folder with `meta.json`); `--predict-fim` is its `fim/fim.npz`.

### The lines are the plot function's lines

Nothing in a project has to describe its outputs. `run` wraps `Axes.plot`, `Axes.fill_between` and `Axes.annotate` while the experiment's plot function
(`EXPERIMENT_dict[KEY]["plot"]`) runs, and keys each line by figure, axes and line: `M0.2` is the third line of the first axes of the first figure; `S`
is the second figure, `F2_`, `F3_` ... the later ones; `.f<n>` is the upper boundary of a stacked area, `.a<n>` an annotation. `plot` draws a
`fill_between` behind each line that has a band. Consequences:

- The key depends on the order a plot function draws, so changing the plot function invalidates band files made before.
- A line that does not change with the parameters (data, fixed physiology) gets no band; a line of fewer than 20 points is a reference mark and gets none.
- **The band is about member 0, the stored optimum.** `plot` reports every drawn line that lies further than 1e-6 of its range from that curve
  (a `WARNING`): the project's own parameter values are then not the profile's optimum, or the plot function changed.

## Project hooks (optional)

If `Modules/Predict_hooks.py` exists it may define

- `before_member(member) -> dict of extra arrays or None`: run before a member is simulated, for a project that must do something per member (refit a
  nuisance factor, say). The arrays are saved into the member's file.
- `prepare_tree(root)`: called with the private results tree before anything is simulated in it.

`member` is the entry of the members file: `id`, `param`, `side`, `x_fixed`, `dnll`, `verdict`, `values` (linear parameter values).

## A run, with a job array

Nothing here is specific to a site; adapt the directives to yours. Keep the request to what a member needs (one simulation, one thread).

```bash
# 1. once: the members
python Model_run.py --predict ensemble --predict-stage members \
    --predict-store results/<MODEL>/profiles/<run>/amici --predict-out members.json

# 2. one task per share of the members
sbatch --array=0-15 member_run.sbatch          # below

# 3. when the array has finished
python Model_run.py --predict ensemble --predict-stage band --predict-out member_files --predict-band band.npz
python Model_run.py --simulate KEY --predict ensemble --predict-stage plot --predict-band band.npz
```

```bash
#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=04:00:00
#SBATCH --requeue                 # a requeued task skips the members that exist
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
python Model_run.py --simulate KEY --predict ensemble --predict-stage run \
    --predict-members members.json --predict-out member_files \
    --predict-task ${SLURM_ARRAY_TASK_ID:-0} --predict-ntasks ${SLURM_ARRAY_TASK_COUNT:-1}
```

Time per member is one `--simulate`; measure one with `--predict-only 0` before sizing the array.
