"""Model-independent noise floor for cross-sectional / low-time-density
observables.

``compute_noise_floor`` is a pure function of the data itself -- it must never
see a model prediction or model parameter. See ``Engine.Optimize.
_sigma_floor_for`` and ``concentrated_nll`` for how the resulting sigma CAPS
(never replaces) a block's profiled sigma: sigma_used = min(sigma_hat,
sigma_floor).

Method: tricube-weighted local-linear regression ("LOESS"), with the
neighborhood width (span, as a fraction of the n nearest points) chosen by
leave-one-out cross-validation over a small grid of candidate spans. The
residual SD of the data around the CV-selected fit is the noise floor.

Unregularized LOOCV span selection has a real failure mode on data whose
neighboring points are even mildly correlated (shared digitization artifacts,
genuinely smooth underlying kinetics, population means over many subjects):
LOOCV keeps rewarding ever-tighter local windows with no interior minimum,
because "predict from the very next neighbor" looks great precisely because
neighbors ARE correlated -- the same phenomenon that motivated this whole
mechanism in the first place, now showing up as a bias in choosing it.
Concretely: the SILK MFL42/40/38 panels and the LY450139 CSF-total-Abeta
panel both drove span selection to the smallest possible window (checked down
to windows of 2 points) with LOOCV error still falling, meaning a naive
implementation would report a near-meaningless, near-interpolated sigma.
_MIN_LOCAL_NEIGHBORS is a first line of defense against that (never trust a
window under 8 points, full stop) -- but on the two panels above it, that
floor is the thing actively chosen, not a value LOOCV found independent of it.

``get_noise_floor``'s optional *shape_reference* is the real fix, one layer
above _MIN_LOCAL_NEIGHBORS: a caller with access to the mechanistic model's
own simulated curve can pass it in (aligned pointwise to x/y) to calibrate the
smallest span worth trusting against the model's SHAPE instead of the data's
own (possibly correlated) noise -- see _calibrate_min_span. This is the one
place a model prediction is allowed to influence the floor, and it is
carefully scoped so the rest of the "never see model output" contract still
holds where it matters: get_noise_floor consults shape_reference ONLY on a
cache miss (the very first call for that cache_key, i.e. whatever simulation
happens to be current at that point in a fit -- typically x0, "far off" and
all), then freezes the result. Every subsequent call -- meaning essentially
every one of the thousands of objective evaluations in a fit -- is a cache hit
that never touches shape_reference again, so the live per-iteration objective
stays exactly as fixed and side-effect-free as it always has.

Computing this is O(n^2) per candidate span -- trivial for the tens of points
these cross-sectional panels carry, but wasteful to redo on every one of the
thousands of objective evaluations in a fit. Callers should go through
``get_noise_floor``, which caches by data content, not through
``compute_noise_floor`` directly, inside anything that runs per-evaluation.
"""

from collections import namedtuple

import numpy as np

NoiseFloor = namedtuple("NoiseFloor", ["sigma", "span", "n", "x_fit", "y_fit"])

_DEFAULT_SPANS = (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9, 1.0)
_MIN_FLOOR_POINTS = 8      # below this, LOOCV span selection isn't trustworthy

# A local window with too few points is itself too unstable a variance
# estimate to let LOOCV chase, for the same reason _MIN_FLOOR_POINTS exists
# for the dataset as a whole -- a 2-3 point local-linear fit has essentially
# no residual degrees of freedom, so it can always "predict" the held-out
# point well by riding the correlation between adjacent points rather than by
# genuinely separating signal from noise. Flooring every window's point count
# here, not just the dataset's, is what turns the degeneracy documented above
# into a real interior minimum: every already-shipped floor (the six
# cross-sectional CSF/sAPP panels, MFL_sa/MFL_sb, ABtot_SP3_frac_baseline_neg)
# already picks a span whose window is well above this, so this floor changes
# none of them -- it only kicks in for the degenerate cases.
_MIN_LOCAL_NEIGHBORS = _MIN_FLOOR_POINTS

_N_DISPLAY_POINTS = 200    # resolution of the fitted curve kept for plotting

_FLOOR_CACHE = {}          # {(cache_key, x.tobytes(), y.tobytes(), spans): NoiseFloor|None}


def _local_linear_predict(x_train, y_train, x_query, span):
    """Tricube-weighted local-linear fit of (x_train, y_train), evaluated at
    each point in x_query. *span* is the fraction of x_train used as each
    query point's neighborhood, floored at _MIN_LOCAL_NEIGHBORS points."""
    n = len(x_train)
    k = min(max(_MIN_LOCAL_NEIGHBORS, int(np.ceil(span * n))), n)
    out = np.empty(len(x_query))
    for i, xq in enumerate(x_query):
        d = np.abs(x_train - xq)
        idx = np.argsort(d)[:k]
        dmax = d[idx].max()
        if dmax <= 0:
            dmax = 1e-12
        w = np.clip(1.0 - (d[idx] / dmax) ** 3, 0.0, None) ** 3
        X = np.column_stack([np.ones(k), x_train[idx] - xq])
        XtW = X.T * w
        try:
            beta = np.linalg.solve(XtW @ X + 1e-12 * np.eye(2), XtW @ y_train[idx])
        except np.linalg.LinAlgError:
            beta = np.array([np.average(y_train[idx], weights=w), 0.0])
        out[i] = beta[0]
    return out


def _loocv_mse(x, y, span):
    """Leave-one-out mean squared prediction error at this span: refit
    excluding point i (not just down-weighting it), predict at x[i]. The
    window is floored at _MIN_LOCAL_NEIGHBORS points, which is what keeps this
    from degenerating toward interpolation on correlated data (see module
    docstring)."""
    n = len(x)
    k = min(max(_MIN_LOCAL_NEIGHBORS, int(np.ceil(span * n))), n - 1)
    errs = np.empty(n)
    for i in range(n):
        xi = x[i]
        d = np.abs(x - xi)
        d[i] = np.inf  # exclude self from the neighborhood entirely
        idx = np.argsort(d)[:k]
        dmax = d[idx].max()
        if not np.isfinite(dmax) or dmax <= 0:
            dmax = 1e-12
        w = np.clip(1.0 - (d[idx] / dmax) ** 3, 0.0, None) ** 3
        X = np.column_stack([np.ones(k), x[idx] - xi])
        XtW = X.T * w
        try:
            beta = np.linalg.solve(XtW @ X + 1e-10 * np.eye(2), XtW @ y[idx])
        except np.linalg.LinAlgError:
            beta = np.array([np.average(y[idx], weights=w), 0.0])
        errs[i] = y[i] - beta[0]
    return float(np.mean(errs ** 2))


def _oscillation_length(diff):
    """Total variation of a difference/residual curve -- how much it wiggles.

    Deliberately not a literal geometric arc length (which would need to mix
    a time axis and a value axis of unrelated units into one Euclidean
    distance, forcing an arbitrary normalization). Total variation only cares
    about the value axis, which is fine here: every span compared in
    _calibrate_min_span shares the same x grid and y units, so it's an
    apples-to-apples "how wiggly" ranking, and it's offset-invariant -- a
    constant shift between the two curves being differenced doesn't change
    it, which is what makes this usable even when a simulation is
    quantitatively far from the data (see _calibrate_min_span).
    """
    diff = np.asarray(diff, dtype=float)
    if diff.size < 2:
        return 0.0
    return float(np.sum(np.abs(np.diff(diff))))


def _calibrate_min_span(x, y, y_sim_ref, spans):
    """One-time, shape-only calibration of the smallest span worth trusting.

    For each candidate span, fits LOESS to (x, y) and measures how much
    (LOESS - y_sim_ref) wiggles (_oscillation_length). A window tighter than
    the mechanistic model's own structure warrants chases noise the model
    doesn't produce, so the difference wiggles a lot; a window wider than that
    smooths away structure the model DOES have, so the difference picks up
    mismatch of its own instead. The result is a genuine interior minimum
    across span -- verified against real csf_abtot_percent_baseline and SILK
    MFL42 data (2026-09-15) even using a crude independent proxy (a degree-3
    polynomial) in place of a real simulation, so the mechanism doesn't
    depend on the reference being quantitatively accurate, only structurally
    plausible.

    Returns the span with the smallest oscillation length.
    """
    best_span, best_len = spans[0], np.inf
    for span in spans:
        fitted = _local_linear_predict(x, y, x, span)
        length = _oscillation_length(fitted - y_sim_ref)
        if length < best_len:
            best_span, best_len = span, length
    return best_span


def compute_noise_floor(x, y, spans=None):
    """Data-only noise floor for one observable's (x, y) pairs, or None if
    there are too few points to trust a span selection.

    Never call this with model output -- x/y must be the raw data columns.
    """
    spans = spans if spans is not None else _DEFAULT_SPANS
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < _MIN_FLOOR_POINTS:
        return None

    best_span, best_mse = spans[0], np.inf
    for span in spans:
        mse = _loocv_mse(x, y, span)
        if mse < best_mse:
            best_span, best_mse = span, mse

    fitted = _local_linear_predict(x, y, x, best_span)
    resid = y - fitted
    dof = max(n - 2, 1)
    sigma = max(float(np.sqrt(np.sum(resid ** 2) / dof)), 1e-12)

    x_grid = np.linspace(x.min(), x.max(), _N_DISPLAY_POINTS)
    y_grid = _local_linear_predict(x, y, x_grid, best_span)

    return NoiseFloor(sigma=sigma, span=float(best_span), n=n,
                       x_fit=x_grid, y_fit=y_grid)


def get_noise_floor(x, y, cache_key, spans=None, shape_reference=None):
    """``compute_noise_floor``, cached by data content and *cache_key* so a
    fit that calls this on every objective evaluation only pays for the LOOCV
    span search once.

    *cache_key* should identify the observable (e.g.
    ``(data_dict_key, data_column, time_column)``) -- the data content is
    hashed in as well so a changed CSV invalidates automatically rather than
    serving a stale floor.

    *shape_reference*, if given, must align pointwise with the RAW (un-
    finite-masked) *x*/*y* passed in -- typically a simulated prediction
    interpolated onto the data's own time points. It is consulted only when
    this call is a cache miss (see the module docstring for why that keeps
    this safe to call every evaluation): _calibrate_min_span uses it to
    restrict the candidate spans to those no tighter than the model's own
    shape supports, then the ordinary data-only LOOCV search runs over
    whatever survives that restriction.

    Earlier versions of this docstring claimed a shape reference could only
    ever remove options a purely data-driven search would have been tempted
    by, never move an already-clean pick. That turned out to be wrong:
    ABtot_SP3_frac_baseline_neg found a genuine-looking interior minimum
    (span=0.5) on data alone, but in a live fit the shape-calibrated version
    corrected it as an overfit anyway. That makes sense in hindsight -- an
    interior minimum in LOOCV-vs-DATA only means tightening further stops
    helping *predict held-out data points*; it says nothing about whether the
    window is still narrow enough to be riding the data's own correlated
    noise, which is the exact failure mode this whole mechanism exists to
    catch. A shape reference free of that noise can and does override a
    data-only pick, not just a degenerate one.
    """
    spans = spans if spans is not None else _DEFAULT_SPANS
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_m, y_m = x[mask], y[mask]
    full_key = (cache_key, x_m.tobytes(), y_m.tobytes(), tuple(spans))
    if full_key in _FLOOR_CACHE:
        return _FLOOR_CACHE[full_key]

    use_spans = spans
    if shape_reference is not None and x_m.size >= _MIN_FLOOR_POINTS:
        ref = np.asarray(shape_reference, dtype=float)
        if ref.shape == mask.shape:
            ref_m = ref[mask]
            ref_valid = np.isfinite(ref_m)
            if ref_valid.sum() >= _MIN_LOCAL_NEIGHBORS:
                min_span = _calibrate_min_span(
                    x_m[ref_valid], y_m[ref_valid], ref_m[ref_valid], spans)
                use_spans = tuple(s for s in spans if s >= min_span) or (spans[-1],)

    result = compute_noise_floor(x_m, y_m, spans=use_spans)
    _FLOOR_CACHE[full_key] = result
    return result


def clear_cache():
    """Forget every cached floor, forcing the next call for each observable
    to recalibrate from scratch.

    Meant to be called once, deliberately, right before a final re-evaluation
    at a converged optimum -- see ``Engine.Optimize.run_optimization_from_groups``.
    The live per-iteration objective calibrates each floor from whatever x0
    happens to produce on the FIRST evaluation of a run, which is typically a
    poor fit; the shape reference it compares against is correspondingly poor.
    Clearing the cache just before the optimum is re-simulated for final
    reporting lets every floor recalibrate against the converged (good) fit
    instead, which is both a better reference and the one that makes separate
    runs' final numbers comparable to each other -- two runs converging to the
    same region calibrate against similar curves, unlike two runs whose only
    connection was an arbitrary, unrelated x0 apiece.

    Never call this mid-optimization: every floored block's sigma would
    change on whatever evaluation follows, discontinuously, which is exactly
    the per-iteration instability the one-time-calibration design exists to
    avoid.
    """
    _FLOOR_CACHE.clear()


def export_cache():
    """A plain-dict snapshot of every calibrated floor, for shipping to a
    worker process.

    ``_FLOOR_CACHE`` is per-process, in-memory state -- a ``ParallelEvaluator``
    worker (``Engine.Evaluator``, ``mp_context="spawn"``) re-imports this
    module from scratch and gets its own empty cache, with nothing carrying
    the parent's already-calibrated floors across. Left alone, each worker
    would recalibrate independently against whatever parameter vector it
    happens to be handed FIRST -- an arbitrary profile-grid point or Sobol
    sample, not the converged optimum -- so two grid points on the same
    profile curve could end up scored against two different sigma floors for
    the same observable, purely because different workers picked them up
    first. ``seed_cache`` is the other half: call this in the parent after
    the calibration you want workers to share (e.g. right after
    ``clear_cache()`` + the post-optimum re-evaluation), thread the result
    through as a field on ``Engine.Evaluator.EvalSpec`` the way
    ``fixed_sigmas`` already is, and call ``seed_cache`` with it in
    ``_init_worker`` before any task runs.

    A plain ``dict`` copy, not the live object -- cloudpickle serializes it by
    value, so mutating the original after export never reaches a worker that
    already received a snapshot.
    """
    return dict(_FLOOR_CACHE)


def seed_cache(snapshot):
    """Pre-populate this process's cache from ``export_cache()``'s output.

    Uses ``update``, not replace: a key already present locally (there
    shouldn't be one, in the fresh-worker case this exists for) is left
    alone, and a key the snapshot doesn't cover still falls through to
    ordinary cache-miss calibration rather than raising -- this only ever
    adds known-good answers, it never removes the ability to compute new
    ones.
    """
    if snapshot:
        _FLOOR_CACHE.update(snapshot)
