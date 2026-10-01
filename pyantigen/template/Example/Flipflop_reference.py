"""
Ground-truth reference profiles for the flip-flop example (Example4/Example5).

This script recomputes the true profile likelihood of k_A_to_B, k_B_to_C and
SF against data/Flipflop.csv using the CLOSED-FORM solution of the A -> B -> C
chain and scipy only — no RoadRunner, no framework loss code. It follows the
same conventions the framework's diagnostics use:

  * observables are log10-transformed with a 1e-12 floor;
  * per-observable sigmas are estimated by MLE from the residuals at the
    anchor optimum and then frozen (the framework's fixed_sigmas), with the
    same dof correction sqrt(SSR / max(1, n - k/n_observables));
  * dNLL is the plain unweighted sum over observables, relative to the anchor
    optimum, compared against chi2(1, 0.95)/2 = 1.9207.

Because the model here is exact, any disagreement between these curves and the
framework's profile output (results/Example/*_profile_likelihood.png and the
profile_traces in the optimization results CSV) is an error in the profile
machinery — the walker, the nuisance re-optimization, the sigma freezing, or
the dNLL bookkeeping — not in the model.

Run from Projects/Example/:

    python Flipflop_reference.py                 # anchored at the global mode (Example4)
    python Flipflop_reference.py --anchor swapped  # anchored at the wrong mode (Example5)

Prints the swap-mode dNLL gap, the inter-mode barrier height, and the correct
95% confidence SET for each parameter (which may be a union of disjoint
intervals — something a first-crossing CI extractor cannot represent).
"""
import argparse
import os

import numpy as np
import pandas as pd

THRESHOLD = 1.9207  # chi2(df=1, p=0.95) / 2
FLOOR = 1e-12
K_PARAMS = 3
PARAM_NAMES = ["k_A_to_B", "k_B_to_C", "SF"]
BOUNDS = [(0.005, 5.0), (0.005, 5.0), (0.05, 50.0)]

TREATMENTS = {
    "Early": {"dose": 10.0, "delay": 5.0},
    "Late":  {"dose": 5.0, "delay": 10.0},
}
# observables per loss config: Early has logB+logA, Late has logB only
N_OBS_CFG = {"Flipflop_Early": 2, "Flipflop_Early_A": 2, "Flipflop_Late": 1}


def _find_data_file():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    try:
        import AntiGen_paths
        candidates.append(os.path.join(AntiGen_paths.REPO_ROOT, "data", "Flipflop.csv"))
    except Exception:
        pass
    candidates.append(os.path.normpath(os.path.join(here, "..", "..", "data", "Flipflop.csv")))
    candidates.append(os.path.normpath(os.path.join(here, "..", "data", "Flipflop.csv")))
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError("Flipflop.csv not found; run data/make_flipflop_data.py first.")


def chain_A(t, dose, delay, k1):
    tau = np.maximum(np.asarray(t, dtype=float) - delay, 0.0)
    return np.where(tau > 0, dose * np.exp(-k1 * tau), 0.0)


def chain_B(t, dose, delay, k1, k2):
    tau = np.maximum(np.asarray(t, dtype=float) - delay, 0.0)
    if abs(k1 - k2) < 1e-10:
        core = dose * k1 * tau * np.exp(-k1 * tau)
    else:
        core = dose * k1 * (np.exp(-k1 * tau) - np.exp(-k2 * tau)) / (k2 - k1)
    return np.where(tau > 0, core, 0.0)


def log10f(x):
    return np.log10(np.maximum(x, FLOOR))


def load_data():
    df = pd.read_csv(_find_data_file())
    data = {}
    for name in TREATMENTS:
        sub = df[df["Treatment"] == name]
        t = np.concatenate([sub["time"].to_numpy()] * 3)
        y = np.concatenate([sub[f"logB{i}"].to_numpy() for i in (1, 2, 3)])
        data[f"Flipflop_{name}"] = (t, y)
    early_A = df[(df["Treatment"] == "Early") & df["logA"].notna()]
    data["Flipflop_Early_A"] = (early_A["time"].to_numpy(), early_A["logA"].to_numpy())
    return data


def ssr_terms(p, data):
    """{observable_key: (SSR, n)} at p = (k_A_to_B, k_B_to_C, SF)."""
    k1, k2, sf = p
    out = {}
    for name, tr in TREATMENTS.items():
        t, y = data[f"Flipflop_{name}"]
        pred = log10f(sf * chain_B(t, tr["dose"], tr["delay"], k1, k2))
        out[f"Flipflop_{name}"] = (float(np.sum((y - pred) ** 2)), len(y))
    t, y = data["Flipflop_Early_A"]
    pred = log10f(chain_A(t, TREATMENTS["Early"]["dose"], TREATMENTS["Early"]["delay"], k1))
    out["Flipflop_Early_A"] = (float(np.sum((y - pred) ** 2)), len(y))
    return out


def frozen_sigmas_at(p, data):
    return {key: max(np.sqrt(ssr / max(1, n - K_PARAMS / N_OBS_CFG[key])), 1e-6)
            for key, (ssr, n) in ssr_terms(p, data).items()}


def nll(p, data, sigmas):
    """Frozen-sigma NLL including the Gaussian constant terms (they cancel in
    dNLL, but keeping them matches the framework's -sum(norm.logpdf(...))."""
    total = 0.0
    for key, (ssr, n) in ssr_terms(p, data).items():
        s = sigmas[key]
        total += ssr / (2.0 * s ** 2) + n * np.log(s * np.sqrt(2.0 * np.pi))
    return total


# sigma values declared in Flipflop_loss_config (shape the fit, not inference)
FIT_SIGMAS = {"Flipflop_Early": 0.05, "Flipflop_Late": 0.05, "Flipflop_Early_A": 0.75}


def fit_objective(p, data):
    """The framework's FITTING objective, which is not the joint NLL: each
    observable's chi-square term is divided by its own point count (the
    per-observable mean the loss functions compute), using the fixed sigmas
    declared in the loss config. With 45 logB points against 4 logA points
    this upweights the noisy logA data ~11x relative to the likelihood, so
    the fit optimum lands measurably away from the NLL optimum — which is
    why the framework's (correct) profiles dip below zero even in Example4.
    Reproducing that anchor offset here is what lets the reference curves be
    compared with the framework's point-for-point."""
    return sum(ssr / (2.0 * FIT_SIGMAS[key] ** 2 * n)
               for key, (ssr, n) in ssr_terms(p, data).items())


def _minimize(obj, x0_log, bounds_log):
    from scipy.optimize import minimize
    res = minimize(obj, x0_log, method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-11, "fatol": 1e-13})
    x = np.clip(res.x, [b[0] for b in bounds_log], [b[1] for b in bounds_log])
    return x, obj(x)


def fit_full(x0, data, sigmas):
    bounds_log = [np.log10(b) for b in BOUNDS]
    obj = lambda q: nll(10.0 ** np.asarray(q), data, sigmas)
    q, f = _minimize(obj, np.log10(x0), bounds_log)
    return 10.0 ** q, f


def profile_one(idx, grid, data, sigmas, mode_a, mode_b):
    """True profile of parameter idx over *grid* (linear values).

    Nuisance parameters re-optimized by Nelder-Mead in log10 space with three
    starts per grid point — continuation from the previous point, plus the
    nuisance components of each known mode — so the profile finds the better
    basin on both sides of the barrier. This multi-start is what makes these
    curves 'reference': a single-start continuation walker can lag behind the
    basin switch and overestimate the barrier.
    """
    nuis_idx = [j for j in range(3) if j != idx]
    bounds_log = [np.log10(BOUNDS[j]) for j in nuis_idx]

    def obj_at(v_fixed):
        def obj(q_nuis):
            p = np.empty(3)
            p[idx] = v_fixed
            for j, q in zip(nuis_idx, q_nuis):
                p[j] = 10.0 ** q
            return nll(p, data, sigmas)
        return obj

    prof = np.empty(len(grid))
    prev = None
    for i, v in enumerate(grid):
        starts = []
        if prev is not None:
            starts.append(prev)
        for mode in (mode_a, mode_b):
            starts.append(np.log10([mode[j] for j in nuis_idx]))
        best_f, best_q = np.inf, None
        for s in starts:
            q, f = _minimize(obj_at(v), np.asarray(s, dtype=float), bounds_log)
            if f < best_f:
                best_f, best_q = f, q
        prof[i] = best_f
        prev = best_q
    return prof


def confidence_set(grid, dnll, threshold=THRESHOLD):
    """All intervals where the profile is below threshold (linear interp at edges)."""
    below = dnll < threshold
    intervals = []
    i = 0
    n = len(grid)
    while i < n:
        if below[i]:
            j = i
            while j + 1 < n and below[j + 1]:
                j += 1
            lo = grid[i]
            if i > 0:
                y0, y1 = dnll[i - 1], dnll[i]
                lo = grid[i - 1] + (threshold - y0) * (grid[i] - grid[i - 1]) / (y1 - y0)
            hi = grid[j]
            if j + 1 < n:
                y0, y1 = dnll[j], dnll[j + 1]
                hi = grid[j] + (threshold - y0) * (grid[j + 1] - grid[j]) / (y1 - y0)
            intervals.append((lo, hi))
            i = j + 1
        else:
            i += 1
    return intervals


def compare_with_framework(json_path, rows, p_anchor, p_other, gap):
    """Score the framework's profile_traces (from the per-run results JSON)
    against the reference curves. Prints one verdict per parameter:

      * max |dNLL_framework - dNLL_reference| over the framework's trace, with
        the reference interpolated in log-parameter space (only where the
        reference is below 6 — beyond that both curves are off any CI scale);
      * whether the trace ever reaches the other mode's neighbourhood, and if
        so the dNLL it reports there versus the known gap. A trace that never
        gets there means the walker stopped at the first threshold crossing
        and the reported CI silently assumes unimodality.
    """
    import json as _json
    with open(json_path) as f:
        snap = _json.load(f)
    traces = snap.get("profile_traces")
    if not traces:
        print(f"\n[compare] no profile_traces found in {json_path}")
        return
    ci = snap.get("profile_ci95", {})
    print(f"\n[compare] framework traces from {os.path.basename(json_path)}")
    for idx, name in enumerate(PARAM_NAMES):
        tr = traces.get(name)
        if not tr:
            print(f"  {name}: no trace")
            continue
        x = np.asarray(tr["x"], dtype=float)
        y = np.asarray(tr["y"], dtype=float)
        grid, dnll = rows[name]
        ref = np.interp(np.log10(x), np.log10(grid), dnll)
        band = ref < 6.0
        max_err = float(np.max(np.abs(y[band] - ref[band]))) if band.any() else np.nan
        other = p_other[idx]
        near_other = np.abs(np.log10(x) - np.log10(other)) < 0.15
        print(f"  {name}: {len(x)} pts, span [{x.min():.4g}, {x.max():.4g}], "
              f"max |dNLL err| (ref<6) = {max_err:.3g}")
        if near_other.any():
            print(f"    reaches other mode ({other:.4g}): reported dNLL "
                  f"{float(np.min(y[near_other])):+.3g} vs expected {gap:+.3g}")
        else:
            print(f"    *** never reaches the other mode at {other:.4g} - "
                  f"CI assumes unimodality ***")
        if name in ci:
            print(f"    framework CI95: {ci[name]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", choices=["true", "swapped"], default="true",
                    help="which mode to freeze sigmas at and anchor dNLL=0 to "
                         "(matches Example4 / Example5 respectively)")
    ap.add_argument("--n-grid", type=int, default=81)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--compare", type=str, default=None, metavar="RESULTS_JSON",
                    help="per-run optimization results JSON (contains "
                         "profile_traces) to score against the reference")
    args = ap.parse_args()

    data = load_data()

    # Replicate the framework pipeline end-to-end:
    # 1. minimize the FITTING objective (per-observable means, config sigmas)
    #    from Example4's / Example5's x0 — this is where the framework's fit
    #    lands, and it is NOT the NLL optimum;
    bounds_log = [np.log10(b) for b in BOUNDS]
    def fit_obj_full(x0):
        obj = lambda q: fit_objective(10.0 ** np.asarray(q), data)
        q, f = _minimize(obj, np.log10(x0), bounds_log)
        return 10.0 ** q, f

    p_fit_true, _ = fit_obj_full([0.3, 0.08, 1.5])    # Example4 x0
    p_fit_swap, _ = fit_obj_full([0.06, 0.4, 7.0])    # Example5 x0
    p_fit, p_fit_other = ((p_fit_true, p_fit_swap) if args.anchor == "true"
                          else (p_fit_swap, p_fit_true))

    # 2. freeze per-observable MLE sigmas at the fit point (fixed_sigmas);
    sigmas = frozen_sigmas_at(p_fit, data)
    # 3. anchor dNLL = 0 at the fit point's inference NLL — exactly what the
    #    framework's diagnostics do. The NLL minima of the two basins under
    #    these frozen sigmas are then located for reference markers.
    nll_anchor = nll(p_fit, data, sigmas)
    p_anchor, nll_min_anchor = fit_full(p_fit, data, sigmas)
    p_other, nll_other = fit_full(p_fit_other, data, sigmas)

    print(f"fit point (anchor={args.anchor}, fitting objective): "
          + "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, p_fit)))
    print(f"NLL minimum, same basin:  "
          + "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, p_anchor)))
    print(f"NLL minimum, other basin: "
          + "  ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, p_other)))
    print("frozen sigmas: " + ", ".join(f"{k}={v:.4f}" for k, v in sigmas.items()))
    print(f"anchor offset (fit objective vs NLL optimum): "
          f"{nll_min_anchor - nll_anchor:+.4f}  <- profiles must dip to this")
    print(f"dNLL at other-basin minimum = {nll_other - nll_anchor:+.4f}   "
          f"(95% threshold {THRESHOLD})")

    rows = {}
    fig = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    except Exception:
        axes = [None] * 3

    for idx, (name, bounds) in enumerate(zip(PARAM_NAMES, BOUNDS)):
        grid = np.logspace(np.log10(bounds[0]), np.log10(bounds[1]), args.n_grid)
        # make sure both mode values are exactly on the grid
        grid = np.unique(np.concatenate([grid, [p_anchor[idx], p_other[idx]]]))
        prof = profile_one(idx, grid, data, sigmas, p_anchor, p_other)
        dnll = prof - nll_anchor
        rows[name] = (grid, dnll)

        cs = confidence_set(grid, dnll)
        in_band = dnll < THRESHOLD
        barrier = np.nan
        i_a = int(np.argmin(np.abs(grid - p_anchor[idx])))
        i_o = int(np.argmin(np.abs(grid - p_other[idx])))
        lo_i, hi_i = sorted((i_a, i_o))
        if hi_i > lo_i:
            barrier = float(np.max(dnll[lo_i:hi_i + 1]))
        cs_str = " U ".join(f"[{a:.4g}, {b:.4g}]" for a, b in cs) or "(empty)"
        print(f"\n{name}:")
        print(f"  dNLL at anchor {p_anchor[idx]:.4g}: {dnll[i_a]:+.4g}   "
              f"at other mode {p_other[idx]:.4g}: {dnll[i_o]:+.4g}")
        print(f"  inter-mode barrier height: {barrier:.4g}")
        print(f"  correct 95% confidence set: {cs_str}"
              + ("   <-- DISJOINT" if len(cs) > 1 else ""))

        if axes[idx] is not None:
            ax = axes[idx]
            ax.semilogx(grid, dnll, "-", lw=1.5)
            ax.axhline(THRESHOLD, color="gray", ls=":", label="95% threshold")
            ax.axhline(0, color="gray", lw=0.5)
            ax.axvline(p_anchor[idx], color="red", ls="--", alpha=0.6, label="anchor mode")
            ax.axvline(p_other[idx], color="green", ls="--", alpha=0.6, label="other mode")
            ax.set_ylim(min(-3.5, np.nanmin(dnll) - 0.5), 8)
            ax.set_xlabel(name)
            ax.set_ylabel("dNLL")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    out_dir = args.out_dir
    if out_dir is None:
        try:
            import AntiGen_paths
            out_dir = os.path.join(AntiGen_paths.REPO_ROOT, "results", "Example")
        except Exception:
            out_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    max_len = max(len(g) for g, _ in rows.values())
    table = {}
    for name, (grid, dnll) in rows.items():
        pad = max_len - len(grid)
        table[f"{name}_value"] = np.concatenate([grid, [np.nan] * pad])
        table[f"{name}_dnll"] = np.concatenate([dnll, [np.nan] * pad])
    csv_path = os.path.join(out_dir, f"Flipflop_reference_profiles_{args.anchor}.csv")
    pd.DataFrame(table).to_csv(csv_path, index=False, float_format="%.6g")
    print(f"\nReference profiles written to {csv_path}")

    if args.compare:
        compare_with_framework(args.compare, rows, p_anchor, p_other,
                               nll_other - nll_anchor)

    if fig is not None:
        fig.suptitle(f"Flip-flop reference profiles (closed form, anchor={args.anchor})")
        fig.tight_layout()
        png_path = os.path.join(out_dir, f"Flipflop_reference_profiles_{args.anchor}.png")
        fig.savefig(png_path, dpi=130, bbox_inches="tight")
        print(f"Reference plot written to {png_path}")


if __name__ == "__main__":
    main()
