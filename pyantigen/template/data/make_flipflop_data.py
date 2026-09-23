"""
Generate the synthetic Flipflop dataset (Flipflop.csv) used by Example4/Example5.

The model is the two-step chain A -> B -> C (rates k_A_to_B, k_B_to_C) dosed by
an event (A = dose at t = delay), observed through the scaled output
predicted_B = SF * B_Comp1 / V_Comp1 on a log10 scale. Because

    B(tau) = dose * k1 * (exp(-k1 tau) - exp(-k2 tau)) / (k2 - k1)

swapping (k1, k2) -> (k2, k1) rescales B by k2/k1, and a fitted SF absorbs that
rescaling exactly: (k1, k2, SF) and (k2, k1, SF*k1/k2) produce *identical*
predicted_B trajectories. That is the classic pharmacokinetic "flip-flop"
ambiguity, and it makes the likelihood exactly bimodal in these parameters.

To turn the exact symmetry into a *known, finite* likelihood gap between the
two modes, the Early treatment also carries a handful of very noisy direct
observations of predicted_A = A_Comp1 / V_Comp1 (A depends only on k1, so it
breaks the swap). sigma_logA is tuned so the swapped mode sits near the 95%
chi-square threshold (dNLL = 1.9207): high enough to make the true confidence
region interesting, low enough that a mis-scaled dNLL moves it across the
threshold and visibly changes the reported CI.

The seed is fixed so the committed Flipflop.csv is reproducible. After writing
the CSV the script refits both modes with scipy against the closed-form model
(no RoadRunner involved) and prints the *realized* swap-mode dNLL gap under the
same frozen-sigma convention the framework's diagnostics use, so the expected
gap is recorded next to the data it belongs to.
"""
import argparse

import numpy as np
import pandas as pd

# Ground truth
K1_TRUE, K2_TRUE, SF_TRUE = 0.35, 0.07, 1.6
SIGMA_LOGB = 0.05   # dex, noise on log10(SF*B)
SIGMA_LOGA = 0.75   # dex, noise on log10(A); tuned to put the swap mode near dNLL ~ 2
N_B_REPS = 3

TREATMENTS = {
    "Early": {"dose": 10.0, "delay": 5.0,
              "t_B": np.arange(6.0, 48.1, 3.0), "t_A": np.array([6.0, 9.0, 12.0, 15.0])},
    "Late":  {"dose": 5.0, "delay": 10.0,
              "t_B": np.arange(12.0, 48.1, 3.0), "t_A": np.array([])},
}

FLOOR = 1e-12  # same floor the loss-config expressions use


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


def generate(seed):
    rng = np.random.default_rng(seed)
    rows = []
    for name, tr in TREATMENTS.items():
        logB_true = log10f(SF_TRUE * chain_B(tr["t_B"], tr["dose"], tr["delay"], K1_TRUE, K2_TRUE))
        logB_reps = logB_true[:, None] + rng.normal(0.0, SIGMA_LOGB, size=(len(logB_true), N_B_REPS))
        logA = {}
        if len(tr["t_A"]):
            logA_true = log10f(chain_A(tr["t_A"], tr["dose"], tr["delay"], K1_TRUE))
            for t, v in zip(tr["t_A"], logA_true + rng.normal(0.0, SIGMA_LOGA, size=len(tr["t_A"]))):
                logA[t] = v
        for i, t in enumerate(tr["t_B"]):
            rows.append({
                "Treatment": name, "time": t,
                "logB1": logB_reps[i, 0], "logB2": logB_reps[i, 1], "logB3": logB_reps[i, 2],
                "logA": logA.get(t, np.nan),
            })
    return pd.DataFrame(rows)


# --- realized-gap check (framework NLL convention, closed-form model) --------

def _stack_data(df):
    """Return per-treatment stacked logB data and Early logA data."""
    out = {}
    for name, tr in TREATMENTS.items():
        sub = df[df["Treatment"] == name]
        t = np.concatenate([sub["time"].to_numpy()] * N_B_REPS)
        y = np.concatenate([sub[f"logB{i+1}"].to_numpy() for i in range(N_B_REPS)])
        out[name] = (t, y)
    early = df[(df["Treatment"] == "Early") & df["logA"].notna()]
    out["Early_A"] = (early["time"].to_numpy(), early["logA"].to_numpy())
    return out


def _ssr_terms(p, data):
    """Per-observable (SSR, n) at parameters p = (k1, k2, SF)."""
    k1, k2, sf = p
    terms = {}
    for name, tr in TREATMENTS.items():
        t, y = data[name]
        pred = log10f(sf * chain_B(t, tr["dose"], tr["delay"], k1, k2))
        terms[name] = (float(np.sum((y - pred) ** 2)), len(y))
    t, y = data["Early_A"]
    pred = log10f(chain_A(t, TREATMENTS["Early"]["dose"], TREATMENTS["Early"]["delay"], k1))
    terms["Early_A"] = (float(np.sum((y - pred) ** 2)), len(y))
    return terms


def _weighted_nll(p, data, sigmas):
    terms = _ssr_terms(p, data)
    return sum(ssr / (2.0 * sigmas[k] ** 2) for k, (ssr, n) in terms.items())


def frozen_sigmas_at(p, data, k_params=3):
    """Framework convention: sigma = sqrt(SSR / max(1, n - k/n_observables)),
    where n_observables counts observables in the same loss config (Early has
    logB+logA -> 2, Late has logB -> 1)."""
    terms = _ssr_terms(p, data)
    n_obs_cfg = {"Early": 2, "Early_A": 2, "Late": 1}
    return {k: max(np.sqrt(ssr / max(1, n - k_params / n_obs_cfg[k])), 1e-6)
            for k, (ssr, n) in terms.items()}


def fit_mode(x0, data, sigmas):
    from scipy.optimize import minimize
    obj = lambda q: _weighted_nll(10.0 ** q, data, sigmas)
    res = minimize(obj, np.log10(x0), method="Nelder-Mead",
                   options={"maxiter": 4000, "xatol": 1e-10, "fatol": 1e-12})
    return 10.0 ** res.x, res.fun


def report_gap(df):
    data = _stack_data(df)
    # crude sigmas for a first fit, then freeze at the global optimum and refit
    sigmas = {"Early": SIGMA_LOGB, "Late": SIGMA_LOGB, "Early_A": SIGMA_LOGA}
    p_true, _ = fit_mode([K1_TRUE, K2_TRUE, SF_TRUE], data, sigmas)
    sigmas = frozen_sigmas_at(p_true, data)
    p_true, nll_true = fit_mode(p_true, data, sigmas)
    p_swap0 = [p_true[1], p_true[0], p_true[2] * p_true[0] / p_true[1]]
    p_swap, nll_swap = fit_mode(p_swap0, data, sigmas)
    print(f"  global mode : k1={p_true[0]:.4f}  k2={p_true[1]:.4f}  SF={p_true[2]:.4f}")
    print(f"  swapped mode: k1={p_swap[0]:.4f}  k2={p_swap[1]:.4f}  SF={p_swap[2]:.4f}")
    print(f"  frozen sigmas: " + ", ".join(f"{k}={v:.4f}" for k, v in sigmas.items()))
    print(f"  realized swap-mode dNLL gap = {nll_swap - nll_true:.4f}"
          f"  (95% threshold = 1.9207)")
    return nll_swap - nll_true


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--out", type=str, default="Flipflop.csv")
    ap.add_argument("--scan", type=int, default=0,
                    help="scan seeds 0..N-1 and report each realized gap instead of writing")
    args = ap.parse_args()

    if args.scan:
        for s in range(args.scan):
            df = generate(s)
            print(f"seed {s}:")
            report_gap(df)
    else:
        df = generate(args.seed)
        df.to_csv(args.out, index=False, float_format="%.6g")
        print(f"Wrote {args.out}  (seed={args.seed})")
        report_gap(df)
