"""Fisher information at an optimum from the AMICI residual Jacobian (fixed sigmas).

With every sigma known the loss is ``const + 0.5 r.r``, so the Fisher information is exactly ``F = J^T J`` with ``J`` the
Jacobian of the scaled residuals. It is reported in the optimizer's space (per decade for a log10 parameter). From it:
eigenvalues and condition number, the Cramer-Rao covariance (pseudo-inverse when singular), Wald standard errors and 95 %
intervals, correlations, conditional standard deviations ``1/sqrt(F_ii)``, and the weakest directions with their loadings.

A local, linear-Gaussian statement at one point: the standard errors are lower bounds that ignore the parameter bounds, and
they are not profile-likelihood intervals. The sigmas are fixed, so they carry the sampling error of the noise floors.
"""
import math

import numpy as np

Z95 = 1.959964


def fim_report(J, names, scales, x, tol=1e-4, n_weak=3, n_load=4):
    """The Fisher information report as a dict of arrays and lists (JSON-friendly except ``F``, ``cov``, ``corr``)."""
    J = np.asarray(J, float)
    x = np.asarray(x, float)
    F = J.T @ J
    n = len(names)
    w, V = np.linalg.eigh(0.5 * (F + F.T))
    d = np.sqrt(np.maximum(np.diag(F), 1e-300))
    Fs = F / np.outer(d, d)
    ws = np.linalg.eigvalsh(0.5 * (Fs + Fs.T))
    cov = np.linalg.pinv(F)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(np.maximum(se, 1e-300), np.maximum(se, 1e-300))
    lin = np.array([10.0 ** v if s == "log10" else v for v, s in zip(x, scales)])
    lo = np.array([10.0 ** (v - Z95 * e) if s == "log10" else v - Z95 * e for v, e, s in zip(x, se, scales)])
    hi = np.array([10.0 ** (v + Z95 * e) if s == "log10" else v + Z95 * e for v, e, s in zip(x, se, scales)])
    weak = []
    for i in range(min(n_weak, n)):
        load = V[:, i]
        top = np.argsort(-np.abs(load))[:n_load]
        weak.append({"eigenvalue": float(w[i]), "loadings": [(names[j], float(load[j])) for j in top]})
    return {"names": list(names), "scales": list(scales), "x": x, "value": lin, "F": F, "eigenvalues": w,
            "eigenvalues_scaled": ws, "condition": float(w[-1] / w[0]) if w[0] > 0 else math.inf,
            "condition_scaled": float(ws[-1] / ws[0]) if ws[0] > 0 else math.inf,
            "n_not_identifiable": int((ws < tol).sum()), "tol": tol, "cov": cov, "se": se, "corr": corr,
            "cond_sd": 1.0 / d, "ci_lo": lo, "ci_hi": hi, "weak": weak, "rank": int(np.linalg.matrix_rank(F))}


def format_fim(rep, f=None):
    names = rep["names"]
    w = max(len(n) for n in names)
    lines = ["[fim] Fisher information F = J'J at the optimum" + (f" (f {f:.6f})" if f is not None else "")
             + ", optimizer space (per decade for log10)",
             f"[fim] {len(names)} parameter(s), rank {rep['rank']}; eigenvalues {rep['eigenvalues'][0]:.3g} .. "
             f"{rep['eigenvalues'][-1]:.3g}, condition {rep['condition']:.3g} (scaled {rep['condition_scaled']:.3g}); "
             f"{rep['n_not_identifiable']} direction(s) below the {rep['tol']:g} relative eigenvalue (not identifiable)",
             f"  {'parameter':<{w}}  {'value':>12}  {'se (opt. space)':>15}  {'cond. sd':>10}  95% Wald interval (parameter units)"]
    for i, n in enumerate(names):
        lines.append(f"  {n:<{w}}  {rep['value'][i]:>12.5g}  {rep['se'][i]:>15.4g}  {rep['cond_sd'][i]:>10.3g}  "
                     f"[{rep['ci_lo'][i]:.5g}, {rep['ci_hi'][i]:.5g}]")
    for k, wk in enumerate(rep["weak"]):
        load = ", ".join(f"{n} {v:+.2f}" for n, v in wk["loadings"])
        lines.append(f"[fim] weakest direction {k + 1}: eigenvalue {wk['eigenvalue']:.3g}; {load}")
    lines.append("[fim] a local lower bound on the uncertainty that ignores the bounds; not a profile interval.")
    return lines


def save_fim(rep, directory, tag="fim", f=None):
    """Write ``<tag>.npz`` and ``<tag>.txt`` into *directory*; returns the .txt path."""
    import os
    os.makedirs(directory, exist_ok=True)
    arrays = {k: np.asarray(rep[k]) for k in ("x", "value", "F", "eigenvalues", "eigenvalues_scaled", "cov", "se", "corr",
                                              "cond_sd", "ci_lo", "ci_hi")}
    np.savez(os.path.join(directory, tag + ".npz"), names=np.array(rep["names"]), scales=np.array(rep["scales"]), **arrays)
    path = os.path.join(directory, tag + ".txt")
    with open(path, "w") as fh:
        fh.write("\n".join(format_fim(rep, f)) + "\n")
    return path
