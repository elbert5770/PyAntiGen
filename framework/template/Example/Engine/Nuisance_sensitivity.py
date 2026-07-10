"""
Nuisance parameter sensitivity analysis via Latin Hypercube Sampling.

Forward-only (no re-optimisation): hold identifiable params at their current
optimal values, vary the non-identifiable nuisance params across their plausible
range, and collect the resulting plasma-PK trajectories.

The compile-once / reset-reuse pattern mirrors Engine/Optimize.py:run_all().
r.reset() restores species to initial conditions; parameter values set by
Update_parameters persist, so only the nuisance overrides need to be applied
each iteration.
"""

import os
import numpy as np
import AntiGen_paths

REPO_ROOT = AntiGen_paths.REPO_ROOT


# ---------------------------------------------------------------------------
# LHS sampling
# ---------------------------------------------------------------------------

def _lhs_sample(nuisance_settings, n_samples, seed=42):
    """
    Return (param_names, samples) where samples has shape (n_samples, n_params).
    FR is sampled uniformly; all other params log-uniformly.
    """
    from scipy.stats.qmc import LatinHypercube

    names = list(nuisance_settings.keys())
    sampler = LatinHypercube(d=len(names), seed=seed)
    unit = sampler.random(n=n_samples)          # (n_samples, n_params) in [0, 1]

    samples = np.zeros_like(unit)
    for i, name in enumerate(names):
        lo, hi = nuisance_settings[name]['bounds']
        if name == 'FR':
            samples[:, i] = lo + unit[:, i] * (hi - lo)
        else:
            samples[:, i] = 10 ** (
                np.log10(lo) + unit[:, i] * (np.log10(hi) - np.log10(lo))
            )
    return names, samples


# ---------------------------------------------------------------------------
# PRCC
# ---------------------------------------------------------------------------

def _compute_prcc(X, y):
    """
    Partial rank correlation coefficient of each column of X against y.
    Removes linear effects of all other columns before computing correlation.
    """
    from scipy.stats import rankdata, pearsonr

    n, p = X.shape
    X_r = np.apply_along_axis(rankdata, 0, X).astype(float)
    y_r = rankdata(y).astype(float)

    prcc = np.zeros(p)
    for i in range(p):
        other = np.delete(X_r, i, axis=1)
        A = np.column_stack([np.ones(n), other])
        cx = np.linalg.lstsq(A, X_r[:, i], rcond=None)[0]
        cy = np.linalg.lstsq(A, y_r,        rcond=None)[0]
        prcc[i], _ = pearsonr(X_r[:, i] - A @ cx, y_r - A @ cy)
    return prcc


# ---------------------------------------------------------------------------
# Total plasma antibody helper
# ---------------------------------------------------------------------------

def _total_plasma(results):
    """Sum free + all Aβ-bound Antibody_Plasma species."""
    from Modules.utils.centiloid_utils import get_column_index

    total = None
    for col in ('[Antibody_Plasma]',
                '[AB38__Antibody_Plasma]',
                '[AB40__Antibody_Plasma]',
                '[AB42__Antibody_Plasma]'):
        idx = get_column_index(results, col)
        if idx is not None:
            conc = results[:, idx]
            total = conc if total is None else total + conc
    return total


# ---------------------------------------------------------------------------
# Target value helper
# ---------------------------------------------------------------------------

def _get_target_values(results, observable_type, placebo_ratio=None):
    """Extract target concentration/values from results based on observable_type."""
    from Modules.utils.centiloid_utils import get_column_index, calculate_centiloids
    
    if observable_type == 'plasma_ab':
        return _total_plasma(results)
    elif observable_type == 'centiloid':
        if placebo_ratio is None:
            from Modules.utils.centiloid_utils import calculate_dense_ratio_77
            placebo_ratio = calculate_dense_ratio_77()
        return calculate_centiloids(results, placebo_ratio)
    else:
        # Generic column name
        idx = get_column_index(results, observable_type)
        if idx is not None:
            return results[:, idx]
        # Try with bracket notation
        idx = get_column_index(results, f'[{observable_type}]')
        if idx is not None:
            return results[:, idx]
        return None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_nuisance_sensitivity(settings, nuisance_param_settings, EXPERIMENT_dict,
                             n_samples=100):
    """
    LHS forward-only nuisance parameter sensitivity analysis.

    For each drug in nuisance_param_settings:
      1. Compile one RoadRunner model per replicate (once).
      2. Run nominal simulation (params already set by Update_parameters).
      3. For each of n_samples LHS combinations: reset + override nuisance
         params + simulate.
      4. Compute PRCC of each nuisance param against mean AUC across replicates.

    Returns
    -------
    output : dict  keyed by drug_name
        param_names, samples, prcc, and per-replicate nominal/spaghetti arrays.
    paths : dict
        Paths dict from AntimonyGen (needed by plot functions).
    """
    from framework.AntimonyGen import AntimonyGen
    from framework.TelluriumGen import TelluriumGen
    from Engine.Simulate import simulate

    MODEL_NAME = settings.get('MODEL_NAME', AntiGen_paths.MODEL_NAME)
    model_text, paths = AntimonyGen(MODEL_NAME, repo_root=REPO_ROOT)
    data_path  = paths['data_path']
    save_dir   = os.path.join(paths['repo_root'], 'generated', MODEL_NAME)
    save_path  = os.path.join(save_dir, MODEL_NAME + '_events.txt')
    os.makedirs(save_dir, exist_ok=True)

    experiment = EXPERIMENT_dict['EXPERIMENT']
    output = {}

    # Check for metadata structure in settings
    if 'params' in nuisance_param_settings:
        drug_settings_dict = nuisance_param_settings['params']
        meta = nuisance_param_settings
    else:
        drug_settings_dict = nuisance_param_settings
        meta = {}

    # Treat a flat parameter dict as a single nuisance case if no drug names match.
    single_global_nuisance = False
    if isinstance(drug_settings_dict, dict):
        rep_drugs = {rep.get('Drug') for rep in experiment.replicates.values() if rep.get('Drug')}
        is_param_dict = (
            drug_settings_dict
            and all(
                isinstance(v, dict) and 'x0' in v and 'bounds' in v
                for v in drug_settings_dict.values()
            )
        )
        if is_param_dict and not (rep_drugs & set(drug_settings_dict.keys())):
            drug_settings_dict = {'Nuisance': drug_settings_dict}
            single_global_nuisance = True

    observable_type = meta.get('target_observable', 'plasma_ab')
    summary_metric  = meta.get('summary_metric', 'auc')
    
    placebo_ratio = None
    if observable_type == 'centiloid':
        from Modules.utils.centiloid_utils import calculate_dense_ratio_77
        placebo_ratio = calculate_dense_ratio_77()

    for drug_name, nuisance_settings in drug_settings_dict.items():
        if single_global_nuisance:
            drug_reps = {
                lbl: rep for lbl, rep in experiment.replicates.items()
                if rep.get('Drug') is not None
            }
        else:
            drug_reps = {
                lbl: rep for lbl, rep in experiment.replicates.items()
                if rep.get('Drug') == drug_name
            }
        if not drug_reps:
            continue

        param_names, samples = _lhs_sample(nuisance_settings, n_samples)
        print(f"\n[nuisance] {drug_name}: "
              f"{len(drug_reps)} replicate(s) × {n_samples} LHS samples  "
              f"({len(param_names)} nuisance params)")

        # ── Compile one model per replicate ──────────────────────────────────
        compiled = {}
        for lbl, rep in drug_reps.items():
            df_dict = rep['Data'](rep, data_path)
            events  = rep['Events'](rep, df_dict)
            with open(save_path, 'w') as f:
                f.write(events)
            r = TelluriumGen(model_text + '\n' + events, paths)
            rep['Update_parameters'](r, rep)
            compiled[lbl] = {
                'r':               r,
                'df_dict':         df_dict,
                'solver_settings': rep['Solver_settings'](rep),
                'observed':        rep['Observed_species'](r),
                'age':             rep['Age'],
                'label':           lbl,
            }

        # ── Nominal trajectories (params already set by Update_parameters) ───
        nominal = {}
        for lbl, c in compiled.items():
            c['r'].reset()
            rep = drug_reps[lbl]
            rep['Update_parameters'](c['r'], rep)
            nominal[lbl] = simulate(c['r'], c['solver_settings'], c['observed'], label=lbl)

        # ── LHS forward simulations ──────────────────────────────────────────
        spaghetti  = {lbl: [] for lbl in drug_reps}
        auc_matrix = np.zeros((n_samples, len(drug_reps)))

        for s_idx, sample in enumerate(samples):
            override = dict(zip(param_names, sample))
            for rep_idx, (lbl, c) in enumerate(compiled.items()):
                r = c['r']
                r.reset()
                rep = drug_reps[lbl]
                rep['Update_parameters'](r, rep)
                param_ids = r.getGlobalParameterIds()
                for pname, pval in override.items():
                    if pname in param_ids:
                        r[pname] = pval
                try:
                    results = simulate(r, c['solver_settings'], c['observed'], label=lbl)
                except Exception as exc:
                    print(f"  [nuisance] sample {s_idx} {lbl} failed: {exc}")
                    spaghetti[lbl].append(None)
                    continue

                spaghetti[lbl].append(results)

                conc = _get_target_values(results, observable_type, placebo_ratio)
                if conc is not None:
                    if summary_metric == 'auc':
                        time    = np.asarray(results['time'])
                        t_event = c['age'] * 365 * 24
                        mask    = time >= t_event
                        if mask.sum() > 1:
                            auc_matrix[s_idx, rep_idx] = np.trapezoid(
                                conc[mask], time[mask]
                            )
                    elif summary_metric == 'final':
                        auc_matrix[s_idx, rep_idx] = conc[-1]

        # ── PRCC ─────────────────────────────────────────────────────────────
        mean_auc = auc_matrix.mean(axis=1)
        prcc = (
            _compute_prcc(samples, mean_auc)
            if mean_auc.std() > 0
            else np.zeros(len(param_names))
        )

        print(f"  PRCC ({summary_metric.upper()} of {observable_type}):")
        for pn, prc in zip(param_names, prcc):
            print(f"    {pn:<35} {prc:+.3f}")

        output[drug_name] = {
            'param_names': param_names,
            'samples':     samples,
            'prcc':        prcc,
            'meta':        meta,
            'replicates': {
                lbl: {
                    'nominal':     nominal[lbl],
                    'spaghetti':   spaghetti[lbl],
                    'auc_samples': auc_matrix[:, i],
                    'age':         compiled[lbl]['age'],
                }
                for i, lbl in enumerate(drug_reps)
            },
        }

    return output, paths
