"""Generic PEtab v2 archive exporter.

Writes 6 files to ``out_dir``:
  - parameters.tsv     parameterId, lowerBound, upperBound, nominalValue, estimate, parameterScale
  - conditions.tsv     conditionId, targetId, targetValue
  - experiments.tsv    experimentId, time, conditionId
  - observables.tsv    observableId, observableFormula, noiseFormula, observableTransformation, noiseDistribution
  - measurements.tsv   observableId, experimentId, time, measurement
  - problem.yaml       top-level PEtab problem manifest

Mapping rules
-------------
- Each Experiment.replicates entry maps to one experimentId.
- Each Antimony event line ``[name:] at (trigger): t = v, t2 = v2`` maps to:
    * one conditionId per unique set of (target, value) assignments
    * one row in experiments.tsv at the evaluated trigger time
- Each loss_config observable maps to one observableId; the data
  referenced via data_dict_key/data_column/time_column is emitted into
  measurements.tsv.
- Each parameter in optimization_settings (across all sub-groups) becomes
  a row in parameters.tsv with bounds + optimized value (or x0 fallback).
"""
import math
import os
import re

import numpy as np
import pandas as pd


PETAB_FORMAT_VERSION = "2.0.0"


# ---------------------------------------------------------------------------
# Event-line parser
# ---------------------------------------------------------------------------

_EVENT_RE = re.compile(
    r'^\s*(?:[A-Za-z_][A-Za-z0-9_]*\s*:\s*)?'  # optional "name:"
    r'at\s*\(\s*(?P<trigger>.+?)\s*\)\s*:\s*'  # at (trigger):
    r'(?P<rhs>.+?)\s*;?\s*$'                   # assignments, optional trailing ;
)


def _split_top_level(s, sep):
    """Split *s* on *sep* characters that are not inside parentheses/brackets."""
    out, depth, buf = [], 0, []
    for ch in s:
        if ch in '([{':
            depth += 1
        elif ch in ')]}':
            depth = max(0, depth - 1)
        if ch == sep and depth == 0:
            out.append(''.join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append(''.join(buf))
    return out


def _parse_event_line(line):
    """Parse one Antimony event line; return (time_expr, [(target, value), ...]) or None."""
    m = _EVENT_RE.match(line.strip())
    if not m:
        return None
    trigger = m.group('trigger').strip()
    rhs = m.group('rhs').strip().rstrip(';').strip()
    t_match = re.match(r'time\s*>=\s*(.+)$', trigger)
    time_expr = t_match.group(1).strip() if t_match else trigger
    assigns = []
    for chunk in _split_top_level(rhs, ','):
        chunk = chunk.strip()
        if not chunk or '=' not in chunk:
            continue
        target, value = chunk.split('=', 1)
        assigns.append((target.strip(), value.strip()))
    return time_expr, assigns


def _eval_time_expr(expr, param_values):
    """Evaluate a time-trigger expression numerically. Return float or None."""
    try:
        ns = {"__builtins__": {}, "math": math, "np": np, "pi": math.pi}
        ns.update(param_values)
        return float(eval(expr, ns))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Identifier helpers
# ---------------------------------------------------------------------------

def _sanitize_id(s):
    out = re.sub(r'[^A-Za-z0-9_]', '_', str(s))
    if not out:
        return "_id"
    if not (out[0].isalpha() or out[0] == '_'):
        out = '_' + out
    return out


def _observable_id(obs_cfg):
    obs = obs_cfg["observed_variable"]
    if callable(obs):
        return _sanitize_id(getattr(obs, "__name__", "obs_callable"))
    return _sanitize_id(obs)


def _observable_formula(obs_cfg):
    obs = obs_cfg["observed_variable"]
    if callable(obs):
        return f"callable:{getattr(obs, '__name__', repr(obs))}"
    return str(obs)


# ---------------------------------------------------------------------------
# df_dict resolution (mirrors Engine.Optimize._resolve_obs_df)
# ---------------------------------------------------------------------------

def _resolve_obs_df(df_dict, obs_cfg):
    if not isinstance(df_dict, dict):
        return df_dict if hasattr(df_dict, 'columns') else None
    key = obs_cfg.get("data_dict_key")
    if key is not None:
        return df_dict.get(key)
    d_col = obs_cfg["data_column"]
    t_col = obs_cfg["time_column"]
    return next(
        (v for v in df_dict.values()
         if hasattr(v, 'columns') and d_col in v.columns and t_col in v.columns),
        None,
    )


# ---------------------------------------------------------------------------
# Parameter-table assembly
# ---------------------------------------------------------------------------

def _as_list(value):
    """Coerce *value* to a plain Python list. None becomes []."""
    if value is None:
        return []
    return list(value)


def _build_parameter_rows(optimization_settings, group_optimizations):
    """Flatten per-group settings into one parameter table.

    Per-group sub-dicts and flat optimization_settings are both supported.
    Optimized x from group_optimizations replaces x0 when available.
    A parameter that appears in multiple groups is written once (first seen).
    """
    rows = []
    seen = set()

    def _add(names, x0s, bounds, optimized_x):
        for i, pn in enumerate(names):
            if pn in seen:
                continue
            seen.add(pn)
            lo, hi = (bounds[i] if i < len(bounds) else (None, None))
            if optimized_x is not None and i < len(optimized_x):
                nominal = float(optimized_x[i])
            elif i < len(x0s):
                nominal = float(x0s[i])
            else:
                nominal = None
            rows.append({
                "parameterId": pn,
                "lowerBound": lo if lo is not None else "",
                "upperBound": hi if hi is not None else "",
                "nominalValue": nominal if nominal is not None else "",
                "estimate": 1,
                "parameterScale": "lin",
            })

    # Flat mode
    if "param_names" in optimization_settings:
        opt = group_optimizations.get("__flat__") if group_optimizations else None
        opt_x = _as_list(opt.get("x")) if opt is not None else None
        _add(
            _as_list(optimization_settings.get("param_names")),
            _as_list(optimization_settings.get("x0")),
            _as_list(optimization_settings.get("bounds")),
            opt_x,
        )
        return rows

    # Per-group mode — only emit params from groups that actually ran.
    for group_name, settings in optimization_settings.items():
        if not isinstance(settings, dict) or "param_names" not in settings:
            continue
        opt = group_optimizations.get(group_name) if group_optimizations else None
        if opt is None:
            continue
        _add(
            _as_list(settings.get("param_names")),
            _as_list(settings.get("x0")),
            _as_list(settings.get("bounds")),
            _as_list(opt.get("x")),
        )
    return rows


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def export_petab(
    out_dir,
    model_name,
    experiment,
    optimization_settings,
    group_optimizations,
    data_path,
    model_file_rel="model.antimony",
):
    """Write a PEtab v2 archive to *out_dir*.

    Parameters
    ----------
    out_dir : str
        Destination directory (created if missing).
    model_name : str
        Model identifier used inside problem.yaml.
    experiment : Experiment
        Object with .replicates dict.
    optimization_settings : dict
        Either flat {param_names, x0, bounds, ...} or per-group sub-dicts.
    group_optimizations : dict
        {group_name: opt_result_dict}.  For flat settings use the key
        "__flat__".  opt_result_dict must have an "x" key with the
        optimized parameter values.  Empty dict is allowed (uses x0).
    data_path : str
        Root data directory passed to replicate Data loaders.
    model_file_rel : str
        Relative path to the model file as referenced from problem.yaml.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Flat name->value lookup for evaluating event-trigger expressions
    param_values = {}
    param_rows = _build_parameter_rows(optimization_settings, group_optimizations)
    for r in param_rows:
        v = r.get("nominalValue")
        if isinstance(v, (int, float)) and not (isinstance(v, float) and math.isnan(v)):
            param_values[r["parameterId"]] = float(v)

    cond_rows = []
    expt_rows = []
    obs_defs = {}
    meas_rows = []

    cond_cache = {}
    cond_counter = [0]

    def _condition_for(assigns):
        key = tuple(sorted(assigns))
        if key in cond_cache:
            return cond_cache[key]
        cond_counter[0] += 1
        cid = f"cond_{cond_counter[0]:03d}"
        cond_cache[key] = cid
        for tgt, val in assigns:
            cond_rows.append({
                "conditionId": cid,
                "targetId": tgt,
                "targetValue": val,
            })
        return cid

    for rep_key, rep in experiment.replicates.items():
        experiment_id = _sanitize_id(rep_key)

        try:
            df_dict = rep["Data"](rep, data_path)
        except Exception as exc:
            print(f"[petab] {rep_key}: Data loader failed — {exc}")
            df_dict = {}

        events_fn = rep.get("Events")
        events_str = ""
        if events_fn is not None:
            try:
                events_str = events_fn(rep, df_dict, r_ic=None)
            except TypeError:
                try:
                    events_str = events_fn(rep, df_dict)
                except Exception as exc:
                    print(f"[petab] {rep_key}: Events call failed — {exc}")
            except Exception as exc:
                print(f"[petab] {rep_key}: Events call failed — {exc}")

        for line in (events_str or "").splitlines():
            parsed = _parse_event_line(line)
            if parsed is None:
                continue
            time_expr, assigns = parsed
            if not assigns:
                continue
            t_num = _eval_time_expr(time_expr, param_values)
            cid = _condition_for(assigns)
            expt_rows.append({
                "experimentId": experiment_id,
                "time": t_num if t_num is not None else time_expr,
                "conditionId": cid,
            })

        lc_fn = rep.get("Loss_config")
        if lc_fn is None:
            continue
        try:
            lc = lc_fn(rep)
        except Exception:
            continue
        observables_cfg = (lc or {}).get("observables", [])
        if not observables_cfg:
            continue

        for obs_cfg in observables_cfg:
            obs_id = _observable_id(obs_cfg)
            if obs_id not in obs_defs:
                obs_defs[obs_id] = {
                    "observableId": obs_id,
                    "observableFormula": _observable_formula(obs_cfg),
                    "noiseFormula": f"noiseParameter1_{obs_id}",
                    "observableTransformation": "lin",
                    "noiseDistribution": "normal",
                }
            obs_df = _resolve_obs_df(df_dict, obs_cfg)
            if obs_df is None:
                continue
            d_col = obs_cfg["data_column"]
            t_col = obs_cfg["time_column"]
            if d_col not in obs_df.columns or t_col not in obs_df.columns:
                continue
            t_vals = np.asarray(obs_df[t_col].values)
            m_vals = np.asarray(obs_df[d_col].values)
            for t_val, m_val in zip(t_vals, m_vals):
                try:
                    fm = float(m_val)
                    ft = float(t_val)
                except (TypeError, ValueError):
                    continue
                if not (np.isfinite(fm) and np.isfinite(ft)):
                    continue
                meas_rows.append({
                    "observableId": obs_id,
                    "experimentId": experiment_id,
                    "time": ft,
                    "measurement": fm,
                })

    # ---- write tables ----
    pd.DataFrame(param_rows, columns=[
        "parameterId", "lowerBound", "upperBound",
        "nominalValue", "estimate", "parameterScale",
    ]).to_csv(os.path.join(out_dir, "parameters.tsv"), sep='\t', index=False)

    pd.DataFrame(cond_rows, columns=["conditionId", "targetId", "targetValue"])\
        .to_csv(os.path.join(out_dir, "conditions.tsv"), sep='\t', index=False)

    pd.DataFrame(expt_rows, columns=["experimentId", "time", "conditionId"])\
        .to_csv(os.path.join(out_dir, "experiments.tsv"), sep='\t', index=False)

    pd.DataFrame(list(obs_defs.values()), columns=[
        "observableId", "observableFormula", "noiseFormula",
        "observableTransformation", "noiseDistribution",
    ]).to_csv(os.path.join(out_dir, "observables.tsv"), sep='\t', index=False)

    pd.DataFrame(meas_rows, columns=[
        "observableId", "experimentId", "time", "measurement",
    ]).to_csv(os.path.join(out_dir, "measurements.tsv"), sep='\t', index=False)

    yaml_lines = [
        f"format_version: {PETAB_FORMAT_VERSION}",
        f"parameter_file: parameters.tsv",
        f"problems:",
        f"  - model_files:",
        f"      {model_name}:",
        f"        location: {model_file_rel}",
        f"        language: antimony",
        f"    condition_files: [conditions.tsv]",
        f"    experiment_files: [experiments.tsv]",
        f"    observable_files: [observables.tsv]",
        f"    measurement_files: [measurements.tsv]",
        "",
    ]
    with open(os.path.join(out_dir, "problem.yaml"), "w") as f:
        f.write("\n".join(yaml_lines))

    print(f"[petab] Archive written to {out_dir}  "
          f"(conditions={len(cond_rows)}, experiments={len(expt_rows)}, "
          f"observables={len(obs_defs)}, measurements={len(meas_rows)}, "
          f"parameters={len(param_rows)})")
