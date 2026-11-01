"""Simulate a replicate with AMICI, and compare the trajectory with RoadRunner's.

``Model_run.py --simulate KEY --backend amici`` lands here. The replicate's RoadRunner model is built as
for any simulation; AMICI then integrates the same model (its SBML, the replicate's clock-driven inputs as a
schedule) on the output grid of the replicate's tracked blocks and the result comes back in the shape
:func:`pyantigen.engine.Simulate.simulate` returns, so the experiment's plot function does not know which
solver ran.

A switch at an output time is read after the switch. RoadRunner reports an input (a parameter an event assigns) at
that instant from before the switch, so the two differ there in that column alone; states agree.
The row at the very start of the simulation window is NaN: AMICI reports the end of each segment, never
the window's own first instant.
"""
import numpy as np

from pyantigen.engine.Simulate import StackedResult


def block_times(solver_settings):
    """The output times of the tracked blocks, in the order and with the repeats RoadRunner's result has."""
    blocks = solver_settings["simulation_blocks"]
    blocks = list(blocks.values()) if isinstance(blocks, dict) else list(blocks)
    last = max((i for i, b in enumerate(blocks) if b.get("tracked", True)), default=-1)
    out = []
    for i, b in enumerate(blocks):
        if i > last:
            break
        if not b.get("tracked", True):
            continue
        if b.get("times") is not None:
            out.append(np.asarray(b["times"], dtype=float))
            continue
        start = b.get("start", b.get("start_time"))
        end = b.get("end", b.get("end_time"))
        n = b.get("n_points", b.get("eval_points"))
        out.append(np.linspace(float(start), float(end), int(n)))
    if not out:
        raise ValueError("no tracked simulation block to read output times from")
    return np.concatenate(out)


def simulate_amici(replicate, r, events, model_text, paths, name, cache_dir, solver_settings,
                   rtol=1e-10, atol=1e-12, max_steps=100000, verbose=True, seed=None):
    """``(result, dropped)``: the replicate simulated by AMICI, shaped like ``simulate()``'s result.

    *r* is the replicate's RoadRunner model after ``Update_parameters`` (and ``attach_event_times``);
    *seed* is ``{state: amount}`` given to AMICI at the start on top of the model's own initial state (a state at
    exactly zero can make a rate 0/0 in the Jacobian); RoadRunner is not given it.
    *dropped* lists observed columns the SBML does not define, which are absent from the result.
    """
    from pyantigen.engine.Analytic_gradient import _SimGradient

    sim = _SimGradient(name, replicate, {"r": r, "events": events}, {}, {}, {}, [], [np.zeros(0)],
                       model_text=model_text, paths=paths, cache_dir=cache_dir, rtol=rtol, atol=atol,
                       max_steps=max_steps, verbose=verbose, seed=seed)
    all_t = block_times(solver_settings)
    unique, inverse = np.unique(all_t, return_inverse=True)
    usable = unique > sim.t_start
    table = sim.trajectory(unique[usable])
    full = np.full((len(unique), len(sim.columns)), np.nan)
    full[:, 0] = unique
    full[usable, 1:] = table
    return StackedResult(full[inverse], sim.columns), list(sim.dropped)


def compare_results(amici_result, roadrunner_result, rel_floor=1e-9):
    """Worst disagreement per shared column: ``[(column, max |a - b|, max |a - b| / max|b|, time, a, b)]``, largest relative first.

    Rows where either solver is NaN (the window start) are skipped. The relative figure is scaled by the
    column's own largest value, so a column that is zero throughout does not blow it up. *time*, *a* and *b* are
    where the largest difference is and what AMICI and RoadRunner hold there.
    """
    a_cols, b_cols = list(amici_result.colnames), list(roadrunner_result.colnames)
    a = np.asarray(amici_result)
    b = np.asarray(roadrunner_result)
    if a.shape[0] != b.shape[0]:
        raise ValueError(f"the solvers returned {a.shape[0]} and {b.shape[0]} rows")
    rows = []
    for c in a_cols:
        if c == "time" or c not in b_cols:
            continue
        x, y = a[:, a_cols.index(c)], b[:, b_cols.index(c)]
        ok = np.isfinite(x) & np.isfinite(y)
        if not ok.any():
            continue
        gap = np.abs(x[ok] - y[ok])
        k = int(np.argmax(gap))
        diff = gap[k]
        scale = max(np.max(np.abs(y[ok])), rel_floor)
        rows.append((c, float(diff), float(diff / scale), float(a[ok, 0][k]), float(x[ok][k]), float(y[ok][k])))
    rows.sort(key=lambda t: -t[2])
    return rows
