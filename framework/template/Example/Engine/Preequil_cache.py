"""Reuse of the pre-dose simulation segment across objective evaluations.

The PK and antibody solver settings open with a ``preequil`` block that ages the
model from birth to the first dose -- seventy-odd years of integration whose
trajectory is never fitted (``tracked: False``); only the state it leaves behind
matters. On the microglia group that block is 1.5 s of the 5.2 s each arm costs,
so across twelve arms it is roughly 28% of every objective evaluation, repeated
unchanged for every one of the thousands of evaluations a fit or a profile runs.

It is unchanged because the parameters being fitted cannot act before the first
dose. Microglial activation is gated on ``f10_dense``, which is zero while
``Sat_Dense`` -- the antibody-bound fraction of dense plaque -- is zero;
``Microglia_high`` starts at zero with no other production route, so the
high-state clearances and ``k_deact_high`` multiply zero; and every remaining
fitted parameter either scales an ``__Antibody`` species or is an antibody
binding rate. With no antibody in the system they are all inert.

So the block is computed once per model and its end state restored thereafter.
Two things make that safe rather than merely fast:

* **The key is the state, not a name.** The cache is keyed on the model's entire
  state after ``reset()`` and ``Update_parameters`` -- before any fitted value is
  applied -- together with the block and solver spec. That state is a pure
  function of (model, arm), so it is identical across evaluations by
  construction and no assumption about *which* parameters are inert is encoded
  in the key. Anything that does change the pre-dose setup changes the key and
  gets its own entry.

* **The assumption is tested, not asserted.** Using the cache means applying the
  fitted parameters *after* the pre-dose block instead of before it, which is
  only valid while those parameters are inert there. :func:`verify_invariance`
  checks exactly that at startup by integrating the block under two different
  parameter vectors and comparing the results, and the cache refuses to enable
  itself if the check fails. Without it, a later model edit that let a fitted
  parameter act before the first dose would silently corrupt every result with
  nothing in the output to show for it.
"""

import hashlib

import numpy as np

from Engine.Simulate import (
    clamp_state_dust,
    configure_integrator,
    restore_model_state,
    safe_simulate,
    save_model_state,
)


def split_preequil_block(solver_settings):
    """Return ``(cacheable_block, remaining_settings)``.

    The candidate is the *leading* block, and only when it is untracked: an
    untracked block contributes no output, so replaying its end state is
    equivalent to running it. A tracked leading block (Figure8's, for instance)
    contributes rows to the result and is left alone.
    """
    blocks = solver_settings.get("simulation_blocks")
    if isinstance(blocks, dict):
        items = list(blocks.items())
    elif isinstance(blocks, (list, tuple)):
        items = list(enumerate(blocks))
    else:
        return None, solver_settings

    if len(items) < 2:
        # Nothing would remain to simulate; not worth special-casing.
        return None, solver_settings

    _name, first = items[0]
    if not isinstance(first, dict) or first.get("tracked", True):
        return None, solver_settings

    rest = dict(solver_settings)
    if isinstance(blocks, dict):
        rest["simulation_blocks"] = {k: v for k, v in items[1:]}
    else:
        rest["simulation_blocks"] = [b for _k, b in items[1:]]
    return first, rest


def _digest(state, block, solver_settings):
    """Identity of a pre-dose result: the starting state and how it is run."""
    h = hashlib.sha256()
    for k, v in zip(state["keys"], state["values"]):
        h.update(str(k).encode("utf-8"))
        h.update(repr(float(v)).encode("utf-8"))
    for field in ("start", "end", "n_points", "variable_step_size",
                  "maximum_num_steps"):
        h.update(f"{field}={block.get(field)!r}".encode("utf-8"))
    for field in ("integrator", "absolute_tolerance", "relative_tolerance",
                  "stiff", "maximum_num_steps"):
        h.update(f"{field}={solver_settings.get(field)!r}".encode("utf-8"))
    return h.hexdigest()


class PreequilCache:
    """Stores the end state of the pre-dose block for one model.

    In practice this holds a single entry -- one arm per RoadRunner instance,
    one pre-dose block per arm -- but it is keyed rather than a bare slot so
    that anything which does alter the pre-dose setup produces a new entry
    instead of silently reusing the wrong state.
    """

    def __init__(self, enabled=True):
        self.enabled = bool(enabled)
        self._store = {}
        self.n_hits = 0
        self.n_misses = 0

    def apply(self, r, solver_settings, observed_species, label=None):
        """Resolve the pre-dose block; return the settings still to simulate.

        On a hit the model is left in the stored end state. On a miss the block
        is integrated here, under the same solver configuration ``simulate``
        would have used, and the resulting state stored.
        """
        if not self.enabled:
            return solver_settings

        block, rest = split_preequil_block(solver_settings)
        if block is None:
            return solver_settings

        configure_integrator(r, solver_settings)
        key = _digest(save_model_state(r), block, solver_settings)

        hit = self._store.get(key)
        if hit is not None:
            restore_model_state(r, hit)
            self.n_hits += 1
            return rest

        safe_simulate(r, block, observed_species, label=label)
        # Clear the dust before snapshotting, so every later cache hit restores
        # an already-clean state and the saving compounds with the cache rather
        # than being re-paid on each hit.
        clamp_state_dust(r)
        self._store[key] = save_model_state(r)
        self.n_misses += 1
        return rest

    def stats(self):
        return {"hits": self.n_hits, "misses": self.n_misses,
                "entries": len(self._store)}


# ---------------------------------------------------------------------------
# The self-check
# ---------------------------------------------------------------------------

def _setup_and_integrate(r, replicate, param_names, p_vec, block,
                         solver_settings, observed_species):
    """One pre-dose run at *p_vec*, in the order the uncached path uses.

    Returns the state immediately after setup and the state after integrating,
    so the caller can tell a value that was *set* differently from one that
    *evolved* differently.
    """
    from Engine.Optimize import set_parameters_from_dict

    r.reset()
    upd = replicate.get("Update_parameters")
    if upd is not None:
        upd(r, replicate)

    set_parameters_from_dict(r, dict(zip(param_names, np.asarray(p_vec).tolist())))
    for hook in replicate.get("parameter_hooks", []):
        hook(r, p_vec)
    upd_opt = replicate.get("Update_opt_parameters")
    if upd_opt is not None:
        upd_opt(r, replicate, p_vec)

    before = save_model_state(r)
    configure_integrator(r, solver_settings)
    _res, meta = safe_simulate(r, block, observed_species, label="preequil-check")
    # Match what apply() stores, so the check is run against the state the cache
    # would actually hand to the dosed block.
    clamp_state_dust(r)
    return before, save_model_state(r), meta


def _achieved_settings(meta, solver_settings):
    """The tolerances the pre-dose block actually converged at.

    ``safe_simulate`` loosens tolerances and subdivides until an integration
    succeeds, then restores the original settings, so the accuracy a block was
    computed at is not the accuracy that was requested. Where it subdivided, the
    loosest tolerance reached anywhere is what limits the result.

    Returns ``(settings_or_None, rel_tol)``; the first is None when the very
    first attempt succeeded, meaning nothing needs overriding.
    """
    requested = float(solver_settings.get("relative_tolerance", 1e-8))

    def walk(m):
        found = []
        if not isinstance(m, dict):
            return found
        if m.get("rel_tol") is not None:
            found.append({k: m.get(k) for k in
                          ("abs_tol", "rel_tol", "max_steps", "initial_time_step")})
        for key in ("meta1", "meta2"):
            found.extend(walk(m.get(key)))
        return found

    candidates = walk(meta)
    if not candidates:
        return None, requested

    worst = max(candidates, key=lambda d: float(d.get("rel_tol") or 0.0))
    rel = float(worst.get("rel_tol") or requested)
    settings = dict(solver_settings)
    if worst.get("abs_tol") is not None:
        settings["absolute_tolerance"] = float(worst["abs_tol"])
    settings["relative_tolerance"] = rel
    if worst.get("max_steps") is not None:
        settings["maximum_num_steps"] = int(worst["max_steps"])
    return settings, rel


def _alternate_vector(x0, bounds):
    """A parameter vector far from *x0* but inside the bounds.

    Multiplying by five moves every parameter far enough that any pre-dose
    influence would show; where that hits a bound the value is moved the other
    way instead, so no entry silently stays put and weakens the test.
    """
    x0 = np.asarray(x0, dtype=float)
    out = np.array(x0, dtype=float, copy=True)
    for i, v in enumerate(x0):
        lo, hi = (bounds[i] if bounds is not None and i < len(bounds)
                  else (None, None))
        lo = -np.inf if lo is None else float(lo)
        hi = np.inf if hi is None else float(hi)
        for cand in (v * 5.0, v / 5.0, (lo + hi) / 2.0 if np.isfinite(lo + hi) else v):
            c = min(max(cand, lo), hi)
            if not np.isclose(c, v, rtol=1e-6, atol=0.0):
                out[i] = c
                break
    return out


def verify_invariance(r, replicate, param_names, x0, bounds, rtol=1e-8,
                      atol=1e-20, verbose=True):
    """Is the pre-dose block independent of the fitted parameters?

    Integrates it under two different parameter vectors and compares. Entries
    that already differ *before* integration are excluded: those are the fitted
    parameters themselves and whatever ``Update_opt_parameters`` derives from
    them, which are expected to differ and say nothing about the dynamics. Any
    entry that starts equal and ends different is a genuine pre-dose dependence,
    and means the cache must not be used.

    Returns ``(ok, report)``.
    """
    solver_settings = replicate["Solver_settings"](replicate)
    block, _rest = split_preequil_block(solver_settings)
    if block is None:
        return False, {"reason": "no cacheable pre-dose block",
                       "offenders": []}

    observed_species = replicate["Observed_species"](r)
    x_alt = _alternate_vector(x0, bounds)
    if np.allclose(np.asarray(x0, dtype=float), x_alt):
        return False, {"reason": "could not build a distinct second parameter "
                                 "vector inside the bounds", "offenders": []}

    before_a, after_a, meta_a = _setup_and_integrate(
        r, replicate, param_names, x0, block, solver_settings, observed_species)

    # If the first run needed the retry ladder, it converged at looser
    # tolerances than were requested. Run the second under those same settled
    # settings: otherwise the two integrations differ in how they were computed
    # as well as in their parameters, and the comparison cannot separate the
    # two. This is what made 'Aducanumab_3mgkg' report a false dependence -- its
    # pre-dose block fell back to 1e-7, and two runs at 1e-7 agreeing only to
    # 5e-6 is the solver working correctly, not a parameter acting pre-dose.
    settled, achieved_rel = _achieved_settings(meta_a, solver_settings)
    before_b, after_b, _meta_b = _setup_and_integrate(
        r, replicate, param_names, x_alt, block,
        settled or solver_settings, observed_species)

    # Two integrations cannot be asked to agree more closely than either was
    # computed. Over seventy years the error accumulates well past the per-step
    # tolerance, so allow three orders of magnitude above it -- still five
    # orders below the ~0.4 relative divergence a genuinely pre-dose-active
    # parameter produces.
    rtol = max(rtol, 1000.0 * achieved_rel)

    keys = list(after_a["keys"])
    va, vb = np.asarray(after_a["values"], float), np.asarray(after_b["values"], float)
    sa, sb = np.asarray(before_a["values"], float), np.asarray(before_b["values"], float)

    if not (list(before_a["keys"]) == list(before_b["keys"]) == keys
            and len(keys) == va.size == vb.size == sa.size):
        return False, {"reason": "state layout changed between runs",
                       "offenders": []}

    set_differently = ~np.isclose(sa, sb, rtol=0.0, atol=0.0)
    diverged = ~np.isclose(va, vb, rtol=rtol, atol=atol) & ~set_differently

    offenders = []
    for i in np.flatnonzero(diverged):
        den = max(abs(va[i]), abs(vb[i]), 1e-300)
        offenders.append({"name": keys[i], "a": float(va[i]), "b": float(vb[i]),
                          "rel": float(abs(va[i] - vb[i]) / den)})
    offenders.sort(key=lambda d: -d["rel"])

    report = {
        "reason": "ok" if not offenders else "pre-dose state depends on the "
                                             "fitted parameters",
        "n_compared": int((~set_differently).sum()),
        "n_excluded": int(set_differently.sum()),
        "offenders": offenders,
        "block_end": float(block.get("end", float("nan"))),
        "achieved_rel_tol": float(achieved_rel),
        "requested_rel_tol": float(solver_settings.get("relative_tolerance",
                                                       achieved_rel)),
        "compare_rtol": float(rtol),
        "retried": settled is not None,
    }
    if verbose:
        _print_report(replicate, report)
    return (not offenders), report


def _print_report(replicate, report):
    label = replicate.get("Label") or "?"
    if report.get("retried"):
        # Worth saying out loud regardless of the verdict: this arm's pre-dose
        # state is computed at lower accuracy than the spec asks for, and that
        # applies to every run of it, not just to this check.
        print(f"[preequil] note: '{label}' needed the solver retry ladder for "
              f"its pre-dose block and converged at rel_tol="
              f"{report['achieved_rel_tol']:.0e} rather than the requested "
              f"{report['requested_rel_tol']:.0e}; the invariance check was "
              f"compared at {report['compare_rtol']:.0e} to match.")
    if not report["offenders"]:
        print(f"[preequil] invariance check passed on '{label}': "
              f"{report['n_compared']} state value(s) agree after "
              f"{report['block_end'] / 24 / 365:.1f} y of pre-dose integration "
              f"under two different parameter vectors "
              f"({report['n_excluded']} parameter value(s) excluded).")
        return
    print()
    print(f"*** preequil cache DISABLED: the pre-dose segment of '{label}' "
          f"depends on the fitted parameters.")
    print(f"    {len(report['offenders'])} state value(s) diverged before the "
          f"first dose; the cache would have replayed a stale state for all of "
          f"them. Largest differences:")
    for d in report["offenders"][:8]:
        print(f"      {d['name']:<48} {d['a']:>15.8g}  {d['b']:>15.8g}  "
              f"rel={d['rel']:.3e}")
    if len(report["offenders"]) > 8:
        print(f"      ... and {len(report['offenders']) - 8} more")
    print(f"    Runs continue uncached and correct, just slower.")
    print()
