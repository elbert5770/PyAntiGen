"""Apply a parameter vector to a RoadRunner model the way a fit does, and read the result.

The fit (``Optimize.run_all``) does not just write ``r[name] = value``: it resets the model, restores the parameter
baseline, runs the replicate's ``Update_parameters``, sets the fitted values, then runs the replicate's
``parameter_hooks`` and ``Update_opt_parameters``. A fitted name may therefore be a model parameter or something a
hook translates (a dissociation constant that becomes a pair of rate constants).

:class:`ParameterReplay` repeats that sequence. Two users:

* the AMICI route (``Analytic_gradient``): the plain parameters read back from the replayed model are what AMICI
  integrates, and differencing the replay gives the Jacobian of those parameters with respect to the fitted ones;
* a prediction (``Model_simulate.run_simulation(member=...)``): a parameter vector from a stored fit or profile
  is given to the model with exactly the meaning it had in the fit, on either backend.

Importing this module needs neither AMICI nor a POSIX-only module, so a RoadRunner-only install can use it.
"""
import numpy as np


class ParameterReplay:
    """Apply fitted parameters to a RoadRunner model exactly as ``Optimize.run_all`` does, and read the result.

    The engine resets the model, restores the parameter baseline, runs the replicate's
    ``Update_parameters``, sets the fitted values, then the ``parameter_hooks`` and
    ``Update_opt_parameters``. Replaying that and reading every plain parameter back
    gives the model parameters AMICI needs, and (by differencing the replay) the
    Jacobian of the model parameters with respect to the fitted ones -- whatever the
    hooks do in between.
    """

    def __init__(self, r, replicate, param_names, plain):
        self.r, self.replicate = r, replicate
        self.param_names, self.plain = list(param_names), list(plain)
        self.species = list(r.model.getFloatingSpeciesIds())

    def apply(self, p_lin):
        from pyantigen.engine.Optimize import restore_parameter_baseline, set_parameters_from_dict
        r, rep = self.r, self.replicate
        params = dict(zip(self.param_names, np.asarray(p_lin, dtype=float).tolist()))
        r.reset()
        restore_parameter_baseline(r)
        if rep.get("Update_parameters") is not None:
            rep["Update_parameters"](r, rep)
        set_parameters_from_dict(r, params)
        for hook in rep.get("parameter_hooks", []):
            hook(r, params)
        if rep.get("Update_opt_parameters") is not None:
            rep["Update_opt_parameters"](r, rep, params)

    def read(self, names):
        return np.array([float(self.r[n]) for n in names])

    def values(self, p_lin):
        """``{plain parameter: value}`` after the engine's updates."""
        self.apply(p_lin)
        return dict(zip(self.plain, self.read(self.plain)))

    def jacobian(self, p_lin, h_rel=1e-6):
        """``(values, J)``: J[m, k] = d(plain parameter m)/d(theta_k), central differences of the replay."""
        p_lin = np.asarray(p_lin, dtype=float)
        base = self.values(p_lin)
        J = np.zeros((len(self.plain), len(p_lin)))
        for k in range(len(p_lin)):
            h = h_rel * max(abs(p_lin[k]), 1e-8)
            up, dn = p_lin.copy(), p_lin.copy()
            up[k] += h
            dn[k] -= h
            vu = np.array(list(self.values(up).values()))
            vd = np.array(list(self.values(dn).values()))
            J[:, k] = (vu - vd) / (2.0 * h)
        self.apply(p_lin)
        return base, J

    def direct_species(self, p_lin, rtol=1e-9):
        """Species a hook set directly: ``{name: amount}`` where the model differs from what ``reset()`` derives.

        ``reset()`` re-evaluates a species' initial assignment from the current parameters,
        so those need no help; a value a hook wrote over that does, because AMICI never sees the hook.
        """
        self.apply(p_lin)
        mine = np.array(self.r.model.getFloatingSpeciesAmounts(), dtype=float)
        self.r.reset()
        derived = np.array(self.r.model.getFloatingSpeciesAmounts(), dtype=float)
        tol = rtol * np.maximum(np.maximum(np.abs(mine), np.abs(derived)), 1e-300)
        off = np.flatnonzero(np.abs(mine - derived) > tol)
        self.apply(p_lin)
        return {self.species[i]: float(mine[i]) for i in off}


def unreached(replay, p_lin, names=None, rel=1e-2):
    """The fitted names (all of ``replay.param_names``, or *names*) that change nothing in the model when moved.

    ``set_parameters_from_dict`` ignores a name the model does not have, as the fit's own path does, so a
    misspelt name, or one that no hook reads, would leave a prediction equal to the optimum without any error.
    A name that is a global parameter of the model is reached by definition. Any other is moved by *rel*
    (or 1e-3 when it is zero) and the model's parameters and species amounts are compared with the unmoved
    replay; a name with no effect on either is returned.
    """
    r = replay.r
    p_lin = np.asarray(p_lin, dtype=float)
    ids = set(r.getGlobalParameterIds()) | set(r.model.getFloatingSpeciesIds())
    out = []

    def state():
        return np.concatenate([np.asarray(r.getGlobalParameterValues(), dtype=float),
                               np.asarray(r.model.getFloatingSpeciesAmounts(), dtype=float)])

    replay.apply(p_lin)
    base = state()
    for k, name in enumerate(replay.param_names):
        if (names is not None and name not in names) or name in ids:
            continue
        moved = p_lin.copy()
        moved[k] = p_lin[k] * (1.0 + rel) if p_lin[k] != 0 else 1e-3
        replay.apply(moved)
        if np.array_equal(state(), base, equal_nan=True):
            out.append(name)
    replay.apply(p_lin)
    return out
