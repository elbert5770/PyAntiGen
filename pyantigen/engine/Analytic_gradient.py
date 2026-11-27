"""The gradient of the fit objective from AMICI's forward sensitivities.

What the optimizer minimizes is the concentrated Gaussian NLL of
``pyantigen.engine.Optimize`` -- groups, pooled sigma blocks, sigma floors, data
weights, observable expressions such as ``np.log10(np.maximum(predicted_B, 1e-12))``.
Writing its derivative out by hand would duplicate all of that, and the copy would
drift. This module does not. It splits the derivative the way the chain rule does:

    dNLL/dtheta_k = sum over simulated values y_ic of  dNLL/dy_ic * dy_ic/dtheta_k

* ``dy/dtheta`` is a table the same shape as the result table: AMICI's sensitivities
  of every observed column, mapped from the model parameters to the fitted ones
  (:class:`ParameterReplay`).
* The contraction with ``dNLL/dy`` is the *directional derivative of the engine's own
  loss* along that table: ``(L(R + h D_k) - L(R - h D_k)) / 2h``, with the loss
  evaluated by ``collect_loss_blocks`` and ``concentrated_nll`` themselves
  (:class:`LossGradient`). It re-evaluates arithmetic on tables already in memory --
  no integration, no noise, so ``h`` can be small and a Richardson step makes the
  result accurate to rounding. Anything the loss does keeps working as it is.

The simulations come from AMICI on the event-free model of
``pyantigen.generate.scaled_time`` (the schedule route), at the observed data times
and the ends of the tracked window; between them the engine interpolates linearly,
and the gradient is that of the loss at the data times.

Where the engine's objective is not a function of the parameters alone -- a
parameter with an initial assignment that RoadRunner's ``reset()`` does not
re-derive (``Q_SN = Q_CSF*f_SN``), a species set by a hook -- the gradient follows
the engine, not the model's own formulas: the value RoadRunner holds after the
engine's parameter updates is what AMICI is given.

AMICI is optional; on Windows it needs WSL. Nothing here imports it until a
gradient is built.
"""
import contextlib
import hashlib
import json
import os
import re
import threading
import tempfile

import numpy as np

try:
    import fcntl
except ImportError:          # native Windows: no POSIX locks. Importing this module must still work there (a RoadRunner fit
    fcntl = None             # builds the gradient provider's module on the way); AMICI itself runs under WSL or Linux.

from pyantigen.engine.Multistart import GradientSetupError
from pyantigen.engine.Parameter_replay import ParameterReplay  # noqa: F401  (moved to its own module; the name stays importable from here)
from pyantigen.generate.input_schedule import InputSchedule, ScheduleError


class AnalyticGradientError(RuntimeError):
    """The analytic gradient cannot be built or evaluated for this problem."""


# ---------------------------------------------------------------------------------------
# A result table the loss can read
# ---------------------------------------------------------------------------------------

class SimTable:
    """What the loss reads of a RoadRunner result: ``table["col"]`` and ``table.colnames``."""

    def __init__(self, colnames, data):
        self.colnames = list(colnames)
        self.data = np.asarray(data, dtype=float)
        if self.data.ndim != 2 or self.data.shape[1] != len(self.colnames):
            raise ValueError("a SimTable is (rows, len(colnames))")
        self._index = {c: i for i, c in enumerate(self.colnames)}

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.data[:, self._index[key]]
        return self.data[key]          # numpy-style, as RoadRunner's result array allows: table[:, j], table[i, j]

    def __len__(self):
        return self.data.shape[0]

    def shifted(self, direction, h):
        return SimTable(self.colnames, self.data + h * direction)


# ---------------------------------------------------------------------------------------
# dNLL/dtheta from the loss itself
# ---------------------------------------------------------------------------------------

_NAME = re.compile(r"[A-Za-z_][A-Za-z_0-9]*")


class LossGradient:
    """The gradient of the concentrated NLL along tables of ``d(result)/d(theta)``.

    ``groups``, ``replicates`` and ``param_names`` are the engine's: the same objects
    ``Optimize.collect_loss_blocks`` is called with.
    """

    def __init__(self, groups, replicates, param_names, rel_step=1e-5):
        self.groups, self.replicates = groups, replicates
        self.param_names = list(param_names)
        self.rel_step = float(rel_step)
        self._unread, self._read = {}, set()
        self.explicit = self._explicit_parameters()
        # {(block, observable): sigma}: the floored blocks' sigmas a profile point holds still (see freeze_sigmas)
        self.frozen_sigmas = None

    # -- which parameters the loss reads by name ------------------------------------
    def _explicit_parameters(self):
        """Indices of the fitted parameters the loss names itself (``SF*predicted_B``).

        Those enter the loss a second way, besides through the simulation, and need
        their own partial derivative (four passes of the loss each). A callable observable could read anything: then
        every parameter counts -- unless its loss_config entry says what it reads with ``"reads_parameters": [names]``
        (``[]`` for a callable of the table alone, which sees a fitted parameter only through a table column, and
        that the sensitivities already carry). The default stays conservative; a wrong declaration would drop a partial. A
        parameter that is only assumed read is *probed* before it is differenced (:meth:`_explicit_partial`): one extra
        evaluation of the loss, and when it does not move the partial is zero instead of four passes.
        """
        found, names = set(), set(self.param_names)
        self._probe = set()      # the ones assumed read because a callable observable did not say (see _explicit_partial)
        for g in self.groups.values():
            for elem in g.get("loss_elements", []):
                sims = list(elem.get("simulations", [])) or [elem.get("simulation")]
                lc_fn = elem.get("loss_config")
                for sim in sims:
                    if sim not in self.replicates:
                        continue
                    lc = lc_fn(self.replicates[sim]) if callable(lc_fn) else lc_fn
                    for obs in (lc or {}).get("observables", []):
                        ov = obs.get("observed_variable")
                        if callable(ov):
                            declared = obs.get("reads_parameters")
                            if declared is None:
                                self._probe = set(range(len(self.param_names)))
                                return set(self._probe)
                            found |= set(declared) & names
                            texts = [str(obs.get("noise_formula") or "")]
                        else:
                            texts = [str(ov), str(obs.get("noise_formula") or "")]
                        for t in texts:
                            found |= set(_NAME.findall(t)) & names
        return {self.param_names.index(n) for n in found}

    def _explicit_partial(self, fn, p_lin, k, base):
        """The partial of ``fn`` (the loss or its residual vector, with the simulation held fixed) in parameter *k*, ``fn(p_lin) = base``.

        Richardson-extrapolated central differences, four passes of ``fn``. For a parameter that is only assumed to be read (a callable
        observable that did not declare ``reads_parameters``) one pass at ``p + d`` comes first: if ``fn`` does not move, the loss does
        not read the parameter and the partial is zero, four passes saved. A parameter that moved ``fn`` is read for good; one that did
        not, three times, is not probed again (a callable of the table alone cannot read a parameter by name, so what it never did
        on three points it does not do).
        """
        p_lin = np.asarray(p_lin, dtype=float)
        d = 1e-6 * max(abs(p_lin[k]), 1e-8)
        if k in self._probe:
            if self._unread.get(k, 0) >= 3:
                return 0.0 * np.asarray(base, dtype=float)
            if k not in self._read:
                up = p_lin.copy()
                up[k] += d
                if np.array_equal(np.asarray(fn(up), dtype=float), np.asarray(base, dtype=float)):
                    self._unread[k] = self._unread.get(k, 0) + 1
                    return 0.0 * np.asarray(base, dtype=float)
                self._read.add(k)

        def partial(sh):
            up, dn = p_lin.copy(), p_lin.copy()
            up[k] += sh
            dn[k] -= sh
            return (np.asarray(fn(up), dtype=float) - np.asarray(fn(dn), dtype=float)) / (2.0 * sh)
        return (4.0 * partial(d / 2.0) - partial(d)) / 3.0

    # -- the loss -------------------------------------------------------------------
    def value(self, items, p_lin):
        from pyantigen.engine.Optimize import collect_loss_blocks, concentrated_nll
        p_dict = dict(zip(self.param_names, np.asarray(p_lin, dtype=float).tolist()))
        blocks = collect_loss_blocks(items, self.groups, self.replicates, self.param_names,
                                     np.asarray(p_lin, dtype=float), p_dict,
                                     frozen_sigmas=self.frozen_sigmas)
        if not blocks:
            raise AnalyticGradientError("the loss has no data to score")
        v = concentrated_nll(blocks)
        if not np.isfinite(v):
            raise AnalyticGradientError("the loss is not finite")
        return v

    def _residual_pass(self, items, p_lin, with_predictions=False):
        """``(r, const, records)``: the scaled residuals, ``sum n ln sigma``, and (if asked) what the chain rule needs.

        ``records[i]`` is ``(prediction record, sigma)`` for the i-th run of ``r``: one entry per scored observable, in the
        order the engine's own traversal produced them, so the Jacobian rows built from it line up with ``r``.
        """
        from pyantigen.engine import Optimize as O
        p_lin = np.asarray(p_lin, dtype=float)
        p_dict = dict(zip(self.param_names, p_lin.tolist()))
        sink, psink = [], ([] if with_predictions else None)
        O._RESIDUAL_SINK, O._PREDICTION_SINK = sink, psink
        try:
            blocks = O.collect_loss_blocks(items, self.groups, self.replicates, self.param_names, p_lin, p_dict,
                                           frozen_sigmas=self.frozen_sigmas)
        finally:
            O._RESIDUAL_SINK = O._PREDICTION_SINK = None
        sig, const = {}, 0.0
        for key, v in blocks.items():
            sse, n, ks, fl = O._unpack_block(v)
            if n <= 0:
                continue
            s, _cost, state = O._block_sigma_resolution(sse, n, ks, fl)
            if state != "declared":
                raise AnalyticGradientError(
                    f"block {key} profiles its sigma ({state}): the residual form needs every sigma known "
                    "(set PYANTIGEN_SIGMA_FLOOR=fixed)")
            sig[key] = s
            const += n * np.log(max(s, 1e-300)) + O._block_log_sigma(v)       # + sum w ln sigma_i for a per-point-sigma block
        keep = [i for i, (key, v) in enumerate(sink) if key in sig and v.size]
        if not keep:
            raise AnalyticGradientError("the loss has no data to score")
        r = np.concatenate([sink[i][1] / sig[sink[i][0]] for i in keep])
        records = [(psink[i], sig[sink[i][0]]) for i in keep] if with_predictions else None
        return r, float(const), records

    def residual_vector(self, items, p_lin):
        """``(r, const)``: the scaled residuals ``sqrt(w) (y - yhat) / sigma`` of every datum, and ``sum n ln sigma``.

        With every block's sigma known (``PYANTIGEN_SIGMA_FLOOR=fixed`` and declared sigmas) the loss is exactly
        ``const + 0.5 * r.r``, a nonlinear least-squares problem. Raises if any block still profiles its sigma.
        """
        r, const, _ = self._residual_pass(items, p_lin)
        return r, const

    def _exact_residual_jacobian(self, items, directions, p_lin):
        """``(r, J, const)`` by the chain rule (see :meth:`_exact_gradient`), or ``None`` when some scored observable cannot
        be differentiated that way.

        A residual is ``sqrt(w) (y - prediction) / sigma``, so its row of J is ``-sqrt(w) / sigma * d prediction / d theta``.
        """
        p_lin = np.asarray(p_lin, dtype=float)
        r0, const, records = self._residual_pass(items, p_lin, with_predictions=True)
        jac = self._prediction_jacobians(items, directions, [rec for rec, _sigma in records])
        if jac is None:
            return None
        rows = [-np.sqrt(rec["weights"])[:, None] * dy[rec["valid"]] / sigma
                for (rec, sigma), dy in zip(records, jac)]
        J = np.vstack(rows)
        if J.shape[0] != r0.size:
            return None
        for k in sorted(self.explicit):                 # the loss names this parameter itself: its own partial
            J[:, k] += self._explicit_partial(lambda q: self.residual_vector(items, q)[0], p_lin, k, r0)
        return r0, J, const

    def residual_jacobian(self, items, directions, p_lin):
        """``(r, J, const)``: residuals, ``J[i, k] = d r_i / d theta_k`` (linear units), and ``sum n ln sigma``.

        By the chain rule where the observables allow it (:meth:`_exact_residual_jacobian`: exact for a bare column, only
        the observable differenced otherwise); failing that it is taken the way :meth:`gradient` takes the loss's: central
        differences of the residual vector along the sensitivity tables (Richardson), plus a partial for each parameter
        the loss names itself. ``J.T @ r`` is then the gradient of ``const + 0.5 r.r`` and ``J.T @ J`` its Gauss-Newton Hessian.
        """
        if self.exact:
            got = self._exact_residual_jacobian(items, directions, p_lin)
            if got is not None:
                return got
        p_lin = np.asarray(p_lin, dtype=float)
        K = len(self.param_names)
        r0, const = self.residual_vector(items, p_lin)
        J = np.zeros((r0.size, K))

        def res(it, p):
            r, _c = self.residual_vector(it, p)
            if r.size != r0.size:
                raise AnalyticGradientError("the number of scored data changed between shifted evaluations")
            return r

        for k in range(K):
            for units, h in self._step_groups(items, directions, k):
                def central(x, units=units, k=k):
                    return (res(self._shift_units(items, directions, units, k, x), p_lin)
                            - res(self._shift_units(items, directions, units, k, -x), p_lin)) / (2.0 * x)
                J[:, k] += (4.0 * central(h / 2.0) - central(h)) / 3.0
            if k in self.explicit:
                J[:, k] += self._explicit_partial(lambda q: res(items, q), p_lin, k, r0)
        return r0, J, const

    @staticmethod
    def _shift(items, directions, k, h):
        out = {}
        for sim, item in items.items():
            D = directions.get(sim)
            table = item["results"] if D is None else item["results"].shifted(D[:, :, k], h)
            out[sim] = {**item, "results": table}
        return out

    def _step(self, items, directions, k, sims=None):
        """The step along direction k that moves the smallest-scale entry by ``rel_step`` of itself."""
        ratio = 0.0
        for sim, item in items.items():
            if sims is not None and sim not in sims:
                continue
            D = directions.get(sim)
            if D is None:
                continue
            R = item["results"].data
            colmax = np.nanmax(np.abs(R), axis=0) if R.size else np.zeros(R.shape[1])
            scale = np.maximum(np.abs(R), 1e-6 * colmax[None, :] + 1e-300)
            ratio = max(ratio, float(np.nanmax(np.abs(D[:, :, k]) / scale)) if D.size else 0.0)
        return self.rel_step / ratio if ratio > 0 else None

    # One finite-difference step for the whole loss is set by the table entry with the largest relative sensitivity in ANY
    # column of ANY simulation. A column the loss never reads (45 states in a typical table) can have sensitivities of 1e12
    # that fix a step far too small for the column it does read, and the central difference of that is lost in rounding:
    # on aggregation_silk the Jacobian of one arm changed by a relative 1e-3 between two points 1e-9 apart while its residuals
    # changed by 1e-10 (the gradient jumped by 1.9). The chain rule allows a step per (simulation, column) unit:
    # dL/dtheta_k = sum_u dL/dR_u . D_u,k, each term by shifting only that column. Units whose steps are within SPLIT_RATIO
    # of each other are shifted together, because splitting everything into single columns turns offsetting partial
    # derivatives (mass balances) into rounding error. Only columns that are not read could be left out, and a probe for
    # "read" columns proved unreliable (it missed columns on silk_appfull, error 18.7), so every column with a non-zero
    # sensitivity is a unit. Measured (silk_appfull optimum, aggregation_silk traced points): SPLIT_RATIO from 3 to 1e5 gives
    # the old gradient on silk_appfull (error vs forward differences 0.021) and no jump on aggregation_silk (3e-4, was 1.9);
    # 1e7 is already too coarse there (8e-3).
    SPLIT_RATIO = 1e3

    def _unit_steps(self, items, directions, k):
        """``{(sim, column): h}``: the step that moves that column's most sensitive entry by ``rel_step`` of itself."""
        steps = {}
        for sim, item in items.items():
            D = directions.get(sim)
            if D is None or not D.size:
                continue
            R = item["results"].data
            colmax = np.nanmax(np.abs(R), axis=0)
            for c in range(R.shape[1]):
                Dc = D[:, c, k]
                if not np.any(Dc):
                    continue
                scale = np.maximum(np.abs(R[:, c]), 1e-6 * colmax[c] + 1e-300)
                ratio = float(np.nanmax(np.abs(Dc) / scale))
                if ratio > 0:
                    steps[(sim, c)] = self.rel_step / ratio
        return steps

    def _step_groups(self, items, directions, k, read=None, p_lin=None):
        """``[(units, h)]``: (simulation, column) units that share a step. The smallest remaining step serves every unit
        within SPLIT_RATIO of it; the rest are grouped the same way. (``read`` and ``p_lin`` are accepted and ignored.)"""
        remaining = self._unit_steps(items, directions, k)
        groups = []
        while remaining:
            h0 = min(remaining.values())
            joint = {u for u, h in remaining.items() if h <= max(self.SPLIT_RATIO, 1.0) * h0}      # always includes the smallest
            groups.append((joint, h0))
            for u in joint:
                del remaining[u]
        return groups

    @staticmethod
    def _shift_units(items, directions, units, k, h):
        from collections import defaultdict
        by_sim = defaultdict(list)
        for sim, c in units:
            by_sim[sim].append(c)
        out = dict(items)
        for sim, cols in by_sim.items():
            table = items[sim]["results"]
            data = table.data.copy()
            D = directions[sim]
            for c in cols:
                data[:, c] += h * D[:, c, k]
            out[sim] = {**items[sim], "results": type(table)(table.colnames, data)}
        return out

    # The chain rule, without differencing the whole loss.
    #   prediction_i = interp(t_i, t_sim, h(table))     h = the observable: a column, an expression, a callable
    #   d prediction_i / d theta = interp(t_i, t_sim, d h(table) / d theta)
    #   dSSE_b / d theta = sum_i -2 w_i r_i d prediction_i / d theta,    dNLL / d theta = sum_b (dNLL_b / dSSE_b) dSSE_b / d theta
    # One traversal of the loss, with the engine's own masks, weights, pooling and sigma rules. d h / d theta is exact for a
    # bare column (the sensitivity table itself). For anything else only h is differenced, along the sensitivity directions,
    # which costs a few numpy evaluations of h per step group instead of a pass of the Python/pandas loss (~600 passes per
    # gradient before, 1.9 s on Zheng's 80 columns). A composite (one simulation subtracted from another, a normalised
    # aggregate) is differentiated simulation by simulation and then through its aggregation. Every differenced h is first
    # checked against the prediction the loss itself made; a mismatch hands the whole gradient back to the old route.
    # ``exact`` turns all of this off, to compare the two.
    exact = True
    #: relative size of the step in prediction space when an aggregation is differenced
    AGG_STEP = 1e-6

    def _fd_series(self, items, directions, reqs):
        """``[dy]`` for ``reqs = [(sim, predict, t_data)]``: d predict(table)/d theta_k interpolated onto ``t_data``.

        Central differences with Richardson extrapolation along each sensitivity direction, a step per group of
        (simulation, column) units (``_step_groups``), exactly as the loss was differenced before -- but of ``predict``,
        which is a few numpy operations, rather than of the whole loss.
        """
        from collections import defaultdict
        K = len(self.param_names)
        out = [np.zeros((len(t), K)) for _s, _p, t in reqs]
        by_sim = defaultdict(list)
        for i, (sim, _p, _t) in enumerate(reqs):
            by_sim[sim].append(i)
        for sim, idxs in by_sim.items():
            D = directions.get(sim)
            if D is None or not D.size:
                continue
            sub_items, sub_dirs = {sim: items[sim]}, {sim: D}
            for k in range(K):
                for units, h in self._step_groups(sub_items, sub_dirs, k):
                    for x, wt in ((h / 2.0, 4.0 / 3.0), (h, -1.0 / 3.0)):
                        up = self._shift_units(sub_items, sub_dirs, units, k, x)[sim]["results"]
                        dn = self._shift_units(sub_items, sub_dirs, units, k, -x)[sim]["results"]
                        t_up, t_dn = np.asarray(up["time"]), np.asarray(dn["time"])
                        for i in idxs:
                            _s, predict, t_data = reqs[i]
                            out[i][:, k] += wt * (np.interp(t_data, t_up, predict(up))
                                                  - np.interp(t_data, t_dn, predict(dn))) / (2.0 * x)
        return out

    @staticmethod
    def _column_jacobian(table, D, column, t_data):
        """d interp(t_data, t_sim, table[column]) / d theta, exactly: the sensitivity column, interpolated."""
        Dc = D[:, table.colnames.index(column), :]
        t_sim = np.asarray(table["time"])
        return np.stack([np.interp(t_data, t_sim, Dc[:, k]) for k in range(Dc.shape[1])], axis=1)

    def _aggregate_jacobian(self, aggregation, y_subs, dy_subs):
        """d aggregation(y_subs) / d theta, given d y_sub / d theta for each simulation (``dy_subs[s]`` is (n, K))."""
        if isinstance(aggregation, str) and aggregation in ("mean", "sum"):
            return (np.mean if aggregation == "mean" else np.sum)(np.asarray(dy_subs), axis=0)
        if isinstance(aggregation, str):
            if aggregation != "median":
                return None
            aggregation = lambda subs: np.median(subs, axis=0)           # noqa: E731
        n, K = dy_subs[0].shape
        out = np.zeros((n, K))
        scale = max(float(np.max(np.abs(y_subs))), 1e-300)
        for k in range(K):
            d = [dy[:, k] for dy in dy_subs]
            dmax = float(np.max(np.abs(d)))
            if dmax == 0.0:
                continue
            eps = self.AGG_STEP * scale / dmax
            up = np.asarray(aggregation([y + eps * dd for y, dd in zip(y_subs, d)]), dtype=float)
            dn = np.asarray(aggregation([y - eps * dd for y, dd in zip(y_subs, d)]), dtype=float)
            out[:, k] = (up - dn) / (2.0 * eps)
        return out

    def _prediction_jacobians(self, items, directions, records):
        """``[d prediction / d theta]`` for every record the engine's traversal reported (each ``(len(t_data), K)``), or
        ``None`` when one of them cannot be differentiated this way (the caller then takes the old route)."""
        owner = {id(item["results"]): sim for sim, item in items.items()}
        out = [None] * len(records)
        reqs, plan = [], []               # reqs: (sim, predict, t_data) to difference; plan: where each result goes

        def table(sim):
            return items[sim]["results"]

        def column_or_request(sim, column, predict, t_data, y_pred):
            """An exact Jacobian for a bare column; else a request to difference ``predict`` (checked first)."""
            D = directions.get(sim)
            if D is None or not D.size:
                return ("zero", np.zeros((len(t_data), len(self.param_names))))
            if column is not None:
                return ("done", self._column_jacobian(table(sim), D, column, t_data))
            base = np.interp(t_data, np.asarray(table(sim)["time"]), predict(table(sim)))
            if not np.allclose(base, y_pred, rtol=1e-8, atol=1e-12, equal_nan=True):
                return None               # the closure does not reproduce the loss's own prediction: do not trust it
            reqs.append((sim, predict, t_data))
            return ("fd", len(reqs) - 1)

        for i, rec in enumerate(records):
            comp = rec.get("composite")
            if comp is None:
                sim = owner.get(id(rec["result"]))
                if sim is None:
                    return None
                got = column_or_request(sim, rec["column"], rec["predict"], rec["t_data"], rec["y_pred"])
                if got is None:
                    return None
                plan.append((i, None, [got]))
            else:
                parts = []
                for sub in comp["subs"]:
                    sim = owner.get(id(sub["result"]))
                    if sim is None:
                        return None
                    got = column_or_request(sim, sub["column"], sub["predict"], sub["t_data"], sub["y_pred"])
                    if got is None:
                        return None
                    parts.append(got)
                plan.append((i, comp, parts))

        fd = self._fd_series(items, directions, reqs)
        for i, comp, parts in plan:
            dys = [p[1] if p[0] != "fd" else fd[p[1]] for p in parts]
            if comp is None:
                out[i] = dys[0]
            else:
                out[i] = self._aggregate_jacobian(comp["aggregation"], comp["y_pred_subs"], dys)
                if out[i] is None:
                    return None
        return out

    def _exact_gradient(self, items, directions, p_lin):
        """``(nll, grad)`` by the chain rule, or ``None`` when some scored observable cannot be differentiated that way."""
        from pyantigen.engine import Optimize as O
        p_lin = np.asarray(p_lin, dtype=float)
        p_dict = dict(zip(self.param_names, p_lin.tolist()))
        sink = []
        O._PREDICTION_SINK = sink
        try:
            blocks = O.collect_loss_blocks(items, self.groups, self.replicates, self.param_names, p_lin, p_dict,
                                           frozen_sigmas=self.frozen_sigmas)
        finally:
            O._PREDICTION_SINK = None
        if not blocks:
            raise AnalyticGradientError("the loss has no data to score")
        jac = self._prediction_jacobians(items, directions, sink)
        if jac is None:
            return None
        f0 = O.concentrated_nll(blocks)
        if not np.isfinite(f0):
            raise AnalyticGradientError("the loss is not finite")
        dsse = O.concentrated_nll_dsse(blocks)
        g = np.zeros(len(self.param_names))
        for rec, dy in zip(sink, jac):
            g += dsse[rec["block"]] * (-2.0 * (rec["weights"] * rec["residuals"]) @ dy[rec["valid"]])
        for k in sorted(self.explicit):                 # the loss names this parameter itself: its own partial, as below
            g[k] += self._explicit_partial(lambda q: self.value(items, q), p_lin, k, f0)
        return f0, g

    def gradient(self, items, directions, p_lin):
        """``(nll, grad)`` at *p_lin*.

        items : ``{sim: {"results": SimTable, "data": df_dict, "replicate": ...}}``.
        directions : ``{sim: array (rows, columns, K)}``, d(table)/d(theta_k) in linear units.
        """
        if self.exact:
            got = self._exact_gradient(items, directions, p_lin)
            if got is not None:
                return got
        p_lin = np.asarray(p_lin, dtype=float)
        K = len(self.param_names)
        f0 = self.value(items, p_lin)
        g = np.zeros(K)
        for k in range(K):
            for units, h in self._step_groups(items, directions, k):
                def central(x, units=units, k=k):
                    return (self.value(self._shift_units(items, directions, units, k, x), p_lin)
                            - self.value(self._shift_units(items, directions, units, k, -x), p_lin)) / (2.0 * x)
                g[k] += (4.0 * central(h / 2.0) - central(h)) / 3.0
            if k in self.explicit:                      # the loss names this parameter itself
                g[k] += self._explicit_partial(lambda q: self.value(items, q), p_lin, k, f0)
        return f0, g


# ---------------------------------------------------------------------------------------
# AMICI
# ---------------------------------------------------------------------------------------

_BUILD_THREAD_LOCKS = {}
_BUILD_THREAD_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _build_lock(directory):
    """Hold an exclusive lock on one model's build, across threads, processes and nodes (a Slurm array compiles the same model at once).

    Whoever gets the lock first compiles; the rest wait, then find the finished module and import it. The lock file
    sits beside the model directory, and the lock is released if the holder dies.

    It is a POSIX (``lockf``) lock, not ``flock``: on GPFS (Hyak's /mmfs1 and /gscratch) ``flock`` is only local to a node, so
    array tasks on different nodes all "held" it at once and built into the same directory (measured 2026-11-02: two
    nodes both acquired an exclusive flock within 0.4 s of each other, while ``lockf`` serialized all six tasks).
    A POSIX lock belongs to the process, so it does not stop a second thread of the same process; a per-directory thread lock
    does that.
    """
    os.makedirs(os.path.dirname(directory) or ".", exist_ok=True)
    with _BUILD_THREAD_LOCKS_GUARD:
        thread_lock = _BUILD_THREAD_LOCKS.setdefault(directory, threading.Lock())
    with thread_lock:
        if fcntl is None:                     # no POSIX locks (native Windows): the thread lock is all there is
            yield
            return
        with open(directory + ".lock", "a") as fh:
            fcntl.lockf(fh, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.lockf(fh, fcntl.LOCK_UN)


def _model_directory(cache_dir, name, tag):
    """Where a compiled model lives: the simulation's name made safe for a path and a shell, plus the model's hash.

    A name such as ``Aducanumab 1 mg/kg (Q4W)`` would otherwise nest a directory at the slash and break the build's
    shell command at the parentheses.
    """
    return os.path.join(cache_dir, re.sub(r"\W", "_", name) + f"_{tag}")


def _never_reached(param_names, lost_per_sim):
    """Fitted parameters that reach no plain parameter of the model in any simulation.

    A parameter one arm does not use (another drug's binding rate) is ``lost`` there by design; only one that no arm
    reaches would silently get a zero gradient.
    """
    return [p for p in param_names if lost_per_sim and all(p in lost for lost in lost_per_sim)]


def _tracked_window(settings):
    blocks = settings["simulation_blocks"]
    blocks = list(blocks.values()) if isinstance(blocks, dict) else list(blocks)
    tracked = [b for b in blocks if b.get("tracked", True)]
    if not tracked:
        raise AnalyticGradientError("no tracked simulation block")
    last = max(i for i, b in enumerate(blocks) if b.get("tracked", True))
    return (float(blocks[0]["start"]), float(tracked[0]["start"]), float(blocks[last]["end"]))


def _data_times(groups, replicates, models, sim, t_lo, t_hi):
    """Every finite data time the loss reads for *sim*, inside [t_lo, t_hi]."""
    from pyantigen.engine.Optimize import _resolve_obs_df
    times = set()
    for g in groups.values():
        for elem in g.get("loss_elements", []):
            sims = list(elem.get("simulations", [])) or [elem.get("simulation")]
            if sim not in sims:
                continue
            data_sim = elem.get("data_simulation") or sims[0]
            df_dict = models[data_sim]["df_dict"]
            lc_fn = elem.get("loss_config")
            lc = lc_fn(replicates[data_sim]) if callable(lc_fn) else lc_fn
            for obs in (lc or {}).get("observables", []):
                df = _resolve_obs_df(df_dict, obs)
                if df is None or obs["time_column"] not in df.columns:
                    continue
                t = np.asarray(df[obs["time_column"]], dtype=float)
                times.update(float(v) for v in t[np.isfinite(t) & (t >= t_lo) & (t <= t_hi)])
    return times


def freeze_stale_assignments(sbml, r, protected=(), varying=(), everything=False):
    """The SBML with the parameters RoadRunner leaves stale made plain, at the values RoadRunner holds.

    A parameter with an initial assignment that (through other symbols) reads one of *varying* -- the
    model parameters the fitted ones move -- keeps, in the engine's in-place simulation, the value it was
    given when the model was built; RoadRunner's ``reset()`` re-derives species but not parameters. The
    returned SBML has that initial assignment removed and the parameter set to ``r``'s value, so it is a
    plain parameter whose number comes from the engine's replay like any other.

    *protected*: the variables the schedule assigns, and whatever their initial values are built from, are
    left alone (:meth:`AmiciGradient`'s inputs handle those). Returns ``(sbml, [frozen names])``.

    *everything*: freeze every parameter that has an initial assignment (not only those that read a
    varying one). A model whose parameters are written as expressions of constants (``k = 9.56e-7*70``,
    ``V = V_base``) has them as initial assignments, and a project's ``Update_parameters`` or hooks may
    overwrite them on the RoadRunner model; AMICI would otherwise integrate the SBML's value, not the
    engine's. Frozen, each is a plain parameter at the value RoadRunner holds.

    Also under *everything*: a plain parameter (no initial assignment, no rule) that ``Update_parameters`` set on the
    RoadRunner model is given the value RoadRunner holds. ``r.getSBML()`` is the model as it was LOADED, so such a
    parameter -- a regional population set per replicate and fitted by nobody, say, which a species' initial assignment
    then reads -- would otherwise be integrated by AMICI at the SBML's default. (The parameters the fitted ones move do not
    need this, and baking in the value RoadRunner holds at a probe point would only change the compiled model's identity:
    AMICI is given their values on every evaluation.)
    """
    import libsbml
    doc = libsbml.readSBMLFromString(sbml)
    model = doc.getModel()
    formulas = {}
    for i in range(model.getNumInitialAssignments()):
        a = model.getInitialAssignment(i)
        formulas[a.getSymbol()] = libsbml.formulaToL3String(a.getMath())
    for i in range(model.getNumRules()):
        rule = model.getRule(i)
        if rule.isAssignment() or rule.isRate():
            formulas.setdefault("rule:" + rule.getVariable(), libsbml.formulaToL3String(rule.getMath()))

    def tokens(sym, seen):
        for src in (formulas.get(sym), formulas.get("rule:" + sym)):
            for tok in set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", src or "")):
                if tok not in seen:
                    seen.add(tok)
                    tokens(tok, seen)
        return seen

    keep = set()
    for v in protected:
        keep |= tokens(v, {v})
    frozen = []
    for sym in list(formulas):
        if sym.startswith("rule:") or sym in keep:
            continue
        p = model.getParameter(sym)
        if p is None or model.getRuleByVariable(sym) is not None:
            continue
        if everything or tokens(sym, set()) & set(varying):
            model.removeInitialAssignment(sym)
            p.setValue(float(r[sym]))
            p.setConstant(True)
            frozen.append(sym)
    if everything:
        for i in range(model.getNumParameters()):
            p = model.getParameter(i)
            sym = p.getId()
            if sym in keep or sym in frozen or sym in varying or model.getRuleByVariable(sym) is not None \
                    or model.getInitialAssignmentBySymbol(sym) is not None:
                continue            # (a parameter the fitted ones move gets its value from AMICI's inputs on every evaluation)
            try:
                held = float(r[sym])
            except Exception:
                continue
            if np.isfinite(held) and held != p.getValue():
                p.setValue(held)
                frozen.append(sym)
    return libsbml.writeSBMLToString(doc), frozen


class _SimGradient:
    """AMICI's view of one simulation: the compiled event-free model, the schedule, the parameter replay."""

    def __init__(self, name, replicate, model, models, groups, replicates, param_names, probes, *,
                 model_text, paths, cache_dir, rtol, atol, max_steps, verbose, seed=None, extra_times=(), retry=None):
        from pyantigen.engine.Optimize import OptRoadRunnerProxy, remember_parameter_baseline
        from pyantigen.engine.Scaled_amici import (AmiciSegmentRunner, _plain_parameters,
                                                   compile_model, concentration_outputs)
        from pyantigen.generate.scaled_time import scale_model
        from pyantigen.generate.TelluriumGen import TelluriumGen

        self.name, self.replicate = name, replicate
        self.extra_times = [float(t) for t in extra_times]
        settings = replicate["Solver_settings"](replicate)
        schedule = settings.get("schedule")
        r = model["r"]
        if schedule is None:
            schedule, r = self._schedule_from_events(model, replicate, param_names, model_text,
                                                     paths, OptRoadRunnerProxy, remember_parameter_baseline,
                                                     TelluriumGen)
        self.schedule = schedule
        self.r = r
        sbml = r.getSBML()
        # Parameters defined by an initial assignment from a fitted parameter (Q_glymph = Q_CSF) are not
        # re-derived by RoadRunner's reset(): the engine integrates with the value they had when the
        # model was built. Freeze them, so AMICI integrates the same model the objective does.
        plain0 = _plain_parameters(sbml)
        varying = set()
        for point in probes:
            _v, J0 = ParameterReplay(r, replicate, param_names, plain0).jacobian(point)
            varying |= {p for p, hit in zip(plain0, np.abs(J0).max(axis=1, initial=0.0) > 0) if hit}
        sbml, self.frozen = freeze_stale_assignments(
            sbml, r, protected=schedule.assigned_variables(), varying=varying, everything=True)
        plain = _plain_parameters(sbml)
        self.replay = ParameterReplay(r, replicate, param_names, plain)

        # which model parameters the fitted ones reach: the union over a few points of the box
        reach = np.zeros((len(plain), len(param_names)), dtype=bool)
        direct = None
        for point in probes:
            _values, J = self.replay.jacobian(point)
            reach |= np.abs(J) > 0
            found = self.replay.direct_species(point)
            if direct is not None and found != direct:
                raise AnalyticGradientError(
                    f"{name}: a hook sets species initial values that depend on the fitted parameters "
                    f"({sorted(set(found) | set(direct))}); their sensitivities are not available")
            direct = found
        # a parameter this arm does not use (another drug's binding rate) has a zero column here by design;
        # AmiciGradient raises if no arm at all is reached by it
        self.lost = [p for p, hit in zip(param_names, reach.any(axis=0)) if not hit]
        self.free = [p for p, used in zip(plain, reach.any(axis=1)) if used]

        columns = [c for c in replicate["Observed_species"](r) if c != "time"]
        columns = list(dict.fromkeys(columns))
        usable, dropped = self._usable_columns(r, sbml, columns)
        self.columns = ["time"] + usable
        self.dropped = dropped

        scaled = scale_model(sbml, schedule)
        self.scaled = scaled
        outputs = concentration_outputs(sbml, usable)
        tag = hashlib.sha256(json.dumps([scaled.sbml, sorted(self.free), list(outputs.items())],
                                        sort_keys=True).encode()).hexdigest()[:16]
        directory = _model_directory(cache_dir, name, tag)
        module_name = re.sub(r"\W", "_", f"pag_{name}_{tag}")
        import amici
        init = os.path.join(directory, module_name, "__init__.py")
        with _build_lock(directory):       # an __init__.py seen without the lock may belong to a build still in progress
            if os.path.exists(init):
                module = amici.import_model_module(module_name, directory)
            else:
                if verbose:
                    print(f"  [amici] compiling {name}: {len(self.free)} sensitivity parameter(s), "
                          f"{len(usable)} output(s) -> {directory}", flush=True)
                module = compile_model(scaled, module_name, directory, free=self.free, outputs=outputs)
        self.runner = AmiciSegmentRunner(scaled, schedule, module, states=[], rtol=rtol, atol=atol,
                                         max_steps=max_steps, retry=retry)
        self.t_start, self.t_first, self.t_last = _tracked_window(settings)
        self.seed = {str(k): float(v) for k, v in (seed or {}).items()}

    @staticmethod
    def _usable_columns(r, sbml, columns):
        import libsbml
        model = libsbml.readSBMLFromString(sbml).getModel()
        ids = ({s.getId() for s in model.getListOfSpecies()} | {p.getId() for p in model.getListOfParameters()}
               | {c.getId() for c in model.getListOfCompartments()})
        usable, dropped = [], []
        for c in columns:
            base = c[1:-1] if c.startswith("[") and c.endswith("]") else c
            (usable if base in ids else dropped).append(c)
        return usable, dropped

    @staticmethod
    def _schedule_from_events(model, replicate, param_names, model_text, paths, Proxy, remember,
                              TelluriumGen):
        """No schedule on the replicate: read it from the event text, and drop those events from the model."""
        events = model.get("events") or ""
        if not events.strip() and replicate.get("no_inputs"):
            # the replicate says nothing is clock-driven (an unperturbed natural-history run): an empty
            # schedule, and the model as it is, since there are no events to drop from it. Without that
            # statement an empty replicate is refused below -- it is more likely a forgotten schedule.
            r = model["r"]
            return InputSchedule({p: float(r[p]) for p in r.getGlobalParameterIds()}), r
        if "at (" not in events and "at(" not in events:
            raise AnalyticGradientError(
                "the replicate has no schedule and no events to read one from; "
                "an analytic gradient needs the inputs as an InputSchedule")
        r = model["r"]
        params = {p: float(r[p]) for p in r.getGlobalParameterIds()}
        try:
            schedule = InputSchedule.from_antimony_events(events, params, ignore_other=True)
        except ScheduleError as exc:
            raise AnalyticGradientError(f"cannot read the replicate's events as a schedule: {exc}") from exc
        rest = "\n".join(schedule.unparsed)
        r_free = TelluriumGen(model_text + "\n" + rest, paths)
        Proxy_r = Proxy(r_free, param_names)
        replicate["Update_parameters"](Proxy_r, replicate)
        remember(r_free)
        return schedule, r_free

    def tables(self, p_lin, groups, replicates, models):
        """``(SimTable, d/d theta array (rows, columns, K))`` at the data times, or raises."""
        p_lin = np.asarray(p_lin, dtype=float)
        values, J = self.replay.jacobian(p_lin)
        direct = self.replay.direct_species(p_lin)
        theta = dict(values)
        inputs = self._engine_initial_inputs(theta)
        times = _data_times(groups, replicates, models, self.name, self.t_first, self.t_last)
        # A data time AT the start of the window is the initial state: the first segment reads its own start (see
        # InputSchedule.plan). One before it cannot be (_data_times keeps only [t_first, t_last]).
        if any(t < self.t_start for t in times):
            raise AnalyticGradientError(
                f"{self.name}: the loss reads a data time before the start of the simulation ({self.t_start:g} h)")
        anchors = {self.t_last} | ({self.t_first} if self.t_first > self.t_start else set())
        # rows the loss reads that are not data times (a pre-dose baseline), asked for in the settings
        anchors |= {t for t in self.extra_times if self.t_start < t <= self.t_last}
        T = np.array(sorted(times | anchors))
        wrt = self.free
        Y, SY = self.runner.run_outputs(theta, self.t_start, self.t_last, list(T), wrt=wrt,
                                        initial={**self.seed, **(direct or {})} or None, initial_inputs=inputs)
        # d(output)/d(fitted) = d(output)/d(model parameter) . d(model parameter)/d(fitted)
        Jfree = J[[self.replay.plain.index(p) for p in wrt], :]
        data = np.zeros((len(T), len(self.columns)))
        data[:, 0] = T
        data[:, 1:] = Y
        D = np.zeros((len(T), len(self.columns), J.shape[1]))
        D[:, 1:, :] = np.einsum("tcm,mk->tck", SY, Jfree)
        return SimTable(self.columns, data), D

    def trajectory(self, times):
        """The outputs at *times* (all after the start of the window) with the model as the engine builds it.

        No fitted parameter: this is the simulation, not the objective; the seed (``analytic_gradient['seed']``) still applies. Returns an array
        ``(len(times), len(self.columns) - 1)``, the columns in ``self.columns[1:]``.
        """
        none = np.zeros(0)
        values = self.replay.values(none)
        direct = self.replay.direct_species(none)
        times = np.asarray(times, dtype=float)
        if (times <= self.t_start).any():
            raise AnalyticGradientError(
                f"{self.name}: AMICI reports the end of each segment, so a time at or before the start of the "
                f"simulation ({self.t_start:g} h) is not available")
        Y, _ = self.runner.run_outputs(dict(values), self.t_start, float(times.max()), list(times), wrt=[],
                                       initial={**self.seed, **(direct or {})} or None,
                                       initial_inputs=self._engine_initial_inputs(dict(values)))
        return Y

    def _engine_initial_inputs(self, theta):
        """What each assigned input holds before its first step, as the engine's model holds it.

        RoadRunner's ``reset()`` re-derives a species' initial assignment but not a parameter's, so an
        input such as ``Q_SN = Q_CSF*f_SN`` keeps the value it had at the start. Where that is the
        model's own expression evaluated now, the expression is used (and differentiated); where it is
        not, the engine's number is, with no sensitivity.
        """
        out = {}
        for v, expr in self.scaled.initial_inputs.items():
            engine = float(self.r[v])
            if isinstance(expr, str):
                try:
                    model_value = self.schedule.evaluate(expr, lambda n: float(theta.get(n, self.r[n])))
                except Exception:
                    model_value = None
                if model_value is not None and np.isclose(model_value, engine, rtol=1e-9, atol=1e-12):
                    out[v] = expr
                    continue
            out[v] = engine
        return out


class AmiciGradient:
    """``x (opt space) -> d NLL / d x`` for the engine's concentrated NLL, from AMICI sensitivities.

    Parameters are the objects ``run_optimization_from_groups`` builds: ``models`` (sim -> dict with the
    RoadRunner model ``"r"``, its ``"df_dict"`` and ``"events"``), ``replicates``, the spec's ``groups``,
    ``param_names``, ``scales`` ("log10" or linear, per parameter), and ``x0_lin`` / ``bounds_lin`` (the
    points at which the parameter reach is probed). *settings* is the ``analytic_gradient`` dict of the
    spec's ``optimizer_kwargs``: ``rtol``, ``atol``, ``max_steps`` (AMICI), ``rel_step`` (the loss
    differentiation), ``cache_dir`` (compiled models), ``seed`` (``{state: amount}`` given to AMICI at the
    start of every run on top of the model's own initial state; see the module's docs), ``retry`` (``[[rtol, atol], ...]``: tolerances a failed AMICI segment is run again under, in order, until one succeeds), ``priors`` (``[{"param", "ref", "sigma", "side"}]``: log10 penalties on fitted parameters, see :meth:`_parse_priors`), ``extra_times``
    (``{simulation: [t, ...]}``: output rows the loss reads that are not data times, such as a pre-dose
    baseline; AMICI's table has rows only at data times and these).
    """

    DEFAULTS = {"provider": "amici", "rtol": 1e-10, "atol": 1e-12, "max_steps": 2_000_000,
                "rel_step": 1e-5, "cache_dir": None, "verbose": True, "seed": None,
                "extra_times": None, "retry": None, "priors": None}

    @classmethod
    def resolve(cls, optimizer_kwargs):
        given = (optimizer_kwargs or {}).get("analytic_gradient")
        if not given:
            return None
        if given is True:
            given = {}
        unknown = sorted(set(given) - set(cls.DEFAULTS))
        if unknown:
            raise ValueError(f"unknown analytic_gradient key(s) {unknown}; expected a subset of "
                             f"{sorted(cls.DEFAULTS)}")
        out = {**cls.DEFAULTS, **given}
        if out["provider"] != "amici":
            raise ValueError("analytic_gradient['provider'] must be 'amici'")
        return out

    def __init__(self, settings, *, models, replicates, groups, param_names, scales, x0_lin, bounds_lin,
                 model_text, paths):
        self.settings = settings
        self.models, self.replicates, self.groups = models, replicates, groups
        self.param_names, self.scales = list(param_names), list(scales)
        self.loss = LossGradient(groups, replicates, param_names, rel_step=settings["rel_step"])
        cache_dir = settings["cache_dir"] or os.path.join(
            (paths or {}).get("plot_path") or tempfile.gettempdir(), "amici_models")
        os.makedirs(cache_dir, exist_ok=True)
        probes = self._probe_points(x0_lin, bounds_lin)
        self.sims = {}
        for name, rep in replicates.items():
            self.sims[name] = _SimGradient(
                name, rep, models[name], models, groups, replicates, param_names, probes,
                model_text=model_text, paths=paths, cache_dir=cache_dir, rtol=settings["rtol"],
                atol=settings["atol"], max_steps=settings["max_steps"], verbose=settings["verbose"],
                seed=settings["seed"], extra_times=(settings["extra_times"] or {}).get(name, ()), retry=settings["retry"])
        never = _never_reached(self.param_names, [sim.lost for sim in self.sims.values()])
        if never:
            raise AnalyticGradientError(
                f"fitted parameter(s) {never} reach no plain parameter of the model in any arm (a species "
                "initial value, or a parameter with an initial assignment?): the analytic gradient "
                "would be zero for them")
        self.n_calls = 0
        self.priors = self._parse_priors(settings.get("priors"), self.param_names)
        self.last_prior_nll = 0.0

    @staticmethod
    def _parse_priors(priors, param_names):
        """``[(index, ref, sigma, side)]`` from the settings' ``priors``: ``{"param", "ref" (linear), "sigma" (decades), "side"}``.

        The residual ``max(0, log10(ref / p)) / sigma`` for ``side="below"`` (a penalty on p under ref), ``max(0, log10(p / ref)) / sigma`` for
        ``"above"``, ``log10(p / ref) / sigma`` for ``"both"`` (a log-normal prior). It joins the data's residual vector, so the loss is
        ``const + 0.5 r.r`` with the prior's ``0.5 (log10 ratio / sigma)^2`` in it, and the FIM, the Wald intervals and the profile include it.
        The value is C1 at ``ref`` for the one-sided kinds. A prior on a parameter that is not fitted is an error.
        """
        out = []
        for item in (priors or []):
            name, ref, sigma = item["param"], float(item["ref"]), float(item["sigma"])
            side = item.get("side", "both")
            if side not in ("below", "above", "both"):
                raise AnalyticGradientError(f"prior on {name}: side must be below, above or both, not {side!r}")
            if name not in param_names:
                raise AnalyticGradientError(f"prior on {name}: not a fitted parameter ({list(param_names)})")
            if not (ref > 0 and sigma > 0):
                raise AnalyticGradientError(f"prior on {name}: ref and sigma must be positive")
            out.append((list(param_names).index(name), ref, sigma, side))
        return out

    def prior_terms(self, p_lin):
        """``(rp, Jp)``: the priors' residuals and their derivatives with respect to the linear parameters (``Jp`` is len(priors) x K)."""
        K = len(self.param_names)
        rp, Jp = np.zeros(len(self.priors)), np.zeros((len(self.priors), K))
        for m, (k, ref, sigma, side) in enumerate(self.priors):
            p = float(p_lin[k])
            if not p > 0:
                raise AnalyticGradientError(f"prior on {self.param_names[k]}: the parameter is {p}, not positive")
            x = np.log10(p / ref)                              # > 0 above ref
            if side == "below":
                x = max(0.0, -x) * -1.0
            elif side == "above":
                x = max(0.0, x)
            rp[m] = abs(x) / sigma if side != "both" else x / sigma
            active = (x != 0.0) if side != "both" else True
            sign = (-1.0 if side == "below" else 1.0)
            Jp[m, k] = (sign / (p * np.log(10.0) * sigma)) if active else 0.0
        self.last_prior_nll = 0.5 * float(rp @ rp)
        return rp, Jp

    @staticmethod
    def _probe_points(x0_lin, bounds_lin):
        """x0 and the two corners of the box: where the fitted parameters' reach is read off."""
        x0 = np.asarray(x0_lin, dtype=float)
        points = [x0]
        if bounds_lin:
            lo = np.array([x0[i] if b is None or b[0] is None else b[0] for i, b in enumerate(bounds_lin)], float)
            hi = np.array([x0[i] if b is None or b[1] is None else b[1] for i, b in enumerate(bounds_lin)], float)
            points += [p for p in (lo, hi) if np.all(np.isfinite(p)) and not np.array_equal(p, x0)]
        return points

    def freeze_sigmas(self, sigmas):
        """Differentiate the loss a profile point minimizes: floored blocks keep the sigma they had at the fit.

        *sigmas* is what ``Optimize.block_sigmas`` returns for the fit (``{(block, observable): sigma_used}``),
        the same dictionary the profile passes to ``evaluate_nll_fixed(frozen_sigmas=...)``. A frozen block is
        scored with that sigma instead of ``min(sigma_hat, floor)`` at the current point, which changes its
        derivative (the sigma no longer moves with the parameters). ``None`` releases them.
        """
        self.loss.frozen_sigmas = dict(sigmas) if sigmas else None

    def linear_gradient(self, p_lin):
        """``(nll, d nll / d p_lin)`` evaluated on AMICI's own simulation."""
        items, dirs = {}, {}
        for name, sim in self.sims.items():
            table, D = sim.tables(p_lin, self.groups, self.replicates, self.models)
            items[name] = {"results": table, "data": self.models[name]["df_dict"],
                           "replicate": self.replicates[name]}
            dirs[name] = D
        self.n_calls += 1
        nll, g = self.loss.gradient(items, dirs, p_lin)
        if self.priors:
            rp, Jp = self.prior_terms(p_lin)
            nll, g = nll + 0.5 * float(rp @ rp), g + Jp.T @ rp
        return nll, g

    def linear_residual_jacobian(self, p_lin):
        """``(r, J, const)`` on AMICI's own simulation; see :meth:`LossGradient.residual_jacobian`."""
        items, dirs = {}, {}
        for name, sim in self.sims.items():
            table, D = sim.tables(p_lin, self.groups, self.replicates, self.models)
            items[name] = {"results": table, "data": self.models[name]["df_dict"],
                           "replicate": self.replicates[name]}
            dirs[name] = D
        self.n_calls += 1
        r, J, const = self.loss.residual_jacobian(items, dirs, p_lin)
        if self.priors:
            rp, Jp = self.prior_terms(p_lin)
            r, J = np.concatenate([r, rp]), np.vstack([J, Jp])
        return r, J, const

    @contextlib.contextmanager
    def tolerances_scaled(self, factor):
        """Run the AMICI solvers of every simulation at *factor* times their relative and absolute tolerances (state and sensitivities), then restore them.

        ``0.01`` tightens them a hundredfold. The profile uses it to measure how much of ``J'r`` is solver error (``Profile_amici.measure_noise_floor``)
        and to polish a point that stopped near the gradient's noise floor.
        """
        runners = [(sim.runner, sim.runner._tol) for sim in self.sims.values()]
        try:
            for rn, (rtol, atol) in runners:
                rn._tol = (rtol * factor, atol * factor)      # a retry that fails over returns to these, not to the unscaled ones
                rn._set_tolerances(*rn._tol)
            yield
        finally:
            for rn, tol in runners:
                rn._tol = tol
                rn._set_tolerances(*tol)

    def __call__(self, x_opt):
        """The gradient in optimizer space (a log10 parameter's derivative is taken with respect to log10 p)."""
        from pyantigen.engine.Optimize import _to_linear
        x_opt = np.asarray(x_opt, dtype=float)
        p_lin = _to_linear(x_opt, self.scales)
        _f, g_lin = self.linear_gradient(p_lin)
        chain = np.array([np.log(10.0) * p if sc == "log10" else 1.0
                          for p, sc in zip(np.asarray(p_lin, dtype=float), self.scales)])
        return g_lin * chain


def build_gradient(optimizer_kwargs, **context):
    """The analytic gradient asked for in *optimizer_kwargs*, or None.

    A problem it cannot handle raises ``Multistart.GradientSetupError`` here, before any fit.
    """
    settings = AmiciGradient.resolve(optimizer_kwargs)
    if settings is None:
        return None
    try:
        return AmiciGradient(settings, **context)
    except GradientSetupError:
        raise
    except (AnalyticGradientError, ScheduleError, ImportError, KeyError) as exc:
        raise GradientSetupError(f"analytic_gradient cannot be set up: {type(exc).__name__}: {exc}") from exc
