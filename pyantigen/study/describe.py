"""Everything about each simulation, in one place: the per-simulation view.

A Study is written by concept -- what a dose level implies once on the
factor, what an assay needs once on the assay -- which keeps large designs
small but scatters what one simulation is across the file. ``describe``
reassembles it: for every simulation, its subject, protocol, levels and what
they imply, covariates, protocol settings, the functions that build and run
it, the parameters it receives, and every assay that scores it with the data
table, noise model and contrast partner. It is computed from the design, so
it cannot drift from it.

    print(describe(study))                  # the paper: datasets on each simulation
    print(describe(study, optimization))    # ... and what one fit scores
"""
from .assay import _eval_arith
from .design import _matches
from .params import by_name, resolve
from .refs import to_ref


def _fmt(d):
    return ", ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}" for k, v in d.items())


def _ref(fn):
    ref = to_ref(fn) if fn is not None else None
    return ref.split(":", 1)[1] if ref and ":" in ref else ref


def _obs(m, attrs):
    o = m.model
    if o.columns is not None:
        try:
            t0 = f" = {_eval_arith(o.baseline_at, attrs)!r} h"
        except Exception as exc:        # shown, not raised: validate() reports it
            t0 = f" (cannot evaluate: {exc})"
        unit = "%" if o.scale == 100.0 else ("fraction" if o.scale == 1.0 else f"x{o.scale}")
        return (f"{o.name}: sum({', '.join(o.columns)}) as {unit} of its value at "
                f"{o.baseline_at}{t0}")
    return o.name if o.name == o.expr else f"{o.name}: {o.engine_form()}"


def _noise(m):
    n = m.noise
    parts = []
    if n.sigma is not None:
        parts.append(f"sigma fixed {n.sigma:g}")
    if n.sd_column:
        parts.append("sigma from data sd column")
    if n.floor_from_data:
        parts.append("sigma floored from data")
    if not parts:
        parts.append("sigma profiled")
    if n.pool == "all":
        parts.append("pooled over all simulations")
    elif n.pool:
        parts.append(f"pooled over {', '.join(n.pool)}")
    else:
        parts.append("one sigma per simulation")
    return "; ".join(parts)


def _data(m, attrs):
    from .assay import _fill
    src = m.data
    where = f" where {_fmt({k: _fill(v, attrs) for k, v in src.where.items()})}" if src.where else ""
    value = src.value if isinstance(src.value, str) else "+".join(src.value)
    if src.input is not None:
        return f"protocol table {_fill(src.input, attrs)!r} [{src.time} -> {value}]{where}"
    return f"file {_fill(src.file, attrs)} [{src.time} -> {value}]{where}"


def describe(study, optimization=None):
    """The per-simulation view of *study*, as text.

    Alone, it lists the datasets on each simulation (the paper). Given an
    Optimization, it shows what THAT fit scores, which simulations it only
    simulates, and which it does not need.
    """
    from .serialize import fingerprint
    lines = []
    opt = optimization
    if opt is not None:
        entries_by_sim = opt.scoring(study)
        needed = opt.simulated(study)
        params = list(opt.params)
    else:
        entries_by_sim = {sid: [(a, partner, None) for a, partner in e]
                          for sid, e in study.scoring().items()}
        needed = None
        params = []
    serves = {}
    for sid, entries in entries_by_sim.items():
        for assay, partner, _only in entries:
            if partner is not None:
                serves.setdefault(partner.id, []).append(f"{sid} ({assay.name})")

    lines.append(f"{study.name}: {len(study.simulations)} simulation(s), "
                 f"{len(study.assays)} dataset(s), fingerprint {fingerprint(study)[:12]}")
    if study.remarks:
        lines.append(f"remarks: {', '.join(study.remarks)}")
    if opt is not None:
        others = [st.name for st in opt.studies if st is not study]
        lines.append(f"optimization {opt.name}: fits {len(params)} parameter(s)"
                     + (f"; also uses {', '.join(others)}" if others else ""))
        for p in params:
            lines.append(f"  {p.name:<24} x0 {p.x0!r:<22} bounds {p.bounds}  scale {p.scale}"
                         + (f"  one per {p.by}" if p.by else ""))
    n_global = sum(1 for p in params if p.by is None)
    lines.append("")

    width = max((len(s) for s in study.simulations), default=10) + 2
    for sim in study.simulations.values():
        subj = study.subjects[sim.subject]
        proto = study.protocols[sim.protocol]
        attrs = study.attributes(sim)
        lines.append(f"{sim.id:<{width}}subject {sim.subject} ({subj.kind})   protocol {sim.protocol}"
                     + (f"   period {sim.period}" if sim.period is not None else ""))
        if needed is not None and sim.id not in needed:
            lines.append("  (not simulated by this optimization)")
            lines.append("")
            continue
        for fac, lev in sim.levels:
            implied = {k: v for k, v in study.factors[fac].levels[lev].items() if k != "params"}
            src = " (subject)" if fac in subj.levels else ""
            lines.append(f"  level       {fac}={lev}{src}" + (f"  ->  {_fmt(implied)}" if implied else ""))
        if subj.covariates:
            lines.append(f"  covariates  {_fmt(subj.covariates)}")
        if proto.settings:
            lines.append(f"  settings    {_fmt(proto.settings)}  (protocol)")
        data, upd, upd_opt = proto.hooks()
        events, solver, observed = proto.resolved()
        lines.append(f"  functions   events {_ref(events)} . solver {_ref(solver)} . observed {_ref(observed)}")
        extra = [f"{k} {_ref(v)}" for k, v in (("data", data), ("update_parameters", upd),
                                                ("update_opt_parameters", upd_opt)) if v]
        if extra:
            lines.append("              " + " . ".join(extra))
        fixed = resolve(study, sim, None)
        rules = [r for r in study.rules if r.applies(attrs)]
        rule_targets = {r.target for r in rules}
        lv = sim.level_dict
        fit_names = {p.name for p in params if p.by is None or p.by in lv}
        fixed_only = [f"{k}={v}" + (" (the fit overrides)" if k in fit_names else "")
                      for k, v in fixed.items() if k not in rule_targets]
        if fixed_only:
            lines.append(f"  fixed       {', '.join(fixed_only)}")
        if rules:
            lines.append("  rules       " + ", ".join(f"{r.target}={r.value} (when {_fmt(r.when)})"
                                                     for r in rules))
        per_level = [f"{p.name} <- {by_name(p.name, p.by, lv[p.by])}"
                     for p in params if p.by is not None and p.by in lv]
        if n_global or per_level:
            parts = ([f"{n_global} global (above)"] if n_global else []) + per_level
            lines.append(f"  {'fitted':<11} {', '.join(parts)}")

        label = "scored by" if opt is not None else "datasets"
        shown = False
        for assay, partner, only_obs in entries_by_sim[sim.id]:
            for m in assay.observables:
                if only_obs is not None and m.name not in only_obs:
                    continue
                if not all(_matches(attrs.get(k), v) for k, v in m.only.items()):
                    continue
                tag = label if not shown else ""
                shown = True
                vs = ""
                if partner is not None:
                    vs = f"  vs {partner.id} ({assay.contrast.op}, pairing={assay.contrast.pairing})"
                norm = f"  normalized to {m.normalize}" if m.normalize else ""
                lines.append(f"  {tag:<11} {assay.name}.{m.name}{vs}{norm}")
                lines.append(f"  {'':<13}model  {_obs(m, attrs)}")
                lines.append(f"  {'':<13}data   {_data(m, attrs)}")
                lines.append(f"  {'':<13}noise  {_noise(m)}")
        if not shown:
            plots = opt is not None and sim.id not in serves
            lines.append(f"  {label:<11} nothing" + (" (simulated for plots)" if plots else ""))
        if sim.id in serves:
            lines.append(f"  control for {', '.join(serves[sim.id])}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"
