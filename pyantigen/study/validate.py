"""Checks a design before anything is simulated.

Every item here is a mistake the 1.x layout let through silently: a
misspelled metadata key falling back to a default, an arm that selected no
data, a composite whose control arm was missing, a duplicate arm integrated
twice.

``validate(study)`` returns a list of Problem(level, message); level is
"error" or "warning". ``validate(..., raise_on_error=True)`` raises on the
first error-level problem list instead.
"""
from dataclasses import dataclass

from .design import _matches
from .refs import from_ref


@dataclass
class Problem:
    level: str
    message: str

    def __str__(self):
        return f"[{self.level}] {self.message}"


class DesignError(ValueError):
    pass


def validate(study, data_path=None, remarks=None, raise_on_error=False):
    probs = []

    def err(msg):
        probs.append(Problem("error", msg))

    def warn(msg):
        probs.append(Problem("warning", msg))

    known = study._attribute_names()

    # protocols resolve to real functions; settings do not shadow covariates
    covariate_names = set()
    for s in study.subjects.values():
        covariate_names |= set(s.covariates)
    for name, p in study.protocols.items():
        for part in ("events", "solver", "observed") + tuple(
                k for k in p._HOOKS if getattr(p, k) is not None):
            try:
                fn = from_ref(getattr(p, part))
                if not callable(fn):
                    err(f"protocol {name!r}: {part} is not callable")
            except Exception as exc:
                err(f"protocol {name!r}: {part} cannot be imported ({exc})")
        clash = set(p.settings) & covariate_names
        if clash:
            err(f"protocol {name!r}: settings {sorted(clash)} are also subject covariates; "
                "keep each attribute in one place")

    # assays select something, and their data exist
    try:
        scoring = study.scoring()
    except (ValueError, KeyError) as exc:
        err(str(exc))
        scoring = {sid: [] for sid in study.simulations}
    scored_by_assay = {a: [] for a in study.assays}
    for sid, entries in scoring.items():
        for assay, partner in entries:
            scored_by_assay[assay.name].append(study.simulations[sid])

    for aname, assay in study.assays.items():
        where = f"assay {aname!r}"
        for m in assay.observables:
            bad = set(m.only) - known
            if bad:
                err(f"{where}/{m.name}: 'only' uses unknown attribute(s) {sorted(bad)}")
            pool = m.noise.pool
            if isinstance(pool, (list, tuple)):
                badp = set(pool) - set(study.factors) - {"subject"}
                if badp:
                    err(f"{where}/{m.name}: pool names unknown factor(s) {sorted(badp)}")
        sims = scored_by_assay[aname]
        if not sims:
            err(f"{where}: scores no simulations")
            continue
        scored_any = False
        for sim in sims:
            attrs = study.attributes(sim)
            inputs = None
            for m in assay.observables:
                if not all(_matches(attrs.get(k), v) for k, v in m.only.items()):
                    continue
                scored_any = True
                if data_path is None:
                    continue
                if m.data.input is not None and inputs is None:
                    loader = study.protocols[sim.protocol].hooks()[0]
                    if loader is None:
                        err(f"{where}/{m.name}: input={m.data.input!r} but protocol "
                            f"{sim.protocol!r} has no data loader")
                        continue
                    inputs = loader(dict(attrs, Label=sim.id), data_path)
                try:
                    n = len(m.data.rows_for(attrs, data_path, inputs=inputs))
                except (KeyError, FileNotFoundError, OSError) as exc:
                    err(f"{where}/{m.name} on {sim.id!r}: {exc}")
                    continue
                if n == 0:
                    err(f"{where}/{m.name} on {sim.id!r}: no data rows")
        if not scored_any:
            err(f"{where}: every simulation is excluded by the observables' 'only' filters")

    # parameters
    fitted = []
    for name, p in study.params.items():
        if p.by and p.by not in study.factors:
            err(f"param {name!r}: by={p.by!r} is not a factor")
            continue
        try:
            fitted += p.fitted_names(study)
        except KeyError as exc:
            err(f"param {name!r}: {exc}")
        if p.estimate and p.bounds is not None and not isinstance(p.bounds, dict) \
                and not isinstance(p.x0, dict):
            lo, hi = p.bounds
            if not (lo <= p.x0 <= hi):
                err(f"param {name!r}: x0={p.x0} outside bounds {p.bounds}")
    dup = {n for n in fitted if fitted.count(n) > 1}
    if dup:
        err(f"fitted parameter names collide: {sorted(dup)}")

    for r in study.rules:
        bad = set(r.when) - known
        if bad:
            err(f"rule on {r.target!r}: unknown attribute(s) {sorted(bad)}")

    # remarks
    if study.remarks:
        if remarks is None:
            warn(f"study cites remarks {study.remarks} but no remarks file was given to check them")
        else:
            missing = [r for r in study.remarks if r not in remarks.ids]
            if missing:
                err(f"study cites remark(s) not in {remarks.path}: {missing}")

    # identical simulations: the same thing integrated twice
    seen = {}
    for sim in study.simulations.values():
        a = study.attributes(sim)
        sig = (sim.protocol, tuple(sorted((k, repr(v)) for k, v in a.items()
                                          if k not in ("subject", "simulation"))))
        if sig in seen:
            warn(f"simulations {seen[sig]!r} and {sim.id!r} are the same simulation "
                 "(same protocol, levels, covariates and settings); score both from one")
        else:
            seen[sig] = sim.id

    if raise_on_error:
        errors = [p for p in probs if p.level == "error"]
        if errors:
            raise DesignError("\n".join(str(p) for p in errors))
    return probs
