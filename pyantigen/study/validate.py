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

    # protocols resolve to real functions
    for name, p in study.protocols.items():
        for part in ("events", "solver", "observed"):
            try:
                fn = from_ref(getattr(p, part))
                if not callable(fn):
                    err(f"protocol {name!r}: {part} is not callable")
            except Exception as exc:
                err(f"protocol {name!r}: {part} cannot be imported ({exc})")

    # measurements select something, and their data exist
    for i, meas in enumerate(study.measurements):
        where = f"measurement {i} ({meas.assay})"
        if meas.assay not in study.assays:
            err(f"{where}: unknown assay")
            continue
        assay = study.assays[meas.assay]
        for m in assay.observables:
            bad = set(m.only) - known
            if bad:
                err(f"{where}/{m.name}: 'only' uses unknown attribute(s) {sorted(bad)}")
            pool = m.noise.pool
            if isinstance(pool, (list, tuple)):
                badp = set(pool) - set(study.factors) - {"subject"}
                if badp:
                    err(f"{where}/{m.name}: pool names unknown factor(s) {sorted(badp)}")
        if meas.contrast:
            if meas.contrast not in study.contrasts:
                err(f"{where}: unknown contrast {meas.contrast!r}")
                continue
            try:
                occs = [n for n, _ in study.contrasts[meas.contrast].pairs(study)]
            except (ValueError, KeyError) as exc:
                err(f"{where}: {exc}")
                continue
        else:
            try:
                occs = study.select(**meas.on) if meas.on else list(study.occasions.values())
            except KeyError as exc:
                err(f"{where}: {exc}")
                continue
            if not occs:
                err(f"{where}: selects no occasions")
        scored_any = False
        for occ in occs:
            attrs = study.attributes(occ)
            for m in assay.observables:
                if not all(_matches(attrs.get(k), v) for k, v in m.only.items()):
                    continue
                scored_any = True
                if data_path is None:
                    continue
                try:
                    n = len(m.data.rows_for(attrs, data_path))
                except (KeyError, FileNotFoundError, OSError) as exc:
                    err(f"{where}/{m.name} on {occ.id!r}: {exc}")
                    continue
                if n == 0:
                    err(f"{where}/{m.name} on {occ.id!r}: no data rows")
        if occs and not scored_any:
            err(f"{where}: every occasion is excluded by the observables' 'only' filters")

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

    # identical occasions: the same simulation run twice
    seen = {}
    for occ in study.occasions.values():
        a = study.attributes(occ)
        sig = (occ.protocol, tuple(sorted((k, repr(v)) for k, v in a.items()
                                          if k not in ("subject", "occasion"))),
               tuple(sorted(study.subjects[occ.subject].covariates.items())))
        if sig in seen:
            warn(f"occasions {seen[sig]!r} and {occ.id!r} are the same simulation "
                 "(same protocol, levels and covariates); score both from one occasion")
        else:
            seen[sig] = occ.id

    if raise_on_error:
        errors = [p for p in probs if p.level == "error"]
        if errors:
            raise DesignError("\n".join(str(p) for p in errors))
    return probs
