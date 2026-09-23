"""Who was studied, what was done to them, and when.

A Study is built from four kinds of thing:

  Factor    a named variable of the design with a few levels -- dose, drug,
            cell line, age decade. Everything a level implies (a dose in nmol,
            an event delay, a cell line's 122 expression values) is written
            ONCE, on the level.
  Subject   one individual, or one cohort when the data are cohort means. It
            carries covariates and may fix the levels of between-subject
            factors (a cohort's amyloid status, a patient's drug arm).
  Protocol  what was done: the event generator, the solver window, which
            species are recorded.
  Occasion  one subject under one protocol at one assignment of the remaining
            factor levels, optionally in a numbered period. An occasion is
            the unit that is simulated. An "arm" is only a selection of
            occasions, not a stored object.

Why this shape, and not a table with one row per condition: a condition
table repeats every attribute of every factor on every row, so its size is the
PRODUCT of the factors. Froehlich 2018's PEtab condition table is 9,570 rows x
147 columns (17 MB) because each row copies its cell line's 122 expression
values; the same information is 290 cell-line levels x 122 values plus 1,105
treatment levels x 23 values, about 23 times smaller. Here the size of a
design is the SUM of its factor tables plus one short reference per occasion.

Pairing is built in from the start. Every occasion names its subject, so a
contrast (drug over vehicle) can be formed within each subject -- the right
operation for a crossover, where each animal is its own control -- and
reduces to the cohort-mean case when the "subject" is a cohort.
"""
from dataclasses import dataclass, field
from itertools import product

from .refs import from_ref, to_ref

# Keys an occasion's attribute dict always has; factor and covariate names may
# not shadow them.
RESERVED = ("subject", "protocol", "period", "occasion")


@dataclass
class Factor:
    name: str
    # level key -> attributes of that level. The reserved attribute "params"
    # holds model parameter values this level fixes ({name: value}); every
    # other attribute is a plain value the protocol, data and assays may read.
    levels: dict

    def __post_init__(self):
        if self.name in RESERVED:
            raise ValueError(f"factor name {self.name!r} is reserved")
        if not self.levels:
            raise ValueError(f"factor {self.name!r} has no levels")
        table = self.levels.get("table")
        if isinstance(table, dict) and "columns" in table:
            raise ValueError(f"factor {self.name!r}: 'table' is reserved for the JSON table form")
        self.levels = {str(k): dict(v or {}) for k, v in self.levels.items()}


@dataclass
class Subject:
    id: str
    kind: str = "individual"          # "individual" | "cohort"
    covariates: dict = field(default_factory=dict)
    # Between-subject factor levels this subject carries, {factor: level}.
    levels: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in ("individual", "cohort"):
            raise ValueError(f"subject {self.id!r}: kind must be 'individual' or 'cohort'")
        clash = set(self.covariates) & set(RESERVED)
        if clash:
            raise ValueError(f"subject {self.id!r}: covariates {sorted(clash)} are reserved names")
        self.levels = {k: str(v) for k, v in self.levels.items()}


@dataclass
class Protocol:
    """What was done, as references to project functions.

    events(replicate, df_dict, r_ic=None) -> Antimony event text
    solver(replicate) -> solver settings dict
    observed(r) -> list of recorded species

    Optional:
    data(replicate, data_path) -> {name: table}: the protocol's inputs (a
        measured drug curve the events write in, a tracer forcing) and any
        prepared tables a DataSource(input=...) scores. Its output is also
        what ``events`` receives as df_dict.
    update_parameters(r, replicate) / update_opt_parameters(r, replicate,
        parameters): model adjustments not yet expressible as Params or
        Rules (a species anatomy overlay, say). They run AFTER resolve()'s
        values are applied, at the same two points 1.x ran them.
    """
    name: str
    events: object
    solver: object
    observed: object
    data: object = None
    update_parameters: object = None
    update_opt_parameters: object = None

    _OPTIONAL = ("data", "update_parameters", "update_opt_parameters")

    def to_json(self):
        d = {"events": to_ref(self.events), "solver": to_ref(self.solver),
             "observed": to_ref(self.observed)}
        for k in self._OPTIONAL:
            if getattr(self, k) is not None:
                d[k] = to_ref(getattr(self, k))
        return d

    @classmethod
    def from_json(cls, name, d):
        return cls(name, d["events"], d["solver"], d["observed"],
                   *(d.get(k) for k in cls._OPTIONAL))

    def resolved(self):
        return (from_ref(self.events), from_ref(self.solver), from_ref(self.observed))

    def hooks(self):
        """(data, update_parameters, update_opt_parameters), imported."""
        return tuple(from_ref(getattr(self, k)) for k in self._OPTIONAL)


@dataclass(frozen=True)
class Occasion:
    id: str
    subject: str
    protocol: str
    levels: tuple            # sorted ((factor, level), ...), the full assignment
    period: object = None

    @property
    def level_dict(self):
        return dict(self.levels)


class Study:
    """A design: factors, subjects, protocols and the occasions built from them.

    Measurements, contrasts and parameters are attached by the methods in
    ``pyantigen.study.assay`` and ``pyantigen.study.params``; see
    ``Study.measure``, ``Study.contrast`` and ``Study.param``.
    """

    def __init__(self, name, remarks=None):
        self.name = name
        self.remarks = list(remarks or [])
        self.factors = {}
        self.subjects = {}
        self.protocols = {}
        self.occasions = {}          # id -> Occasion, in creation order
        self._occasion_records = []  # compact generator records, for JSON
        self.assays = {}
        self.measurements = []
        self.contrasts = {}
        self.params = {}
        self.rules = []

    # --- building --------------------------------------------------------
    def factor(self, name, levels):
        if name in self.factors:
            raise ValueError(f"factor {name!r} already defined")
        f = Factor(name, levels)
        self.factors[name] = f
        return f

    def subject(self, id, kind="individual", covariates=None, **levels):
        if id in self.subjects:
            raise ValueError(f"subject {id!r} already defined")
        for fac, lev in levels.items():
            self._check_level(fac, lev, f"subject {id!r}")
        s = Subject(id, kind, dict(covariates or {}), levels)
        self.subjects[id] = s
        return s

    def protocol(self, name, events, solver, observed, data=None,
                 update_parameters=None, update_opt_parameters=None):
        if name in self.protocols:
            raise ValueError(f"protocol {name!r} already defined")
        p = Protocol(name, events, solver, observed, data,
                     update_parameters, update_opt_parameters)
        self.protocols[name] = p
        return p

    def occasion(self, subject, protocol, period=None, id=None, **levels):
        """Add one occasion. Returns it."""
        occ = self._make_occasion(subject, protocol, period, id, levels)
        rec = [subject, protocol, dict(levels)]
        extra = {k: v for k, v in (("period", period), ("id", id)) if v is not None}
        if extra:
            rec.append(extra)
        self._occasion_records.append(rec)
        return occ

    def cross(self, subjects, protocol, **factor_levels):
        """Add every combination of subjects x the listed factor levels.

        factor_levels maps a factor name to a list of level keys, or to "*"
        for all of that factor's levels. Stored in the JSON as this one call,
        not as its expansion.
        """
        subjects = list(subjects)
        spec = {}
        for fac, levs in factor_levels.items():
            if fac not in self.factors:
                raise KeyError(f"cross: unknown factor {fac!r}")
            spec[fac] = (list(self.factors[fac].levels) if levs == "*"
                         else [str(x) for x in levs])
        out = []
        names = list(spec)
        for s in subjects:
            for combo in product(*(spec[n] for n in names)):
                out.append(self._make_occasion(s, protocol, None, None, dict(zip(names, combo))))
        self._occasion_records.append({"cross": {
            "subjects": subjects, "protocol": protocol,
            "factors": {k: ("*" if factor_levels[k] == "*" else spec[k]) for k in names}}})
        return out

    # --- what is measured, and what is estimated -------------------------
    def assay(self, name, *observables):
        """Register an assay: observables measured together (Measured objects)."""
        from .assay import Assay
        if name in self.assays:
            raise ValueError(f"assay {name!r} already defined")
        a = Assay(name, list(observables))
        self.assays[name] = a
        return a

    def contrast(self, name, op, numerator, denominator, pairing="subject", match=()):
        from .assay import Contrast
        if name in self.contrasts:
            raise ValueError(f"contrast {name!r} already defined")
        c = Contrast(name, op, dict(numerator), dict(denominator), pairing, list(match))
        self.contrasts[name] = c
        return c

    def measure(self, assay, on=None, contrast=None):
        """Score *assay* on the occasions selected by *on*, or on *contrast*."""
        from .assay import Measurement
        if assay not in self.assays:
            raise KeyError(f"measure: unknown assay {assay!r}")
        if contrast is not None and contrast not in self.contrasts:
            raise KeyError(f"measure: unknown contrast {contrast!r}")
        if contrast is not None and on is not None:
            raise ValueError("measure: give on= or contrast=, not both")
        m = Measurement(assay, dict(on) if on else None, contrast)
        self.measurements.append(m)
        return m

    def param(self, name, **kw):
        from .params import Param
        if name in self.params:
            raise ValueError(f"param {name!r} already defined")
        if kw.get("by") and kw["by"] not in self.factors:
            raise KeyError(f"param {name!r}: by={kw['by']!r} is not a factor")
        p = Param(name, **kw)
        self.params[name] = p
        return p

    def rule(self, target, value, **when):
        from .params import Rule
        r = Rule(target, value, dict(when))
        self.rules.append(r)
        return r

    # --- queries ---------------------------------------------------------
    def attributes(self, occ):
        """Everything known about an occasion, as one flat dict.

        Precedence, lowest first: subject covariates, factor-level attributes
        (in factor definition order), then the factor names themselves mapped
        to their level keys and the reserved keys. Model parameter settings
        ("params") are not included; see pyantigen.study.params.resolve.
        """
        if isinstance(occ, str):
            occ = self.occasions[occ]
        subj = self.subjects[occ.subject]
        attrs = dict(subj.covariates)
        lv = occ.level_dict
        for fname, f in self.factors.items():
            if fname in lv:
                attrs.update({k: v for k, v in f.levels[lv[fname]].items() if k != "params"})
        attrs.update(lv)
        attrs.update({"subject": occ.subject, "protocol": occ.protocol,
                      "period": occ.period, "occasion": occ.id})
        return attrs

    def select(self, **where):
        """Occasions whose attributes match every key in *where*.

        A value may be a single value or a list/tuple/set of accepted values.
        An unknown key raises: a misspelled factor would otherwise select
        nothing, silently.
        """
        known = self._attribute_names()
        unknown = set(where) - known
        if unknown:
            raise KeyError(f"select: unknown attribute(s) {sorted(unknown)}; "
                           f"known: {sorted(known)}")
        out = []
        for occ in self.occasions.values():
            a = self.attributes(occ)
            if all(_matches(a.get(k), v) for k, v in where.items()):
                out.append(occ)
        return out

    # --- internals -------------------------------------------------------
    def _attribute_names(self):
        names = set(RESERVED) | set(self.factors)
        for s in self.subjects.values():
            names |= set(s.covariates)
        for f in self.factors.values():
            for attrs in f.levels.values():
                names |= set(attrs) - {"params"}
        return names

    def _check_level(self, fac, lev, where):
        if fac not in self.factors:
            raise KeyError(f"{where}: unknown factor {fac!r}")
        if str(lev) not in self.factors[fac].levels:
            raise KeyError(f"{where}: factor {fac!r} has no level {lev!r}; "
                           f"levels: {list(self.factors[fac].levels)}")

    def _make_occasion(self, subject, protocol, period, id, levels):
        if subject not in self.subjects:
            raise KeyError(f"occasion: unknown subject {subject!r}")
        if protocol not in self.protocols:
            raise KeyError(f"occasion: unknown protocol {protocol!r}")
        subj = self.subjects[subject]
        full = dict(subj.levels)
        for fac, lev in levels.items():
            self._check_level(fac, lev, f"occasion of {subject!r}")
            if fac in subj.levels and subj.levels[fac] != str(lev):
                raise ValueError(
                    f"occasion of {subject!r}: factor {fac!r} is fixed to "
                    f"{subj.levels[fac]!r} for this subject, got {lev!r}")
            full[fac] = str(lev)
        if id is None:
            # Subject id, then the occasion's own (within-subject) levels in
            # factor-definition order, then the period.
            parts = [subject] + [full[f] for f in self.factors
                                 if f in levels and f not in subj.levels]
            if period is not None:
                parts.append(f"P{period}")
            id = "_".join(parts)
        if id in self.occasions:
            raise ValueError(f"occasion id {id!r} already exists; pass id= to disambiguate")
        occ = Occasion(id, subject, protocol, tuple(sorted(full.items())), period)
        self.occasions[id] = occ
        return occ


def _matches(actual, wanted):
    if isinstance(wanted, (list, tuple, set, frozenset)):
        return any(_matches(actual, w) for w in wanted)
    if isinstance(actual, str) or isinstance(wanted, str):
        return str(actual) == str(wanted)
    return actual == wanted
