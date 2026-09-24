"""Who was studied, what was done to them, and what was simulated.

A Study is built from:

  Factor      a named variable of the design with a few levels -- dose, drug,
              cell line, age decade. Everything a level implies (a dose in
              nmol, an event delay, a cell line's 122 expression values) is
              written ONCE, on the level.
  Subject     one individual, or one cohort when the data are cohort means.
              Carries covariates and may fix the levels of between-subject
              factors (a cohort's amyloid status, a patient's drug arm).
  Protocol    what was done: the event generator, solver window, recorded
              species, the loader for its inputs, and any settings every
              simulation under it shares.
  Simulation  one subject under one protocol at one choice of the remaining
              factor levels (optionally in a numbered period) -- one thing the
              Engine integrates. ``simulate`` adds one; ``simulate_all`` adds
              every combination of subjects x factor levels. An "arm" is only
              a selection of simulations, not a stored object.
  Assay       a dataset: observables measured together, and ``on=`` the
              simulations it exists for (see pyantigen.study.assay).
  Rule        a constraint the design imposes (pyantigen.study.params).

A Study is a paper -- its experiments and every dataset it reports. It makes
no choice about fitting: which datasets are scored, and which parameters
they fit, is an Optimization (pyantigen.study.optimization), which can pull
datasets from several studies.

Why this shape, and not a table with one row per condition: a condition
table repeats every attribute of every factor on every row, so its size is the
PRODUCT of the factors. Froehlich 2018's PEtab condition table is 9,570 rows x
147 columns (17 MB) because each row copies its cell line's 122 expression
values; the same information is 290 cell-line levels x 122 values plus 1,105
treatment levels x 23 values, about 23 times smaller. Here the size of a
design is the SUM of its factor tables plus one short reference per
simulation. The per-simulation view -- everything about one simulation in one
place -- is generated from the design by ``pyantigen.study.describe``.

Pairing is built in from the start. Every simulation names its subject, so a
contrast (drug over vehicle) can be formed within each subject -- the right
operation for a crossover, where each animal is its own control -- and
reduces to the cohort-mean case when the "subject" is a cohort.
"""
from dataclasses import dataclass, field
from itertools import product

from .refs import from_ref, to_ref

# Keys a simulation's attribute dict always has; factor, covariate and
# protocol-setting names may not shadow them.
RESERVED = ("subject", "protocol", "period", "simulation")


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
    settings: {name: value} every simulation under this protocol has as an
        attribute -- a property of the procedure, not of the people (whether
        CSF production compensates a catheter's drain, say).
    """
    name: str
    events: object
    solver: object
    observed: object
    data: object = None
    update_parameters: object = None
    update_opt_parameters: object = None
    settings: dict = field(default_factory=dict)

    _HOOKS = ("data", "update_parameters", "update_opt_parameters")

    def __post_init__(self):
        clash = set(self.settings) & set(RESERVED)
        if clash:
            raise ValueError(f"protocol {self.name!r}: settings {sorted(clash)} are reserved names")
        self.settings = dict(self.settings)

    def to_json(self):
        d = {"events": to_ref(self.events), "solver": to_ref(self.solver),
             "observed": to_ref(self.observed)}
        for k in self._HOOKS:
            if getattr(self, k) is not None:
                d[k] = to_ref(getattr(self, k))
        if self.settings:
            d["settings"] = dict(self.settings)
        return d

    @classmethod
    def from_json(cls, name, d):
        return cls(name, d["events"], d["solver"], d["observed"],
                   *(d.get(k) for k in cls._HOOKS), d.get("settings", {}))

    def resolved(self):
        return (from_ref(self.events), from_ref(self.solver), from_ref(self.observed))

    def hooks(self):
        """(data, update_parameters, update_opt_parameters), imported."""
        return tuple(from_ref(getattr(self, k)) for k in self._HOOKS)


@dataclass(frozen=True)
class Simulation:
    id: str
    subject: str
    protocol: str
    levels: tuple            # sorted ((factor, level), ...), the full assignment
    period: object = None

    @property
    def level_dict(self):
        return dict(self.levels)


class Study:
    """A paper: its DOI, factors, subjects, protocols, the simulations built
    from them, and the datasets (assays) it reports on them."""

    def __init__(self, name, doi=None, remarks=None):
        self.name = name
        # The paper this study is. One study, one publication: a dataset from
        # somewhere else belongs to a study of its own, so that every fit
        # records which papers it drew on.
        self.doi = doi
        self.remarks = list(remarks or [])
        self.factors = {}
        self.subjects = {}
        self.protocols = {}
        self.simulations = {}          # id -> Simulation, in creation order
        self._simulation_records = []  # compact generator records, for JSON
        self.assays = {}
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
                 update_parameters=None, update_opt_parameters=None, settings=None):
        if name in self.protocols:
            raise ValueError(f"protocol {name!r} already defined")
        p = Protocol(name, events, solver, observed, data,
                     update_parameters, update_opt_parameters, dict(settings or {}))
        self.protocols[name] = p
        return p

    def simulate(self, subject, protocol, period=None, id=None, **levels):
        """Add one simulation: *subject* under *protocol* at these factor levels."""
        sim = self._make_simulation(subject, protocol, period, id, levels)
        rec = [subject, protocol, dict(levels)]
        extra = {k: v for k, v in (("period", period), ("id", id)) if v is not None}
        if extra:
            rec.append(extra)
        self._simulation_records.append(rec)
        return sim

    def simulate_all(self, subjects, protocol, **factor_levels):
        """Add a simulation for every combination of subjects x factor levels.

        factor_levels maps a factor name to a list of level keys, or to "*"
        for all of that factor's levels. Stored in the JSON as this one call,
        not as its expansion.
        """
        subjects = list(subjects)
        spec = {}
        for fac, levs in factor_levels.items():
            if fac not in self.factors:
                raise KeyError(f"simulate_all: unknown factor {fac!r}")
            spec[fac] = (list(self.factors[fac].levels) if levs == "*"
                         else [str(x) for x in levs])
        out = []
        names = list(spec)
        for s in subjects:
            for combo in product(*(spec[n] for n in names)):
                out.append(self._make_simulation(s, protocol, None, None, dict(zip(names, combo))))
        self._simulation_records.append({"all": {
            "subjects": subjects, "protocol": protocol,
            "factors": {k: ("*" if factor_levels[k] == "*" else spec[k]) for k in names}}})
        return out

    def assay(self, name, *observables, on=None, source=None):
        """A dataset: observables measured together (Measured objects), on the
        simulations selected by *on*: None (all), a Study.select filter, or a
        Contrast. Whether it is scored is an Optimization's choice.

        *source* says where in the paper the data are (a figure or table), and
        how they were extracted when that is not obvious."""
        from .assay import Assay
        if name in self.assays:
            raise ValueError(f"assay {name!r} already defined")
        a = Assay(name, list(observables), on, source)
        self.assays[name] = a
        return a

    def rule(self, target, value, **when):
        from .params import Rule
        r = Rule(target, value, dict(when))
        self.rules.append(r)
        return r

    # --- queries ---------------------------------------------------------
    def attributes(self, sim):
        """Everything known about a simulation, as one flat dict.

        Precedence, lowest first: subject covariates, protocol settings,
        factor-level attributes (in factor definition order), then the factor
        names themselves mapped to their level keys and the reserved keys.
        Model parameter settings ("params") are not included; see
        pyantigen.study.params.resolve.
        """
        if isinstance(sim, str):
            sim = self.simulations[sim]
        attrs = dict(self.subjects[sim.subject].covariates)
        attrs.update(self.protocols[sim.protocol].settings)
        lv = sim.level_dict
        for fname, f in self.factors.items():
            if fname in lv:
                attrs.update({k: v for k, v in f.levels[lv[fname]].items() if k != "params"})
        attrs.update(lv)
        attrs.update({"subject": sim.subject, "protocol": sim.protocol,
                      "period": sim.period, "simulation": sim.id})
        return attrs

    def select(self, **where):
        """Simulations whose attributes match every key in *where*.

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
        for sim in self.simulations.values():
            a = self.attributes(sim)
            if all(_matches(a.get(k), v) for k, v in where.items()):
                out.append(sim)
        return out

    def scoring(self):
        """{simulation id: [(assay, contrast partner or None), ...]}.

        Every simulation appears, unscored ones with an empty list; each
        list is in assay declaration order. A contrast scores its numerator
        simulations; the partner is the paired denominator.
        """
        out = {sid: [] for sid in self.simulations}
        for a in self.assays.values():
            if a.contrast is not None:
                for num, den in a.contrast.pairs(self):
                    out[num.id].append((a, den))
            else:
                sims = self.select(**a.on) if a.on else self.simulations.values()
                for sim in sims:
                    out[sim.id].append((a, None))
        return out

    # --- internals -------------------------------------------------------
    def _attribute_names(self):
        names = set(RESERVED) | set(self.factors)
        for s in self.subjects.values():
            names |= set(s.covariates)
        for p in self.protocols.values():
            names |= set(p.settings)
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

    def _make_simulation(self, subject, protocol, period, id, levels):
        if subject not in self.subjects:
            raise KeyError(f"simulate: unknown subject {subject!r}")
        if protocol not in self.protocols:
            raise KeyError(f"simulate: unknown protocol {protocol!r}")
        subj = self.subjects[subject]
        full = dict(subj.levels)
        for fac, lev in levels.items():
            self._check_level(fac, lev, f"simulation of {subject!r}")
            if fac in subj.levels and subj.levels[fac] != str(lev):
                raise ValueError(
                    f"simulation of {subject!r}: factor {fac!r} is fixed to "
                    f"{subj.levels[fac]!r} for this subject, got {lev!r}")
            full[fac] = str(lev)
        if id is None:
            # Subject id, then the simulation's own (within-subject) levels in
            # factor-definition order, then the period.
            parts = [subject] + [full[f] for f in self.factors
                                 if f in levels and f not in subj.levels]
            if period is not None:
                parts.append(f"P{period}")
            id = "_".join(parts)
        if id in self.simulations:
            raise ValueError(f"simulation id {id!r} already exists; pass id= to disambiguate")
        sim = Simulation(id, subject, protocol, tuple(sorted(full.items())), period)
        self.simulations[id] = sim
        return sim


def _matches(actual, wanted):
    if isinstance(wanted, (list, tuple, set, frozenset)):
        return any(_matches(actual, w) for w in wanted)
    if isinstance(actual, str) or isinstance(wanted, str):
        return str(actual) == str(wanted)
    return actual == wanted
