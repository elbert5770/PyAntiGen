"""Fitted parameters, rules, and the one function that decides a simulation's values.

  Param   a model parameter an Optimization fits. Global by default;
          ``by="<factor>"`` makes one fitted value per level of that factor
          (drug-specific binding, subject-specific SF). Params belong to an
          Optimization, not to a Study: which constants a set of datasets can
          pin down is a decision about the fit, not a fact about the papers.
          Values a design FIXES per level live on the factor level
          ("params"), in the Study.
  Rule    a constraint applied LAST, after every fitted value: "k_oligo1 is
          0 when amyloid_positive is False". A fact about the design, so it
          lives in the Study. 1.x needed a hook re-run after set_parameters
          for exactly this; here the order is structural.

``resolve(study, simulation, theta, params)`` returns the absolute value of
every parameter the design and the fit set for that simulation, from

    factor-level params -> theta (fitted, via params) -> rules

It reads nothing from a model instance, so the result cannot depend on what
a previous evaluation left behind -- the bug class behind the 1.x species
overlay, amyloid-status and warm-start contamination findings.
"""
from dataclasses import dataclass, field


def by_name(name, factor, level):
    """Name of the fitted parameter for one level of a by-factor Param."""
    return f"{name}[{factor}={level}]"


@dataclass
class Param:
    name: str
    x0: object                      # float, or {level: float} when by= is set
    bounds: object = None           # (lo, hi), or {level: (lo, hi)}
    scale: str = "auto"             # "auto" | "lin" | "log10"
    by: str = None

    def __post_init__(self):
        if self.x0 is None:
            raise ValueError(f"Param {self.name!r}: needs x0")
        if isinstance(self.bounds, list):
            self.bounds = tuple(self.bounds)

    def levels(self, studies):
        """Levels of the by-factor that occur in *studies*' simulations, in
        factor order (first study that defines the factor wins the order)."""
        order, used = [], set()
        for st in studies:
            if self.by in st.factors:
                for lev in st.factors[self.by].levels:
                    if lev not in order:
                        order.append(lev)
                used |= {s.level_dict.get(self.by) for s in st.simulations.values()}
        return [lev for lev in order if lev in used]

    def fitted_names(self, studies):
        """The optimizer-facing names this Param contributes, in level order."""
        if self.by is None:
            return [self.name]
        return [by_name(self.name, self.by, lev) for lev in self.levels(studies)]

    def x0_for(self, level=None):
        return self.x0[level] if isinstance(self.x0, dict) else self.x0

    def bounds_for(self, level=None):
        return self.bounds[level] if isinstance(self.bounds, dict) else self.bounds

    def to_json(self):
        d = {"x0": self.x0}
        if self.bounds is not None:
            d["bounds"] = ({k: list(v) for k, v in self.bounds.items()}
                           if isinstance(self.bounds, dict) else list(self.bounds))
        if self.scale != "auto":
            d["scale"] = self.scale
        if self.by:
            d["by"] = self.by
        return d

    @classmethod
    def from_json(cls, name, d):
        b = d.get("bounds")
        if isinstance(b, dict):
            b = {k: tuple(v) for k, v in b.items()}
        elif b is not None:
            b = tuple(b)
        return cls(name, d["x0"], b, d.get("scale", "auto"), d.get("by"))


@dataclass
class Rule:
    target: str
    value: float
    when: dict = field(default_factory=dict)   # attribute -> required value

    def applies(self, attrs):
        from .design import _matches
        return all(_matches(attrs.get(k), v) for k, v in self.when.items())

    def to_json(self):
        return {"target": self.target, "value": self.value, "when": self.when}

    @classmethod
    def from_json(cls, d):
        return cls(d["target"], d["value"], d.get("when", {}))


def resolve(study, simulation, theta=None, params=()):
    """Absolute parameter values for *simulation*.

    params are the fitted Params (an Optimization's); theta maps their
    optimizer-facing names (see Param.fitted_names) to values. theta may be
    None or partial, in which case only the design's own values apply --
    which is what is set before the pre-dose block, when nothing fitted is
    known.
    """
    if isinstance(simulation, str):
        simulation = study.simulations[simulation]
    lv = simulation.level_dict
    attrs = study.attributes(simulation)
    out = {}
    # 1. what the factor levels fix
    for fname, f in study.factors.items():
        if fname in lv:
            out.update(f.levels[lv[fname]].get("params", {}))
    # 2. fitted values
    if theta:
        for p in params:
            if p.by is None:
                if p.name in theta:
                    out[p.name] = theta[p.name]
            elif p.by in lv:
                key = by_name(p.name, p.by, lv[p.by])
                if key in theta:
                    out[p.name] = theta[key]
    # 3. rules, last
    for rule in study.rules:
        if rule.applies(attrs):
            out[rule.target] = rule.value
    return out
