"""Parameters, and the one function that decides an occasion's values.

  Param   a model parameter the estimation cares about. Global by default;
          ``by="<factor>"`` makes one value per level of that factor (drug-
          specific binding, subject-specific SF). ``estimate=False`` with
          ``values`` fixes it per level instead of fitting it.
  Rule    a constraint applied LAST, after every fitted value: "k_oligo1 is
          0 when amyloid_positive is False". 1.x needed a hook re-run after
          set_parameters for exactly this; here the order is structural.

``resolve(study, occasion, theta)`` returns the absolute value of every
parameter the design sets for that occasion, from

    factor-level params -> fixed Param values -> theta (fitted) -> rules

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
    x0: object = None               # float, or {level: float} when by= is set
    bounds: object = None           # (lo, hi), or {level: (lo, hi)}
    scale: str = "auto"             # "auto" | "lin" | "log10"
    by: str = None
    estimate: bool = True
    values: dict = None             # fixed {level: value} when estimate=False

    def __post_init__(self):
        if not self.estimate and self.by and not self.values:
            raise ValueError(f"Param {self.name!r}: a fixed by-{self.by} parameter needs values")
        if self.estimate and self.x0 is None:
            raise ValueError(f"Param {self.name!r}: an estimated parameter needs x0")

    def fitted_names(self, study):
        """The optimizer-facing names this Param contributes, in level order."""
        if not self.estimate:
            return []
        if self.by is None:
            return [self.name]
        return [by_name(self.name, self.by, lev) for lev in self._used_levels(study)]

    def _used_levels(self, study):
        used = {o.level_dict.get(self.by) for o in study.occasions.values()}
        return [lev for lev in study.factors[self.by].levels if lev in used]

    def x0_for(self, level=None):
        return self.x0[level] if isinstance(self.x0, dict) else self.x0

    def bounds_for(self, level=None):
        return self.bounds[level] if isinstance(self.bounds, dict) else self.bounds

    def to_json(self):
        d = {}
        if self.x0 is not None:
            d["x0"] = self.x0
        if self.bounds is not None:
            d["bounds"] = ({k: list(v) for k, v in self.bounds.items()}
                           if isinstance(self.bounds, dict) else list(self.bounds))
        if self.scale != "auto":
            d["scale"] = self.scale
        if self.by:
            d["by"] = self.by
        if not self.estimate:
            d["estimate"] = False
        if self.values:
            d["values"] = self.values
        return d

    @classmethod
    def from_json(cls, name, d):
        b = d.get("bounds")
        if isinstance(b, dict):
            b = {k: tuple(v) for k, v in b.items()}
        elif b is not None:
            b = tuple(b)
        return cls(name, d.get("x0"), b, d.get("scale", "auto"), d.get("by"),
                   d.get("estimate", True), d.get("values"))


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


def resolve(study, occasion, theta=None):
    """Absolute parameter values for *occasion*, given fitted values *theta*.

    theta maps optimizer-facing names (see Param.fitted_names) to values; it
    may be None or partial, in which case only the fixed parts apply -- which
    is what is set before the pre-dose block, when nothing fitted is known.
    """
    if isinstance(occasion, str):
        occasion = study.occasions[occasion]
    lv = occasion.level_dict
    attrs = study.attributes(occasion)
    out = {}
    # 1. what the factor levels fix
    for fname, f in study.factors.items():
        if fname in lv:
            out.update(f.levels[lv[fname]].get("params", {}))
    # 2. fixed Params
    for p in study.params.values():
        if not p.estimate and p.values is not None:
            key = lv.get(p.by) if p.by else None
            if p.by is None:
                out[p.name] = p.values
            elif key in p.values:
                out[p.name] = p.values[key]
    # 3. fitted values
    if theta:
        for p in study.params.values():
            if not p.estimate:
                continue
            if p.by is None:
                if p.name in theta:
                    out[p.name] = theta[p.name]
            else:
                key = by_name(p.name, p.by, lv.get(p.by))
                if key in theta:
                    out[p.name] = theta[key]
    # 4. rules, last
    for rule in study.rules:
        if rule.applies(attrs):
            out[rule.target] = rule.value
    return out
