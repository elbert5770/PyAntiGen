"""What was measured, where its data live, how noisy it is, and contrasts.

  Obs         a model quantity: an expression string, optionally log10- or
              baseline-transformed. Always carries a name, which is what the
              Engine keys sigma blocks and plots by.
  DataSource  where an observable's data are, declared: file, the rows that
              belong to an occasion (``where``, with "{attr}" placeholders
              filled from the occasion), the time column, and one or more
              value columns (several = replicate draws, stacked).
  Noise       declared sigma, per-point SD column, data-derived floor, and
              which factors a sigma is POOLED across.
  Assay       observables measured together.
  Contrast    a derived measurement between two occasions, drug over vehicle
              or drug minus placebo. Pairs are formed within each subject by
              default, which is the correct operation for a crossover and
              reduces to the cohort-mean case when the subject is a cohort.
"""
from dataclasses import dataclass, field

import numpy as np

from .refs import from_ref, to_ref


# --- model observables ------------------------------------------------------

@dataclass
class Obs:
    expr: str
    name: str = None
    transform: str = None           # None | "log10"
    floor: float = 1e-12            # log10 guard, as in the 1.x Flipflop spec

    def __post_init__(self):
        if self.transform not in (None, "log10"):
            raise ValueError(f"Obs {self.expr!r}: unknown transform {self.transform!r}")
        if self.name is None:
            self.name = self.expr if self.transform is None else f"log10({self.expr})"

    @classmethod
    def log10(cls, expr, name=None, floor=1e-12):
        return cls(expr, name=name, transform="log10", floor=floor)

    def engine_form(self):
        """What the Engine's observed_variable receives.

        A plain expression string, identical to what a hand-written 1.x
        loss_config would have used, so a lowered study scores exactly as its
        1.x original did.
        """
        if self.transform == "log10":
            return f"np.log10(np.maximum({self.expr}, {self.floor!r}))"
        return self.expr

    def to_json(self):
        d = {"expr": self.expr}
        if self.name != self.expr:
            d["name"] = self.name
        if self.transform:
            d["transform"] = self.transform
            if self.floor != 1e-12:
                d["floor"] = self.floor
        return d

    @classmethod
    def from_json(cls, d):
        if isinstance(d, str):
            return cls(d)
        return cls(d["expr"], d.get("name"), d.get("transform"), d.get("floor", 1e-12))


# --- data ------------------------------------------------------------------

@dataclass
class DataSource:
    file: str                        # relative to the data folder; may use "{attr}"
    time: str
    value: object                    # column name, or list of replicate columns
    where: dict = field(default_factory=dict)   # column -> value or "{attr}"
    sd: str = None                   # per-point SD column, if the data have one
    dropna: bool = True

    def rows_for(self, attrs, data_path, _cache=None):
        """The rows belonging to one occasion, as a (time, value[, sd]) frame.

        Replicate value columns are stacked into one "value" column, in the
        order listed, which is how the 1.x loaders stacked B1/B2/B3.
        """
        import os
        import pandas as pd
        path = os.path.join(data_path, _fill(self.file, attrs))
        if _cache is not None and path in _cache:
            df = _cache[path]
        else:
            df = pd.read_csv(path)
            if _cache is not None:
                _cache[path] = df
        mask = np.ones(len(df), dtype=bool)
        for col, want in self.where.items():
            if col not in df.columns:
                raise KeyError(f"{self.file}: no column {col!r} to filter on")
            want = _fill(want, attrs)
            mask &= (df[col].astype(str) == str(want)).to_numpy()
        sub = df[mask]
        cols = self.value if isinstance(self.value, (list, tuple)) else [self.value]
        parts = []
        for c in cols:
            if c not in sub.columns:
                raise KeyError(f"{self.file}: no value column {c!r}")
            keep = [self.time, c] + ([self.sd] if self.sd else [])
            part = sub[keep].rename(columns={self.time: "time", c: "value"})
            if self.sd:
                part = part.rename(columns={self.sd: "sd"})
            parts.append(part)
        out = pd.concat(parts, ignore_index=True)
        if self.dropna:
            out = out.dropna(subset=["time", "value"]).reset_index(drop=True)
        return out

    def to_json(self):
        d = {"file": self.file, "time": self.time, "value": self.value}
        if self.where:
            d["where"] = dict(self.where)
        if self.sd:
            d["sd"] = self.sd
        if not self.dropna:
            d["dropna"] = False
        return d

    @classmethod
    def from_json(cls, d):
        return cls(d["file"], d["time"], d["value"], d.get("where", {}),
                   d.get("sd"), d.get("dropna", True))


def _fill(template, attrs):
    if isinstance(template, str) and "{" in template:
        try:
            return template.format(**attrs)
        except KeyError as exc:
            raise KeyError(f"data filter {template!r} names {exc.args[0]!r}, "
                           "which this occasion does not have") from None
    return template


# --- noise -----------------------------------------------------------------

@dataclass
class Noise:
    """How an observable's residuals are scored.

    sigma       a declared constant. The block then costs no parameter.
    sd_column   use the DataSource's per-point ``sd`` column instead.
    floor_from_data  cap a profiled sigma at a floor estimated from the data
                (the 1.x sigma_floor_from_data).
    pool        which occasions share one estimated sigma:
                  None     one sigma per occasion (the 1.x default)
                  "all"    one sigma across every occasion of the measurement
                  [f, ...] one sigma per combination of the OTHER factors,
                           i.e. pooled across the listed ones
    Neither sigma nor sd_column: the sigma is profiled out (concentrated).
    """
    sigma: float = None
    sd_column: bool = False
    floor_from_data: bool = False
    pool: object = None

    def engine_keys(self):
        d = {}
        if self.sigma is not None:
            d["sigma_method"] = "fixed"
            d["sigma_value"] = self.sigma
        if self.sd_column:
            d["noise_column"] = "sd"
        if self.floor_from_data:
            d["sigma_floor_from_data"] = True
        return d

    def to_json(self):
        d = {}
        if self.sigma is not None:
            d["sigma"] = self.sigma
        if self.sd_column:
            d["sd_column"] = True
        if self.floor_from_data:
            d["floor_from_data"] = True
        if self.pool is not None:
            d["pool"] = self.pool
        return d

    @classmethod
    def from_json(cls, d):
        return cls(d.get("sigma"), d.get("sd_column", False),
                   d.get("floor_from_data", False), d.get("pool"))


# --- assays ----------------------------------------------------------------

@dataclass
class Measured:
    """One observable of an assay: the model side, the data side, the noise."""
    name: str
    model: Obs
    data: DataSource
    noise: Noise = field(default_factory=Noise)
    # Only score occasions whose attributes match this (e.g. {"has_A_data":
    # True}); the 1.x equivalent was an if-statement inside a loss_config.
    only: dict = field(default_factory=dict)

    def to_json(self):
        d = {"model": self.model.to_json(), "data": self.data.to_json()}
        nj = self.noise.to_json()
        if nj:
            d["noise"] = nj
        if self.only:
            d["only"] = dict(self.only)
        return d

    @classmethod
    def from_json(cls, name, d):
        return cls(name, Obs.from_json(d["model"]), DataSource.from_json(d["data"]),
                   Noise.from_json(d.get("noise", {})), d.get("only", {}))


@dataclass
class Assay:
    name: str
    observables: list               # [Measured]

    def to_json(self):
        return {m.name: m.to_json() for m in self.observables}

    @classmethod
    def from_json(cls, name, d):
        return cls(name, [Measured.from_json(k, v) for k, v in d.items()])


# --- contrasts -------------------------------------------------------------

def ratio_pct(parts):
    """100 * numerator / denominator, on predictions already at the data times."""
    num, den = (np.asarray(p, dtype=float) for p in parts)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 100.0 * num / den


def difference(parts):
    """numerator - denominator, on predictions already at the data times."""
    num, den = (np.asarray(p, dtype=float) for p in parts)
    return num - den


CONTRAST_OPS = {"ratio_pct": ratio_pct, "diff": difference}


@dataclass
class Contrast:
    """Numerator occasions against their matching denominator occasions.

    numerator / denominator are ``Study.select`` filters. Each numerator
    occasion is paired with exactly one denominator occasion:

      pairing="subject"  the denominator must be the SAME subject, and agree
                         on every factor in ``match``. Within-subject: right
                         for crossovers, and for cohort-mean data where the
                         cohort is the subject.
      pairing="between"  any subject; agreement on ``match`` only. For
                         parallel-group designs compared at the arm level.

    Zero or more than one candidate raises: a silently missing or ambiguous
    control is exactly the failure the 1.x positional composites could hide.
    """
    name: str
    op: str
    numerator: dict
    denominator: dict
    pairing: str = "subject"
    match: list = field(default_factory=list)

    def __post_init__(self):
        if self.op not in CONTRAST_OPS:
            raise ValueError(f"contrast {self.name!r}: op must be one of {sorted(CONTRAST_OPS)}")
        if self.pairing not in ("subject", "between"):
            raise ValueError(f"contrast {self.name!r}: pairing must be 'subject' or 'between'")

    def pairs(self, study):
        nums = study.select(**self.numerator)
        dens = study.select(**self.denominator)
        if not nums:
            raise ValueError(f"contrast {self.name!r}: numerator selects no occasions")
        out = []
        for n in nums:
            na = study.attributes(n)
            cands = [d for d in dens
                     if d.id != n.id
                     and (self.pairing == "between" or d.subject == n.subject)
                     and all(study.attributes(d).get(m) == na.get(m) for m in self.match)]
            if len(cands) != 1:
                what = "no" if not cands else f"{len(cands)} ({[c.id for c in cands]})"
                raise ValueError(
                    f"contrast {self.name!r}: numerator {n.id!r} has {what} matching "
                    f"denominator occasions (pairing={self.pairing!r}, match={self.match})")
            out.append((n, cands[0]))
        return out

    def to_json(self):
        d = {"op": self.op, "numerator": self.numerator, "denominator": self.denominator}
        if self.pairing != "subject":
            d["pairing"] = self.pairing
        if self.match:
            d["match"] = list(self.match)
        return d

    @classmethod
    def from_json(cls, name, d):
        return cls(name, d["op"], d["numerator"], d["denominator"],
                   d.get("pairing", "subject"), d.get("match", []))


@dataclass
class Measurement:
    """An assay scored on a selection of occasions, or on a contrast."""
    assay: str
    on: dict = None                  # Study.select filter; None = all occasions
    contrast: str = None             # or the name of a Contrast

    def to_json(self):
        d = {"assay": self.assay}
        if self.contrast:
            d["contrast"] = self.contrast
        elif self.on:
            d["on"] = self.on
        return d

    @classmethod
    def from_json(cls, d):
        return cls(d["assay"], d.get("on"), d.get("contrast"))


# Re-exported so a lowered replicate can pickle a reference to the op.
__all__ = ["Obs", "DataSource", "Noise", "Measured", "Assay", "Contrast",
           "Measurement", "ratio_pct", "difference", "CONTRAST_OPS",
           "to_ref", "from_ref"]
