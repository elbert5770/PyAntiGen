"""An Optimization: which datasets, from which studies, fit which parameters.

A Study is a paper: its experiments (subjects, protocols, the simulations
built from them) and every dataset it reports (assays, each saying on which
simulations that dataset exists). It makes no choice about fitting.

An Optimization makes those choices. It names the parameters to fit, and
USES datasets from one or more studies:

    fit = Optimization("rhesus_joint", params=[
              Param("Q_CSF_base", x0=..., bounds=(...)),
              Param("GSI_KI_MK0752", x0=..., bounds=(1, 2e4), scale="log10"),
              Param("BACEI_KI_MBI5", x0=..., bounds=(1e-5, 1e3), scale="log10")])
    fit.use(cook, assays=["csf_pct_baseline", "csf_mfl"], on={"subject": "crossover"})
    fit.use(dobrowolska, assays=["csf_pct_vehicle", "csf_mfl"])
    fit.simulate_only(cook, on={"subject": "followup"})     # plotted, not scored

``assays`` names whole assays ("csf_pct_vehicle") or single observables of
one ("csf_pct_vehicle.AB40"); ``on`` further restricts the simulations,
intersected with where each assay's data exist. The same paper can serve
several fits, and one fit can pull from several papers -- which is how
experiments are actually combined.

Loss elements are emitted use by use, and within a use simulation by
simulation (creation order), then assay by assay (the order given).
"""
import hashlib
import json
from dataclasses import dataclass, field

from .params import Param


@dataclass
class Use:
    study: object                  # a Study
    assays: list = None            # ["assay", "assay.observable", ...]; None = all
    on: dict = None                # Study.select filter; None = everywhere the data exist
    score: bool = True             # False: simulate for plots, score nothing

    def selected(self):
        """[(assay, [observable names] or None)] in the order given."""
        st = self.study
        names = self.assays if self.assays is not None else list(st.assays)
        out = []
        for spec in names:
            aname, _, oname = spec.partition(".")
            if aname not in st.assays:
                raise KeyError(f"study {st.name!r} has no assay {aname!r}")
            assay = st.assays[aname]
            if oname and oname not in {m.name for m in assay.observables}:
                raise KeyError(f"assay {st.name}.{aname} has no observable {oname!r}")
            for a, obs in out:
                if a is assay:
                    if obs is None or not oname:
                        raise ValueError(f"assay {st.name}.{aname} used twice")
                    obs.append(oname)
                    break
            else:
                out.append((assay, [oname] if oname else None))
        return out

    def simulations(self):
        st = self.study
        return st.select(**self.on) if self.on else list(st.simulations.values())

    def scoring(self, full=None):
        """{simulation id: [(assay, partner, [observable names] or None)]} for
        what THIS use scores, each list in the order the assays were given.
        *full* is study.scoring(), passed in to avoid recomputing it."""
        st = self.study
        out = {sid: [] for sid in st.simulations}
        if not self.score:
            return out
        full = full if full is not None else st.scoring()
        allowed = {s.id for s in self.simulations()}
        for assay, obs in self.selected():
            for sid in st.simulations:
                if sid in allowed:
                    for a, partner in full[sid]:
                        if a is assay:
                            out[sid].append((assay, partner, obs))
        return out

    def to_json(self):
        d = {"study": self.study.name}
        if self.assays is not None:
            d["assays"] = list(self.assays)
        if self.on:
            d["on"] = dict(self.on)
        if not self.score:
            d["score"] = False
        return d


@dataclass
class Optimization:
    name: str
    params: list = field(default_factory=list)    # [Param]
    method: str = "Nelder-Mead"
    optimizer_kwargs: dict = field(default_factory=dict)
    n_starts: int = 1
    start_seed: int = None
    search_decades: float = None
    remarks: list = field(default_factory=list)
    uses: list = field(default_factory=list)      # [Use]

    def __post_init__(self):
        names = [p.name for p in self.params]
        dup = {n for n in names if names.count(n) > 1}
        if dup:
            raise ValueError(f"optimization {self.name!r}: params repeat: {sorted(dup)}")

    # --- building --------------------------------------------------------
    def use(self, study, assays=None, on=None):
        """Score these datasets of *study* (all of them by default)."""
        u = Use(study, list(assays) if assays is not None else None, dict(on) if on else None)
        u.selected()                       # fail now on a misspelled assay
        self.uses.append(u)
        return u

    def simulate_only(self, study, on=None):
        """Simulate these simulations of *study* for plotting; score nothing."""
        u = Use(study, [], dict(on) if on else None, score=False)
        self.uses.append(u)
        return u

    # --- queries ---------------------------------------------------------
    @property
    def studies(self):
        seen, out = set(), []
        for u in self.uses:
            if id(u.study) not in seen:
                seen.add(id(u.study))
                out.append(u.study)
        return out

    def param(self, name):
        for p in self.params:
            if p.name == name:
                return p
        raise KeyError(name)

    def scoring(self, study):
        """{simulation id: [(assay, partner, [observable names] or None)]}
        for everything this optimization scores in *study*, over all uses."""
        out = {sid: [] for sid in study.simulations}
        full = study.scoring()
        for u in self.uses:
            if u.study is study:
                for sid, entries in u.scoring(full).items():
                    out[sid].extend(entries)
        return out

    def simulated(self, study):
        """Ids of *study*'s simulations this optimization needs at all."""
        ids = set()
        sc = self.scoring(study)
        for sid, entries in sc.items():
            for _a, partner, _o in entries:
                ids.add(sid)
                if partner is not None:
                    ids.add(partner.id)
        for u in self.uses:
            if u.study is study and not u.score:
                ids |= {s.id for s in u.simulations()}
        return ids

    # --- record ----------------------------------------------------------
    def to_dict(self):
        from .serialize import fingerprint
        d = {"pyantigen_optimization": 1, "name": self.name}
        if self.remarks:
            d["remarks"] = list(self.remarks)
        d["studies"] = {st.name: fingerprint(st) for st in self.studies}
        d["uses"] = [u.to_json() for u in self.uses]
        d["params"] = {p.name: p.to_json() for p in self.params}
        d["method"] = self.method
        if self.optimizer_kwargs:
            d["optimizer_kwargs"] = self.optimizer_kwargs
        if self.n_starts != 1:
            d["n_starts"] = self.n_starts
        if self.start_seed is not None:
            d["start_seed"] = self.start_seed
        if self.search_decades is not None:
            d["search_decades"] = self.search_decades
        return d

    @classmethod
    def from_dict(cls, d, studies):
        """Rebuild from a record; *studies* maps study name -> Study. A study
        whose design no longer matches the fingerprint recorded raises."""
        from .serialize import fingerprint
        if d.get("pyantigen_optimization") != 1:
            raise ValueError("not a pyantigen optimization record (format 1)")
        for name, fp in d["studies"].items():
            if name not in studies:
                raise KeyError(f"optimization {d['name']!r} needs study {name!r}")
            if fingerprint(studies[name]) != fp:
                raise ValueError(f"study {name!r} has changed since optimization "
                                 f"{d['name']!r} was recorded")
        opt = cls(d["name"], [Param.from_json(n, p) for n, p in d["params"].items()],
                  d.get("method", "Nelder-Mead"), d.get("optimizer_kwargs", {}),
                  d.get("n_starts", 1), d.get("start_seed"), d.get("search_decades"),
                  d.get("remarks", []))
        for u in d["uses"]:
            st = studies[u["study"]]
            if u.get("score", True):
                opt.use(st, u.get("assays"), u.get("on"))
            else:
                opt.simulate_only(st, u.get("on"))
        return opt

    def fingerprint(self):
        text = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()


def save_optimization(opt, path):
    from .serialize import dumps
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(dumps(opt.to_dict()) + "\n")
    return path


def load_optimization(path, studies):
    with open(path, encoding="utf-8") as f:
        return Optimization.from_dict(json.load(f), studies)
