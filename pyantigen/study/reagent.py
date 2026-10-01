"""Reagent: a named substance, declared once and referenced wherever used.

A reagent appears in two roles. It is part of the TARGET -- medium, serum,
diet, the vehicle a drug is dissolved in -- and it is part of the
INTERVENTION -- the drug, the tracer. Same substance, two roles, so it is one
declaration referenced twice rather than two strings that can disagree.

What a Reagent is for, concretely:

  * A molecule gets an identifier, so "MK-0752" in one study and "MK0752" in
    another are the same substance or are visibly not.
  * A vehicle is itself a reagent, recursively, because a vehicle is a
    mixture. "0.4% methylcellulose" is a reagent with a concentration.
  * A potency is attached to a (reagent, system) PAIR, not to the model.
    MK-0752's 5 nM IC50 in SH-SY5Y and its fitted Ki in rhesus CSF are
    different quantities about the same molecule in different systems. Today
    ``system`` is a free string; when Target lands it becomes a reference to
    one, and the registries that currently explain this in English will not
    have to.

What a Reagent is NOT: it is not a model parameter and never becomes one. A
potency recorded here is what a paper measured somewhere else. A fitted
constant stays a Param on an Optimization, and nothing in this module is
read by ``resolve`` or reaches a replicate dict. The two are related by the
prose that cites one when reporting the other, which is exactly as much
coupling as is wanted: a fitted Ki that disagrees with a published one is a
finding, not an inconsistency to be reconciled automatically.

    LEU13C6 = Reagent("leucine_13C6", id="CHEBI:15603",
                      label="[U-13C6] L-leucine",
                      supplier="Cambridge Isotope Laboratories CLM-2262",
                      concentration=Q(7.5, "mg/mL"),
                      vehicle=Reagent("saline", id="CHEBI:75958"))
"""
from dataclasses import dataclass

from .quantity import Q


@dataclass(frozen=True)
class Potency:
    """One published potency of a reagent against one system.

    kind is the measurement's own word for itself -- "Ki", "IC50", "ED50" --
    kept verbatim rather than normalized, because IC50 and Ki are not
    interchangeable and a layer that pretended otherwise would be lying.
    """
    kind: str
    value: Q
    system: str                      # free text now; a Target reference later
    source: str = None               # paper, table or figure it is quoted from

    def __post_init__(self):
        object.__setattr__(self, "value", Q.parse(self.value))

    def __str__(self):
        s = f"{self.kind} {self.value} in {self.system}"
        return f"{s} ({self.source})" if self.source else s

    def to_json(self):
        d = {"kind": self.kind, "value": self.value.to_json(), "system": self.system}
        if self.source:
            d["source"] = self.source
        return d

    @classmethod
    def from_json(cls, d):
        return cls(d["kind"], Q.from_json(d["value"]), d["system"], d.get("source"))


@dataclass(frozen=True)
class Reagent:
    """A substance: what it is, what it is dissolved in, what it does.

    Every field but ``name`` is optional. A reagent with nothing but a name
    is still worth declaring -- it is a place for the identifier to go when
    someone looks it up, and ``validate`` will say it is missing.
    """
    name: str
    id: str = None                   # ChEBI, DrugBank, InChIKey, RRID, ...
    label: str = None                # what the paper calls it
    concentration: Q = None
    vehicle: object = None           # a Reagent, or the name of one
    supplier: str = None
    lot: str = None
    potencies: tuple = ()
    notes: str = None

    def __post_init__(self):
        if not self.name:
            raise ValueError("a Reagent needs a name")
        if self.concentration is not None:
            object.__setattr__(self, "concentration", Q.parse(self.concentration))
        if isinstance(self.vehicle, dict):
            object.__setattr__(self, "vehicle", Reagent.from_json(
                self.vehicle.get("name"), self.vehicle))
        object.__setattr__(self, "potencies", tuple(self.potencies))

    @property
    def vehicle_name(self):
        """The vehicle's name, whether it is nested or a reference."""
        if self.vehicle is None:
            return None
        return self.vehicle if isinstance(self.vehicle, str) else self.vehicle.name

    def potency(self, kind=None, system=None):
        """Recorded potencies, optionally filtered by kind and/or system."""
        return [p for p in self.potencies
                if (kind is None or p.kind == kind)
                and (system is None or p.system == system)]

    def mixture(self):
        """This reagent and every nested vehicle, outermost first."""
        out, seen = [], set()
        r = self
        while isinstance(r, Reagent) and id(r) not in seen:
            seen.add(id(r))
            out.append(r)
            r = r.vehicle
        return out

    def __str__(self):
        bits = [self.name]
        if self.concentration is not None:
            bits.append(f"at {self.concentration}")
        if self.id:
            bits.append(f"[{self.id}]")
        if self.vehicle_name:
            bits.append(f"in {self.vehicle_name}")
        return " ".join(bits)

    def to_json(self):
        d = {}
        for k in ("id", "label", "supplier", "lot", "notes"):
            if getattr(self, k):
                d[k] = getattr(self, k)
        if self.concentration is not None:
            d["concentration"] = self.concentration.to_json()
        if self.vehicle is not None:
            d["vehicle"] = (self.vehicle if isinstance(self.vehicle, str)
                            else dict(self.vehicle.to_json(), name=self.vehicle.name))
        if self.potencies:
            d["potencies"] = [p.to_json() for p in self.potencies]
        return d

    @classmethod
    def from_json(cls, name, d):
        d = dict(d or {})
        d.pop("name", None)
        conc = d.get("concentration")
        veh = d.get("vehicle")
        if isinstance(veh, dict):
            veh = cls.from_json(veh.get("name"), veh)
        return cls(name,
                   id=d.get("id"), label=d.get("label"),
                   concentration=Q.from_json(conc) if conc is not None else None,
                   vehicle=veh, supplier=d.get("supplier"), lot=d.get("lot"),
                   potencies=tuple(Potency.from_json(p) for p in d.get("potencies", [])),
                   notes=d.get("notes"))
