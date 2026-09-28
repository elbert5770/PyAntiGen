"""Q: a number with a UCUM unit, checked at the protocol boundary.

The Engine is unitless by convention and stays that way. Q exists so that
what a PAPER says -- "1.5 mL of CSF per draw", "4 mg/kg/h for 12 h" -- can be
written down as it was said, converted once where it enters the design, and
checked for dimensional sense before it gets there. Nothing in the Engine
sees a Q; ``resolve`` and the replicate dicts still carry plain floats.

Why this is worth having at all: two of the three errors recorded in
docs/PROTOCOL_LAYER.md are arithmetic on quantities whose units were kept in
a comment. A CSF drain rate that is "16 draws x 1.5 mL over 144 h" is a
derivation, and a derivation that runs is one that cannot be aliased to
another paper's number:

    >>> (16 * Q(1.5, "mL") / Q(144, "h")).in_("L/h")
    0.00016666666666666666

Scope. This is a boundary check, not a unit system. It knows the SI base
units, the prefixes, and the handful of atoms these protocols use (litre,
minute, hour, day, week, year, percent). It does NOT know derived electrical
or mechanical units, degrees Celsius (an offset, not a factor), or anything
that needs chemistry to convert -- mol to g needs a molar mass, and nM to
mg/L is therefore a conversion Q refuses. That refusal is the point: those
are the conversions that need a molecule named, which is what a Reagent is
for.

Grammar. UCUM's, restricted: terms joined by "." (multiply) and "/" (divide),
each an optional prefix, an atom, and an optional integer exponent -- "mg",
"mL", "m3", "mg/kg/h", "1/h", "10*3/L" is NOT supported. Division is
left-associative, so "mg/kg/h" is (mg/kg)/h, which is what it means on a
label. An atom is matched before a prefix+atom is tried, so "h" is an hour
and not a hecto-nothing, "d" is a day, "min" is a minute, and "a" is a year.
"""
import math
import re
from dataclasses import dataclass, field

# (mass, length, time, amount, temperature, current, luminous)
_NAMES = ("M", "L", "T", "N", "K", "I", "J")
_NONE = (0,) * 7


def _dim(**kw):
    return tuple(kw.get(n, 0) for n in _NAMES)


# code -> (factor to the SI base unit, dimension). Base units: kg, m, s, mol,
# K, A, cd -- so a gram is 1e-3, not 1.
_ATOMS = {
    "g":   (1e-3, _dim(M=1)),
    "m":   (1.0, _dim(L=1)),
    "s":   (1.0, _dim(T=1)),
    "mol": (1.0, _dim(N=1)),
    "K":   (1.0, _dim(K=1)),
    "A":   (1.0, _dim(I=1)),
    "cd":  (1.0, _dim(J=1)),
    # volume
    "L":   (1e-3, _dim(L=3)),
    "l":   (1e-3, _dim(L=3)),
    # time, as papers write it
    "min": (60.0, _dim(T=1)),
    "h":   (3600.0, _dim(T=1)),
    "d":   (86400.0, _dim(T=1)),
    "wk":  (604800.0, _dim(T=1)),
    "a":   (31557600.0, _dim(T=1)),      # UCUM a_j, the Julian year
    # dimensionless
    "1":   (1.0, _NONE),
    "%":   (0.01, _NONE),
}

_PREFIXES = {
    "Y": 1e24, "Z": 1e21, "E": 1e18, "P": 1e15, "T": 1e12, "G": 1e9,
    "M": 1e6, "k": 1e3, "h": 1e2, "da": 1e1,
    "d": 1e-1, "c": 1e-2, "m": 1e-3, "u": 1e-6, "n": 1e-9, "p": 1e-12,
    "f": 1e-15, "a": 1e-18, "z": 1e-21, "y": 1e-24,
}

# An atom is letters, or the two standalone atoms "1" and "%"; an exponent is
# a signed integer glued to it ("m3", "h-1").
_TERM = re.compile(r"^(1|%|[A-Za-z]+)([+-]?[0-9]+)?$")


class UnitError(ValueError):
    """A unit string that will not parse, or a conversion that is not a factor."""


def _atom(code):
    """(factor, dimension) for one prefixed atom code.

    An exact atom wins over any prefix reading, which is what makes "h" an
    hour rather than a hecto-unit and "min" a minute rather than milli-inch.
    """
    if code in _ATOMS:
        return _ATOMS[code]
    for plen in (2, 1):                      # "da" before "d"
        pre, rest = code[:plen], code[plen:]
        if pre in _PREFIXES and rest in _ATOMS:
            factor, dim = _ATOMS[rest]
            if rest == "%" or rest == "1":   # a prefixed percent is not a thing
                break
            return factor * _PREFIXES[pre], dim
    raise UnitError(f"unknown unit atom {code!r}")


def parse_unit(unit):
    """(factor to base units, dimension tuple) for a UCUM expression."""
    if unit is None:
        return 1.0, _NONE
    text = str(unit).strip()
    if text == "":
        return 1.0, _NONE
    factor, dim, sign = 1.0, list(_NONE), 1
    # Split keeping the operators; a leading "/" means the first term divides.
    # An operator with nothing after it ("mg/") leaves an empty token, which
    # is an error rather than something to skip over.
    parts = re.split(r"([./*])", text)
    for i, tok in enumerate(parts):
        if tok == "":
            if i == 0 and len(parts) > 1:      # leading "/h"
                continue
            raise UnitError(f"missing unit term in {unit!r}")
        if tok in "./*":
            sign = -1 if tok == "/" else 1
            continue
        m = _TERM.match(tok)
        if not m:
            raise UnitError(f"cannot parse unit term {tok!r} in {unit!r}")
        code, exp = m.group(1), int(m.group(2) or 1)
        f, d = _atom(code)
        power = sign * exp
        factor *= f ** power
        dim = [a + power * b for a, b in zip(dim, d)]
    return factor, tuple(dim)


def dimension_name(dim):
    if dim == _NONE:
        return "dimensionless"
    parts = [f"{n}^{e}" if e != 1 else n for n, e in zip(_NAMES, dim) if e]
    return ".".join(parts)


def _amount_for_mass(a, b):
    """True when two dimensions differ only by swapping amount for mass.

    The common near-miss at this boundary: nM against ug/mL, mg/kg against
    nmol/kg. Worth naming in the error, because the fix is never a unit
    string -- it is a molar mass, which belongs to a named substance.
    """
    iM, iN = _NAMES.index("M"), _NAMES.index("N")
    rest = [i for i in range(len(_NAMES)) if i not in (iM, iN)]
    if any(a[i] != b[i] for i in rest):
        return False
    return (a[iM], a[iN]) != (b[iM], b[iN]) and a[iM] + a[iN] == b[iM] + b[iN]


def _combine(a, b, op):
    """Unit string for a*b or a/b, written as UCUM without simplifying."""
    left = a if a not in ("", "1") else "1"
    right = b if b not in ("", "1") else "1"
    if right == "1":
        return left
    if left == "1" and op == ".":
        return right
    return f"{left}{op}({right})" if re.search(r"[./*]", right) else f"{left}{op}{right}"


@dataclass(frozen=True)
class Q:
    """A number with a unit. Immutable; arithmetic returns new Qs.

    Equality is semantic and exact: two Qs are equal when they have the same
    dimension and the same value in base units, so Q(1, "L") == Q(1000, "mL").
    Exact means exact -- for anything computed, use ``isclose``.
    """
    value: float
    unit: str = "1"
    _parsed: tuple = field(default=None, repr=False, compare=False, hash=False)

    def __post_init__(self):
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "unit", "1" if self.unit is None else str(self.unit).strip() or "1")
        object.__setattr__(self, "_parsed", parse_unit(self.unit))

    # --- reading ---------------------------------------------------------
    @property
    def dimension(self):
        return self._parsed[1]

    @property
    def base_value(self):
        """The value in SI base units (kg, m, s, mol, K, A, cd)."""
        return self.value * self._parsed[0]

    def is_dimensionless(self):
        return self.dimension == _NONE

    def in_(self, unit):
        """This quantity's value expressed in *unit*, as a plain float.

        The one function that takes a Q out of the protocol layer and into
        the model, where numbers are unitless.
        """
        factor, dim = parse_unit(unit)
        if dim != self.dimension:
            msg = (f"cannot express {self} in {unit!r}: "
                   f"{dimension_name(self.dimension)} is not {dimension_name(dim)}")
            if _amount_for_mass(self.dimension, dim):
                msg += (" -- a mole/mass conversion needs a molar mass, which "
                        "a Q does not carry; that is what naming the Reagent "
                        "is for")
            raise UnitError(msg)
        return self.base_value / factor

    def to(self, unit):
        """The same quantity as a Q in *unit*."""
        return Q(self.in_(unit), unit)

    def isclose(self, other, rel_tol=1e-9, abs_tol=0.0):
        other = Q.parse(other)
        if other.dimension != self.dimension:
            return False
        return math.isclose(self.base_value, other.base_value,
                            rel_tol=rel_tol, abs_tol=abs_tol)

    # --- arithmetic ------------------------------------------------------
    def _same_dim(self, other, what):
        other = Q.parse(other)
        if other.dimension != self.dimension:
            raise UnitError(f"cannot {what} {self} and {other}: "
                            f"{dimension_name(self.dimension)} vs "
                            f"{dimension_name(other.dimension)}")
        return other

    def __add__(self, other):
        o = self._same_dim(other, "add")
        return Q(self.value + o.in_(self.unit), self.unit)

    def __sub__(self, other):
        o = self._same_dim(other, "subtract")
        return Q(self.value - o.in_(self.unit), self.unit)

    def __neg__(self):
        return Q(-self.value, self.unit)

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Q(self.value * other, self.unit)
        other = Q.parse(other)
        return Q(self.value * other.value, _combine(self.unit, other.unit, "."))

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Q(self.value / other, self.unit)
        other = Q.parse(other)
        return Q(self.value / other.value, _combine(self.unit, other.unit, "/"))

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return Q(other / self.value, _combine("1", self.unit, "/"))
        return Q.parse(other) / self

    def __eq__(self, other):
        if not isinstance(other, Q):
            return NotImplemented
        return (self.dimension == other.dimension
                and self.base_value == other.base_value)

    def __hash__(self):
        return hash((self.dimension, self.base_value))

    def __lt__(self, other):
        return self.base_value < self._same_dim(other, "compare").base_value

    def __le__(self, other):
        return self.base_value <= self._same_dim(other, "compare").base_value

    def __gt__(self, other):
        return self.base_value > self._same_dim(other, "compare").base_value

    def __ge__(self, other):
        return self.base_value >= self._same_dim(other, "compare").base_value

    # --- text ------------------------------------------------------------
    def __str__(self):
        num = f"{self.value:g}"
        return num if self.unit == "1" else f"{num} {self.unit}"

    def __repr__(self):
        return f"Q({self.value!r}, {self.unit!r})"

    @classmethod
    def parse(cls, text):
        """Q from "<number> <unit>", a bare number, or a Q (returned as is)."""
        if isinstance(text, Q):
            return text
        if isinstance(text, (int, float)):
            return cls(text, "1")
        if text is None:
            raise UnitError("cannot parse None as a quantity")
        s = str(text).strip()
        m = re.match(r"^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*(.*)$", s)
        if not m:
            raise UnitError(f"cannot parse {text!r} as a quantity")
        return cls(float(m.group(1)), m.group(2) or "1")

    def to_json(self):
        return str(self)

    @classmethod
    def from_json(cls, text):
        return cls.parse(text)
