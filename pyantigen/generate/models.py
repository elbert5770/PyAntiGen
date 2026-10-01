"""
Structured models for PyAntiGen reactions and species.
Replaces raw dict usage with validated dataclasses for clearer types and early validation.
"""
from dataclasses import dataclass
from typing import List, Union, Optional

# Valid rate types; RMA and BDF require two rate constants in Rate_eqtn_prototype
VALID_RATE_TYPES = frozenset({
    "RMA", "BDF", "MA", "UDF",
    "custom_conc_per_time", "custom_amt_per_time", "custom"
})

RATE_TYPES_TWO_CONSTANTS = frozenset({"RMA", "BDF"})


def normalize_species_list(value: Union[str, List[str], None]) -> List[str]:
    """
    Normalize reactants/products to a list of species name strings.
    Accepts: None, list of strings, or string (e.g. '[A, B]' or '[A] + [B]').
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(s).strip() for s in value if s is not None and str(s).strip()]
    s = str(value).strip()
    if not s or s in ("0", "[0]"):
        return []
    return parse_species_list(s)


def _ensure_string_list(value: Union[str, List[str]]) -> List[str]:
    """Alias for normalize_species_list for internal use."""
    return normalize_species_list(value)


def parse_species_list(s: str) -> List[str]:
    """
    Parse a species list from string, supporting:
      - '[A, B, C]' (comma-separated inside brackets)
      - '[A] + [B]' or '[A]+[B]' (bracket-wrapped terms with +)
    Returns list of stripped species names. Uses regex to extract bracketed or comma-separated tokens.
    """
    import re
    s = s.strip()
    if not s:
        return []
    # Already a single token (no brackets, no comma)
    if not re.search(r'[\[\],+]', s):
        return [s] if s and s != "0" else []
    inner = s.strip()
    # Only remove outer brackets for "[A, B]" style (single list), not "[A] + [B]"
    if inner.startswith('[') and inner.endswith(']') and '+' not in inner[1:-1]:
        inner = inner[1:-1].strip()
    if not inner:
        return []
    # Split by comma only if we're in bracket list form (no + in the middle)
    if '+' not in inner:
        return [x.strip() for x in re.split(r'\s*,\s*', inner) if x.strip()]
    # "[A] + [B]" or "A + B" form: split by + and clean each part
    parts = re.split(r'\s*\+\s*', inner)
    result = []
    for p in parts:
        p = p.strip()
        if p.startswith('[') and p.endswith(']'):
            p = p[1:-1].strip()
        if p and p != "0":
            result.append(p)
    return result


def parse_rate_equation_list(rate_proto: Union[str, List[str]], rate_type: str) -> List[str]:
    """
    Parse Rate_eqtn_prototype into a list of one or two rate expression strings.
    For RMA/BDF we expect two constants; for others one.
    """
    if rate_proto is None:
        return []
    if isinstance(rate_proto, list):
        return [str(x).strip() for x in rate_proto if str(x).strip()]
    s = str(rate_proto).strip()
    if not s:
        return []
    if s.startswith('[') and s.endswith(']'):
        inner = s[1:-1].strip()
        return [x.strip() for x in inner.split(',') if x.strip()]
    return [s]


@dataclass(frozen=True)
class Species:
    """A species with optional explicit compartment (otherwise inferred from name suffix)."""
    name: str
    compartment: Optional[str] = None

    def resolve_compartment(self) -> str:
        """Compartment to use: explicit if set, else last segment after underscore."""
        if self.compartment is not None:
            return self.compartment
        if '_' in self.name:
            return self.name.split('_')[-1]
        return self.name


@dataclass
class Reaction:
    """
    A single reaction with validated fields.
    Reactants and Products are stored as lists of species names.
    Rate_eqtn_prototype is one string (MA, UDF, custom) or two (RMA, BDF).
    """
    Reaction_name: str
    Reactants: List[str]
    Products: List[str]
    Rate_type: str
    Rate_eqtn_prototype: Union[str, List[str]]
    # Optional: explicit compartment(s) so framework doesn't infer from name suffix
    compartment: Optional[str] = None
    compartment_reverse: Optional[str] = None  # for product side when different

    def __post_init__(self):
        r = self.Reaction_name.replace(" ", "") if self.Reaction_name else "NA"
        object.__setattr__(self, 'Reaction_name', r)
        if self.Rate_type and self.Rate_type not in VALID_RATE_TYPES:
            raise ValueError(
                f"Invalid Rate_type '{self.Rate_type}'. "
                f"Valid types: {', '.join(sorted(VALID_RATE_TYPES))}"
            )
        rate_list = parse_rate_equation_list(self.Rate_eqtn_prototype, self.Rate_type)
        if self.Rate_type == "RMA" and len(rate_list) < 2:
            raise ValueError(
                f"RMA reaction requires two rate constants in Rate_eqtn_prototype, got: {self.Rate_eqtn_prototype!r}"
            )
        if self.Rate_type == "BDF" and len(rate_list) < 1:
            raise ValueError(
                f"BDF reaction requires at least one rate constant in Rate_eqtn_prototype, got: {self.Rate_eqtn_prototype!r}"
            )

    def to_dict(self) -> dict:
        """Export as dict for serialization and RxnDict_to_antimony compatibility."""
        # Serialize reactants/products as "[A, B]" for file format compatibility
        def to_bracket(lst):
            if not lst:
                return "[0]"
            return "[" + ", ".join(lst) + "]"
        rate = self.Rate_eqtn_prototype
        if isinstance(rate, list):
            rate = "[" + ", ".join(rate) + "]"
        d = {
            "Reaction_name": self.Reaction_name,
            "Reactants": to_bracket(self.Reactants),
            "Products": to_bracket(self.Products),
            "Rate_type": self.Rate_type,
            "Rate_eqtn_prototype": rate,
        }
        if self.compartment is not None:
            d["compartment"] = self.compartment
        if self.compartment_reverse is not None:
            d["compartment_reverse"] = self.compartment_reverse
        return d


def reaction_from_args(
    name: str,
    reactants: Union[str, List[str]],
    products: Union[str, List[str]],
    rate_type: str,
    rate_eqtn: Union[str, List[str]],
    compartment: Optional[str] = None,
    compartment_reverse: Optional[str] = None,
) -> Reaction:
    """Build and validate a Reaction from add_reaction-style arguments."""
    if not name or not name.strip():
        name = "NA"
    if rate_type is None or rate_type == "":
        raise ValueError("Rate_type is required")
    if rate_eqtn is None or (isinstance(rate_eqtn, str) and rate_eqtn.strip() == ""):
        raise ValueError("Rate_eqtn_prototype is required")
    reactants_list = _ensure_string_list(reactants)
    products_list = _ensure_string_list(products)
    return Reaction(
        Reaction_name=name.strip().replace(" ", ""),
        Reactants=reactants_list,
        Products=products_list,
        Rate_type=rate_type.strip(),
        Rate_eqtn_prototype=rate_eqtn,
        compartment=compartment,
        compartment_reverse=compartment_reverse,
    )
