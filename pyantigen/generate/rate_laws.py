"""
Rate law semantics and volumetric scaling for PyAntiGen.

Each rate type has defined units and explicit volume scaling behavior so that
generated Antimony has correct [amount/time] ODE terms.

Units summary:
- MA / RMA: User supplies [concentration]^n / time; framework multiplies by V_compartment
  to yield [amount/time]. Single-compartment reactions.
- UDF / BDF: User supplies [volume/time] (flow rate); framework uses concentration
  of species (no global V multiplier). Trans-compartment transport.
- custom_conc_per_time: User equation in [concentration]^n/time; framework multiplies
  by V_compartment. Set species_names_are_conc_per_time=True (default).
- custom_amt_per_time: User equation already in [amount/time]; no volume scaling.
- custom: Raw expression as-is; user must include volume scaling if needed.
"""
from abc import ABC, abstractmethod
from typing import Optional


class RateLaw(ABC):
    """Base for rate law semantics (units and volume scaling)."""

    @property
    @abstractmethod
    def multiplies_by_volume(self) -> bool:
        """Whether the framework multiplies the rate expression by compartment volume."""
        pass

    @property
    @abstractmethod
    def user_units_description(self) -> str:
        """Expected units of the user-supplied rate expression."""
        pass


class MassActionLaw(RateLaw):
    """MA / RMA: [concentration]^n / time; framework multiplies by V_compartment."""
    rate_type = "MA"

    @property
    def multiplies_by_volume(self) -> bool:
        return True

    @property
    def user_units_description(self) -> str:
        return "[concentration]^n / time → framework multiplies by V_compartment to yield amount/time"


class VolumeTransportLaw(RateLaw):
    """UDF / BDF: [volume/time]; species appear as concentration; no V multiplier on expression."""
    rate_type = "UDF"

    @property
    def multiplies_by_volume(self) -> bool:
        return False

    @property
    def user_units_description(self) -> str:
        return "[volume/time]; species in equation are concentrations; no V_compartment multiplier"


class CustomLaw(RateLaw):
    """
    custom_conc_per_time / custom_amt_per_time / custom.
    custom_conc_per_time: equation in conc/time → we multiply by V.
    custom_amt_per_time: equation already in amount/time → no scaling.
    custom: as-is; user must include volume scaling if needed.
    """
    def __init__(
        self,
        rate_type: str = "custom",
        species_names_are_conc_per_time: Optional[bool] = None,
    ):
        self.rate_type = rate_type
        self._mult_vol = (
            species_names_are_conc_per_time
            if species_names_are_conc_per_time is not None
            else (rate_type == "custom_conc_per_time")
        )

    @property
    def multiplies_by_volume(self) -> bool:
        return self._mult_vol

    @property
    def user_units_description(self) -> str:
        if self.multiplies_by_volume:
            return "[concentration]^n/time → framework multiplies by V_compartment"
        return "Equation already in [amount/time]; no volume scaling applied"


def get_rate_law_info(rate_type: str) -> str:
    """Return a short docstring for the given rate type (units and scaling)."""
    if rate_type in ("MA", "RMA"):
        return MassActionLaw(rate_type).user_units_description
    if rate_type in ("UDF", "BDF"):
        return VolumeTransportLaw(rate_type).user_units_description
    if rate_type in ("custom_conc_per_time", "custom_amt_per_time", "custom"):
        return CustomLaw(rate_type).user_units_description
    return "Unknown rate type"
