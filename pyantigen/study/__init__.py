"""Study-centred experimental designs.

A Study is a paper: who was studied (subjects), what was done (protocols and
factor levels), what was simulated (subject x protocol x levels) and every
dataset it reports (assays, on simulations or contrasts). An Optimization is
a fit: the parameters it estimates and the datasets it uses, from one or
more studies. Both are authored in Python and recorded as factored JSON;
``describe`` prints the per-simulation view. See ``docs/V2_DESIGN.md``.

A Study may also declare the substances it used (``Reagent``, with ``Q`` for
quantities with units). Those are recorded and printed and read by nothing
else: the Engine stays unitless, and declaring a reagent cannot change a
simulation. See ``docs/PROTOCOL_LAYER.md``.

    from pyantigen.study import (Study, Measured, Obs, DataSource, Noise, Contrast,
                                 Optimization, Param)
"""
from .assay import Assay, Contrast, DataSource, Measured, Noise, Obs
from .describe import describe
from .design import Factor, Protocol, Simulation, Study, Subject
from .lower_v1 import lower
from .optimization import Optimization, Use, load_optimization, save_optimization
from .params import Param, Rule, by_name, resolve
from .quantity import Q, UnitError, parse_unit
from .reagent import Potency, Reagent
from .remarks import Remarks
from .serialize import fingerprint, from_dict, load, save, to_dict
from .validate import validate, validate_optimization

__all__ = [
    "Assay", "Contrast", "DataSource", "Measured", "Noise", "Obs", "describe",
    "Factor", "Protocol", "Simulation", "Study", "Subject",
    "Optimization", "Use", "load_optimization", "save_optimization",
    "lower", "Param", "Rule", "by_name", "resolve", "Remarks",
    "Q", "UnitError", "parse_unit", "Reagent", "Potency",
    "fingerprint", "from_dict", "load", "save", "to_dict", "validate",
    "validate_optimization",
]
