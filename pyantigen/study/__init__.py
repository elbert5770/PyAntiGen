"""Study-centred experimental designs.

A Study declares who was studied (subjects), what was done (protocols and
factor levels), what was simulated (subject x protocol x levels), what was
measured (assays, scoring simulations or contrasts) and which parameters are
estimated. It is authored in Python and recorded as one factored JSON file;
``describe`` prints the per-simulation view. See ``docs/V2_DESIGN.md``.

    from pyantigen.study import Study, Measured, Obs, DataSource, Noise, Contrast, Estimation
"""
from .assay import Assay, Contrast, DataSource, Measured, Noise, Obs
from .describe import describe
from .design import Factor, Protocol, Simulation, Study, Subject
from .lower_v1 import Estimation, lower
from .params import Param, Rule, by_name, resolve
from .remarks import Remarks
from .serialize import fingerprint, from_dict, load, save, to_dict
from .validate import validate

__all__ = [
    "Assay", "Contrast", "DataSource", "Measured", "Noise", "Obs", "describe",
    "Factor", "Protocol", "Simulation", "Study", "Subject",
    "Estimation", "lower", "Param", "Rule", "by_name", "resolve", "Remarks",
    "fingerprint", "from_dict", "load", "save", "to_dict", "validate",
]
