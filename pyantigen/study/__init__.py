"""Study-centred experimental designs.

A Study declares who was studied (subjects and cohorts), what was done
(protocols and factor levels), what was measured (assays, contrasts) and
which parameters are estimated. It is authored in Python and recorded as one
factored JSON file; see ``docs/V2_DESIGN.md``.

    from pyantigen.study import Study, Measured, Obs, DataSource, Noise, Estimation
"""
from .assay import Assay, Contrast, DataSource, Measured, Measurement, Noise, Obs
from .design import Factor, Occasion, Protocol, Study, Subject
from .lower_v1 import Estimation, lower
from .params import Param, Rule, by_name, resolve
from .remarks import Remarks
from .serialize import fingerprint, from_dict, load, save, to_dict
from .validate import validate

__all__ = [
    "Assay", "Contrast", "DataSource", "Measured", "Measurement", "Noise", "Obs",
    "Factor", "Occasion", "Protocol", "Study", "Subject",
    "Estimation", "lower", "Param", "Rule", "by_name", "resolve", "Remarks",
    "fingerprint", "from_dict", "load", "save", "to_dict", "validate",
]
