"""The design JSON: one file per study, factored, never expanded.

What is written is what was declared -- factor tables, subjects, protocol
references, the occasion GENERATORS (a ``cross`` is one record, not its
product), assays, contrasts, parameters and rules. Nothing is repeated per
occasion, so the file grows with the sum of the factor tables, not their
product.

No dates or hashes are written into the design. A run records the study's
``fingerprint`` (SHA-256 of the canonical JSON) alongside its results, which
is what ties a result to the exact design that produced it; editing the
design changes the fingerprint and nothing else has to be renamed.
"""
import hashlib
import json

from .assay import Assay, Contrast, Measurement
from .design import Factor, Protocol, Study, Subject
from .params import Param, Rule

FORMAT = 1


# A factor whose levels all carry the same attribute and parameter names is
# written as a table -- names once, then one row of values per level -- once it
# has at least this many levels. Below that the plain {level: {...}} form is
# easier to read and no larger.
TABLE_MIN_LEVELS = 8


def _factor_out(levels):
    keys = None
    for attrs in levels.values():
        k = (tuple(a for a in attrs if a != "params"), tuple(attrs.get("params", {})))
        if keys is None:
            keys = k
        elif k != keys:
            return levels
    if len(levels) < TABLE_MIN_LEVELS or keys is None:
        return levels
    cols, pcols = keys
    return {"table": {
        "columns": list(cols) + [f"params.{c}" for c in pcols],
        "rows": {lev: [a[c] for c in cols] + [a["params"][c] for c in pcols]
                 for lev, a in levels.items()}}}


def _factor_in(levels):
    if set(levels) != {"table"} or not isinstance(levels["table"], dict):
        return levels
    t = levels["table"]
    out = {}
    for lev, row in t["rows"].items():
        if len(row) != len(t["columns"]):
            raise ValueError(f"factor table row {lev!r} has {len(row)} values for "
                             f"{len(t['columns'])} columns")
        attrs, params = {}, {}
        for c, v in zip(t["columns"], row):
            if c.startswith("params."):
                params[c[len("params."):]] = v
            else:
                attrs[c] = v
        if params:
            attrs["params"] = params
        out[lev] = attrs
    return out


def to_dict(study):
    d = {"pyantigen_design": FORMAT, "name": study.name}
    if study.remarks:
        d["remarks"] = list(study.remarks)
    d["factors"] = {n: _factor_out(f.levels) for n, f in study.factors.items()}
    subs = {}
    for sid, s in study.subjects.items():
        e = {}
        if s.kind != "individual":
            e["kind"] = s.kind
        if s.covariates:
            e["covariates"] = s.covariates
        if s.levels:
            e["levels"] = s.levels
        subs[sid] = e
    d["subjects"] = subs
    d["protocols"] = {n: p.to_json() for n, p in study.protocols.items()}
    d["occasions"] = study._occasion_records
    if study.assays:
        d["assays"] = {n: a.to_json() for n, a in study.assays.items()}
    if study.contrasts:
        d["contrasts"] = {n: c.to_json() for n, c in study.contrasts.items()}
    if study.measurements:
        d["measurements"] = [m.to_json() for m in study.measurements]
    if study.params:
        d["params"] = {n: p.to_json() for n, p in study.params.items()}
    if study.rules:
        d["rules"] = [r.to_json() for r in study.rules]
    return d


def from_dict(d):
    if d.get("pyantigen_design") != FORMAT:
        raise ValueError(f"not a pyantigen design (format {FORMAT}): "
                         f"pyantigen_design={d.get('pyantigen_design')!r}")
    s = Study(d["name"], remarks=d.get("remarks"))
    for n, levels in d.get("factors", {}).items():
        s.factors[n] = Factor(n, _factor_in(levels))
    for sid, e in d.get("subjects", {}).items():
        s.subjects[sid] = Subject(sid, e.get("kind", "individual"),
                                  e.get("covariates", {}), e.get("levels", {}))
    for n, p in d.get("protocols", {}).items():
        s.protocols[n] = Protocol.from_json(n, p)
    for rec in d.get("occasions", []):
        if isinstance(rec, dict) and "cross" in rec:
            c = rec["cross"]
            s.cross(c["subjects"], c["protocol"], **c["factors"])
        else:
            subject, protocol, levels = rec[0], rec[1], rec[2]
            extra = rec[3] if len(rec) > 3 else {}
            s.occasion(subject, protocol, period=extra.get("period"),
                       id=extra.get("id"), **levels)
    for n, a in d.get("assays", {}).items():
        s.assays[n] = Assay.from_json(n, a)
    for n, c in d.get("contrasts", {}).items():
        s.contrasts[n] = Contrast.from_json(n, c)
    s.measurements = [Measurement.from_json(m) for m in d.get("measurements", [])]
    for n, p in d.get("params", {}).items():
        s.params[n] = Param.from_json(n, p)
    s.rules = [Rule.from_json(r) for r in d.get("rules", [])]
    return s


def canonical(study):
    return json.dumps(to_dict(study), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def fingerprint(study):
    return hashlib.sha256(canonical(study).encode("utf-8")).hexdigest()


def dumps(obj, width=88, _indent=0):
    """JSON with every container that fits in *width* kept on one line.

    Plain ``indent=`` puts each number of a bounds pair on its own line; a
    design is read by people and LLMs far more than it is written, so short
    things stay short.
    """
    flat = json.dumps(obj, ensure_ascii=False)
    if _indent + len(flat) <= width or not isinstance(obj, (dict, list)) or not obj:
        return flat
    inner = " " * (_indent + 1)
    if isinstance(obj, dict):
        items = [f"{inner}{json.dumps(k, ensure_ascii=False)}: {dumps(v, width, _indent + 1)}"
                 for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + " " * _indent + "}"
    items = [inner + dumps(v, width, _indent + 1) for v in obj]
    return "[\n" + ",\n".join(items) + "\n" + " " * _indent + "]"


def save(study, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(dumps(to_dict(study)) + "\n")
    return path


def load(path):
    with open(path, encoding="utf-8") as f:
        return from_dict(json.load(f))
