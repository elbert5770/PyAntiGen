"""Unit checks on pyantigen.study that need no model and no RoadRunner.

Run from the repository root: python -m pytest tests/study
"""
import json
import os

import pytest

from pyantigen.study import (Contrast, DataSource, Estimation, Measured, Noise, Obs, Remarks,
                             Study, by_name, describe, fingerprint, from_dict, lower,
                             resolve, to_dict, validate)
from pyantigen.study.assay import ratio_pct


def _events(replicate, df_dict, r_ic=None):
    return ""


def _solver(replicate):
    return {}


def _observed(r):
    return ["time"]


def crossover():
    """A crossover: animals x dose, every animal gets every dose."""
    s = Study("crossover")
    s.factor("dose", {"0": {"mgkg": 0}, "30": {"mgkg": 30}, "125": {"mgkg": 125}})
    for a in ("M1", "M2", "M3"):
        s.subject(a, covariates={"weight_kg": 8.0})
    s.protocol("p", _events, _solver, _observed)
    s.simulate_all(["M1", "M2", "M3"], "p", dose="*")
    return s


# --- design ------------------------------------------------------------------

def test_simulate_all_names_simulations_by_subject_then_levels():
    s = crossover()
    assert list(s.simulations)[:3] == ["M1_0", "M1_30", "M1_125"]
    assert len(s.simulations) == 9


def test_attributes_merge_covariates_levels_and_reserved_keys():
    a = crossover().attributes("M2_30")
    assert a["weight_kg"] == 8.0 and a["mgkg"] == 30 and a["dose"] == "30"
    assert a["subject"] == "M2" and a["simulation"] == "M2_30"


def test_select_unknown_attribute_raises_instead_of_matching_nothing():
    with pytest.raises(KeyError, match="dsoe"):
        crossover().select(dsoe="30")


def test_between_subject_level_cannot_be_overridden_per_simulation():
    s = Study("x")
    s.factor("status", {"neg": {}, "pos": {}})
    s.subject("A", status="neg")
    s.protocol("p", _events, _solver, _observed)
    with pytest.raises(ValueError, match="fixed"):
        s.simulate("A", "p", status="pos")


# --- contrasts: pairing is within subject by default --------------------------

def test_contrast_pairs_each_dose_with_the_same_animals_vehicle():
    s = crossover()
    c = Contrast.ratio_pct(numerator={"dose": ["30", "125"]}, denominator={"dose": "0"})
    pairs = [(n.id, d.id) for n, d in c.pairs(s)]
    assert pairs == [("M1_30", "M1_0"), ("M1_125", "M1_0"), ("M2_30", "M2_0"),
                     ("M2_125", "M2_0"), ("M3_30", "M3_0"), ("M3_125", "M3_0")]


def test_contrast_with_a_missing_control_raises():
    s = Study("x")
    s.factor("dose", {"0": {}, "30": {}})
    s.subject("M1")
    s.subject("M2")
    s.protocol("p", _events, _solver, _observed)
    s.simulate("M1", "p", dose="0")
    s.simulate("M1", "p", dose="30")
    s.simulate("M2", "p", dose="30")          # M2 never had vehicle
    c = Contrast.ratio_pct({"dose": "30"}, {"dose": "0"})
    with pytest.raises(ValueError, match="M2_30.*no matching"):
        c.pairs(s)


def test_between_pairing_for_parallel_groups():
    s = Study("x")
    s.factor("arm", {"drug": {}, "placebo": {}})
    s.subject("drug_cohort", kind="cohort", arm="drug")
    s.subject("placebo_cohort", kind="cohort", arm="placebo")
    s.protocol("p", _events, _solver, _observed)
    s.simulate("drug_cohort", "p")
    s.simulate("placebo_cohort", "p")
    c = Contrast.diff({"arm": "drug"}, {"arm": "placebo"}, pairing="between")
    assert [(n.id, d.id) for n, d in c.pairs(s)] == [("drug_cohort", "placebo_cohort")]
    with pytest.raises(ValueError, match="no matching"):
        Contrast.diff({"arm": "drug"}, {"arm": "placebo"}).pairs(s)


def test_ratio_pct():
    assert list(ratio_pct([[50.0, 20.0], [100.0, 40.0]])) == [50.0, 50.0]


# --- parameters ----------------------------------------------------------------

def test_resolve_order_is_levels_then_fixed_then_fitted_then_rules():
    s = Study("x")
    s.factor("drug", {"Lec": {"params": {"k": 1.0, "V": 3.0}}, "Adu": {"params": {"k": 2.0}}})
    s.factor("status", {"neg": {"amyloid_positive": False}, "pos": {"amyloid_positive": True}})
    s.subject("S", status="neg")
    s.protocol("p", _events, _solver, _observed)
    s.simulate_all(["S"], "p", drug="*")
    s.param("kf", x0=0.5, by="drug")
    s.param("SF", x0=1.0)
    s.rule("k_oligo", 0.0, amyloid_positive=False)
    theta = {by_name("kf", "drug", "Lec"): 0.7, by_name("kf", "drug", "Adu"): 0.9,
             "SF": 2.5, "V": 99.0}
    assert resolve(s, "S_Lec", None) == {"k": 1.0, "V": 3.0, "k_oligo": 0.0}
    assert resolve(s, "S_Lec", theta) == {"k": 1.0, "V": 3.0, "kf": 0.7, "SF": 2.5, "k_oligo": 0.0}
    assert resolve(s, "S_Adu", theta)["kf"] == 0.9
    # "V" in theta is not a declared Param, so it is not applied.
    assert "V" not in resolve(s, "S_Adu", theta)


def test_rule_beats_a_fitted_value():
    s = Study("x")
    s.factor("status", {"neg": {"amyloid_positive": False}})
    s.subject("S", status="neg")
    s.protocol("p", _events, _solver, _observed)
    s.simulate("S", "p")
    s.param("k_oligo", x0=1e-3)
    s.rule("k_oligo", 0.0, amyloid_positive=False)
    assert resolve(s, "S", {"k_oligo": 5e-3})["k_oligo"] == 0.0


def test_by_level_params_only_for_levels_that_occur():
    s = crossover()
    s.param("KI", x0=1.0, by="dose")
    assert s.params["KI"].fitted_names(s) == ["KI[dose=0]", "KI[dose=30]", "KI[dose=125]"]


# --- JSON ----------------------------------------------------------------------

def with_assay(s):
    s.assay("csf", Measured("AB40", Obs("AB40_CM"),
                            DataSource("d.csv", "t", "v", where={"animal": "{subject}"}),
                            Noise(pool=["dose"])),
            on=Contrast.ratio_pct({"dose": ["30", "125"]}, {"dose": "0"}))
    s.param("KI", x0=0.3, bounds=(1e-5, 1e3), scale="log10")
    s.rule("k_oligo", 0.0, weight_kg=8.0)
    return s


def test_json_round_trip_is_exact():
    s = with_assay(crossover())
    d = to_dict(s)
    s2 = from_dict(json.loads(json.dumps(d)))
    assert to_dict(s2) == d
    assert fingerprint(s2) == fingerprint(s)
    assert list(s2.simulations) == list(s.simulations)


def test_json_stores_simulate_all_not_its_expansion():
    d = to_dict(crossover())
    assert d["simulations"] == [{"all": {"subjects": ["M1", "M2", "M3"], "protocol": "p",
                                         "factors": {"dose": "*"}}}]


def test_callables_are_references():
    d = to_dict(crossover())
    assert d["protocols"]["p"]["events"].endswith("test_design:_events")


def test_froehlich_sized_design_stays_small():
    """290 cell lines x 33 treatments = 9,570 conditions, Froehlich 2018's count.

    PEtab stores that as a 9,570 x 147 table (17 MB) because each row repeats
    its cell line's 122 expression values. Factored, each value is written once.
    """
    s = Study("froehlich_like")
    s.factor("cell_line", {f"CL{i:03d}": {"params": {f"r{j}_k_RPKM2protein": 10.0 + i + j / 1000
                                                          for j in range(122)}}
                           for i in range(290)})
    s.factor("treatment", {f"T{i:02d}": {"params": {f"SP_{j}": float(i * j) for j in range(23)}}
                           for i in range(33)})
    s.subject("line", kind="cohort")
    s.protocol("p", _events, _solver, _observed)
    s.simulate_all(["line"], "p", cell_line="*", treatment="*")
    assert len(s.simulations) == 9570
    from pyantigen.study.serialize import dumps
    text = dumps(to_dict(s))
    # The real PEtab file is 18,060,637 bytes; this design is ~0.54 MB.
    assert len(text) < 18_060_637 / 30, len(text)
    assert '"table"' in text          # the 290-level factor is written as a table
    s2 = from_dict(json.loads(text))
    assert to_dict(s2) == to_dict(s)
    assert resolve(s2, "line_CL007_T05")["r3_k_RPKM2protein"] == 10.0 + 7 + 0.003


# --- lowering -------------------------------------------------------------------

def test_lowering_builds_one_composite_per_subject_pair_and_pools_sigma():
    s = with_assay(crossover())
    exp, spec = lower(s, Estimation("e"))
    elems = spec.groups["crossover"]["loss_elements"]
    assert [e["simulations"] for e in elems][:2] == [["M1_30", "M1_0"], ["M1_125", "M1_0"]]
    # pooled across dose: one block per animal
    assert {e["sigma_block"] for e in elems} == {"csf.AB40|subject=M1", "csf.AB40|subject=M2",
                                                 "csf.AB40|subject=M3"}
    assert spec.param_names == ["KI"] and spec.parameter_scale == {"KI": "log10"}
    assert set(exp.replicates) == set(s.simulations)
    assert exp.replicates["M1_30"]["mgkg"] == 30


def test_lowered_hooks_apply_resolve_in_full():
    s = with_assay(crossover())
    exp, _ = lower(s, Estimation("e"))
    hook = exp.replicates["M1_30"]["Update_opt_parameters"]
    assert hook.values({"KI": 0.02}) == {"KI": 0.02, "k_oligo": 0.0}


def test_lowered_replicates_pickle():
    import pickle
    s = with_assay(crossover())
    exp, spec = lower(s, Estimation("e"))
    pickle.loads(pickle.dumps((exp.replicates, spec)))


# --- validation ------------------------------------------------------------------

def test_validate_reports_design_mistakes():
    s = crossover()
    s.assay("a", Measured("x", Obs("x"), DataSource("d.csv", "t", "v"),
                          only={"has_dataa": True}))
    s.simulate("M1", "p", dose="30", id="M1_30_again")
    msgs = [str(p) for p in validate(s)]
    assert any("unknown attribute(s) ['has_dataa']" in m for m in msgs)
    assert any("'M1_30' and 'M1_30_again' are the same simulation" in m for m in msgs)


def test_validate_checks_remark_ids(tmp_path):
    rm = Remarks(str(tmp_path / "remarks.json"))
    rid = rm.add("Why vehicle ratio", "<p>because</p>")
    ok = Study("x", remarks=[rid])
    bad = Study("x", remarks=[rid, "2026-01-01-nope"])
    assert not [p for p in validate(ok, remarks=rm) if p.level == "error"]
    assert any("nope" in str(p) for p in validate(bad, remarks=rm))


# --- remarks ----------------------------------------------------------------------

def test_remarks_are_timestamped_append_only_and_linkable(tmp_path):
    import datetime as dt
    path = str(tmp_path / "remarks.json")
    rm = Remarks(path)
    t = dt.datetime(2026, 9, 23, 10, 14)
    a = rm.add("Drug over vehicle", "<p>v1</p>", now=t)
    b = rm.add("Drug over vehicle", "<p>v2</p>", supersedes=a, now=t)
    assert a == "2026-09-23-drug-over-vehicle" and b == a + "-2"
    rm2 = Remarks(path)
    assert rm2.get(b)["supersedes"] == a and rm2.link(a) == f"remarks.html#{a}"
    html_path = rm2.render_html()
    text = open(html_path, encoding="utf-8").read()
    assert f"id='{a}'" in text and "superseded by" in text
    with pytest.raises(KeyError):
        rm2.add("x", "y", supersedes="missing")


# --- baseline observables, protocol inputs and hooks ------------------------------

class _Result(dict):
    """Minimal stand-in for a RoadRunner result: columns by name, colnames."""
    @property
    def colnames(self):
        return list(self)


def test_baseline_observable_matches_hand_arithmetic():
    import numpy as np
    from pyantigen.study.assay import Obs
    o = Obs.of_baseline(["[A]", "[A_L]"], "Age*365.0*24.0 + 1.0", "A_pct")
    t0 = 2.0 * 365.0 * 24.0 + 1.0
    fn = o.engine_form({"Age": 2.0})
    res = _Result({"time": np.array([t0 - 5, t0, t0 + 3]),
                   "[A]": np.array([1.0, 2.0, 4.0]), "[A_L]": np.array([0.0, 0.5, 1.0])})
    assert fn.__name__ == "A_pct" and fn.t0 == t0
    assert list(fn(res)) == list(100.0 * (res["[A]"] + res["[A_L]"]) / 2.5)
    import pickle
    assert list(pickle.loads(pickle.dumps(fn))(res)) == list(fn(res))


def test_baseline_time_is_arithmetic_only():
    from pyantigen.study.assay import Obs
    with pytest.raises(ValueError):
        Obs.of_baseline(["x"], "__import__('os').getcwd()", "bad")
    with pytest.raises(KeyError, match="Agee"):
        Obs.of_baseline(["x"], "Agee*2", "n").engine_form({"Age": 1})


def _loader(replicate, data_path):
    import pandas as pd
    return {"pk": pd.DataFrame({"t": [0.0, 1.0]}),
            "prepared": pd.DataFrame({"t": [1.0, 2.0, 3.0], "v": [5.0, None, 7.0],
                                      "arm": [replicate["dose"]] * 3})}


def _hook(r, replicate):
    r["hooked"] = 1.0


def _opt_hook(r, replicate, parameters):
    r["opt_hooked"] = r.get("KI", -1)


def test_protocol_inputs_reach_events_and_scored_tables():
    s = crossover()
    s.protocols["p"].data = _loader
    s.assay("a", Measured("v", Obs("v"), DataSource(input="prepared", time="t", value="v")))
    exp, _ = lower(s, Estimation("e", params=[]))
    rep = exp.replicates["M1_30"]
    d = rep["Data"](rep, "unused")
    assert list(d["pk"]["t"]) == [0.0, 1.0]           # passed through for the events
    assert list(d["a.v"]["v"]) == [5.0, 7.0]           # NaN row dropped, column renamed


def test_protocol_hooks_run_after_resolved_values_and_round_trip():
    s = crossover()
    s.protocols["p"].update_parameters = _hook
    s.protocols["p"].update_opt_parameters = _opt_hook
    s.assay("a", Measured("v", Obs("v"), DataSource("d.csv", "t", "v")))
    s.param("KI", x0=0.3)
    exp, _ = lower(s, Estimation("e"))
    r = {}
    exp.replicates["M1_30"]["Update_parameters"](r, {})
    exp.replicates["M1_30"]["Update_opt_parameters"](r, {}, {"KI": 0.02})
    assert r == {"hooked": 1.0, "KI": 0.02, "opt_hooked": 0.02}
    d = to_dict(s)
    assert d["protocols"]["p"]["update_opt_parameters"].endswith(":_opt_hook")
    assert to_dict(from_dict(json.loads(json.dumps(d)))) == d


def test_both_sides_of_a_contrast_pair_hold_the_numerators_rows():
    """The Engine interpolates each composite simulation at its OWN table's
    times and silently drops one without the table; so the denominator must
    carry the numerator's rows, under a key unique to the pair."""
    s = crossover()
    s.protocols["p"].data = _loader
    s.assay("a", Measured("v", Obs("v"), DataSource(input="prepared", time="t", value="v")),
            on=Contrast.ratio_pct({"dose": ["30", "125"]}, {"dose": "0"}))
    exp, spec = lower(s, Estimation("e", params=[]))
    keys = [e["loss_config"]["observables"][0]["data_dict_key"]
            for e in spec.groups["crossover"]["loss_elements"]]
    assert keys[:2] == ["a.v|M1_30/M1_0", "a.v|M1_125/M1_0"]
    veh = exp.replicates["M1_0"]
    d = veh["Data"](veh, "unused")
    # the vehicle holds both numerators' tables, each read through the
    # numerator's own loader call
    assert list(d["a.v|M1_30/M1_0"]["v"]) == [5.0, 7.0]
    assert set(d) >= {"a.v|M1_30/M1_0", "a.v|M1_125/M1_0"}


def test_remark_ids_never_end_in_a_hyphen(tmp_path):
    rm = Remarks(str(tmp_path / "remarks.json"))
    rid = rm.add("A fitted constant moves once the data are corrected", "<p>x</p>")
    assert not rid.endswith("-") and len(rid) <= 11 + 48


def test_templated_input_names_pick_each_simulations_table():
    s = crossover()
    s.protocols["p"].data = _loader
    for sid in ("M1", "M2", "M3"):
        s.subjects[sid].covariates["table"] = "prepared"
    s.assay("a", Measured("v", Obs("v"), DataSource(input="{table}", time="t", value="v")))
    exp, _ = lower(s, Estimation("e", params=[]))
    rep = exp.replicates["M2_30"]
    assert list(rep["Data"](rep, "unused")["a.v"]["v"]) == [5.0, 7.0]


def test_peak_normalized_measurement_lowers_to_a_one_simulation_composite():
    from pyantigen.study.assay import peak_normalize
    s = crossover()
    s.assay("gad", Measured("CM", Obs("[G_CM]"), DataSource("d.csv", "t", "v"), normalize="peak"),
            on={"subject": "M1", "dose": "0"})
    _, spec = lower(s, Estimation("e", params=[]))
    (e,) = spec.groups["crossover"]["loss_elements"]
    assert e["simulations"] == ["M1_0"] and e["aggregation"] is peak_normalize
    assert list(peak_normalize([[1.0, 4.0, 2.0]])) == [0.25, 1.0, 0.5]
    assert list(peak_normalize([[0.0, 0.0]])) == [0.0, 0.0]
    d = to_dict(s)
    assert to_dict(from_dict(json.loads(json.dumps(d)))) == d


def test_baseline_tolerance_is_part_of_the_observable():
    import numpy as np
    from pyantigen.study.assay import Obs
    o = Obs.of_baseline(["[A]"], "t0", "A0", tol=0.0)
    res = _Result({"time": np.array([1.0, 2.0 + 5e-7]), "[A]": np.array([2.0, 4.0])})
    assert list(o.engine_form({"t0": 2.0})(res)) == [100.0, 200.0]      # 2+5e-7 is AFTER t0
    loose = Obs.of_baseline(["[A]"], "t0", "A1")
    assert list(loose.engine_form({"t0": 2.0})(res)) == [50.0, 100.0]   # within 1e-6
    assert Obs.from_json(o.to_json()).tol == 0.0


# --- protocol settings, element order, describe ----------------------------------

def test_protocol_settings_are_simulation_attributes():
    s = crossover()
    s.protocols["p"].settings["drain_on"] = True
    assert s.attributes("M1_30")["drain_on"] is True
    assert [x.id for x in s.select(drain_on=True)][:1] == ["M1_0"]
    d = to_dict(s)
    assert d["protocols"]["p"]["settings"] == {"drain_on": True}
    assert to_dict(from_dict(json.loads(json.dumps(d)))) == d


def test_a_setting_that_is_also_a_covariate_is_an_error():
    s = crossover()
    s.protocols["p"].settings["weight_kg"] = 9.0
    assert any("also subject covariates" in str(p) for p in validate(s))


def test_loss_elements_are_in_simulation_order_then_assay_order():
    s = crossover()
    s.assay("b", Measured("y", Obs("y"), DataSource("d.csv", "t", "v")), on={"dose": "30"})
    s.assay("a", Measured("x", Obs("x"), DataSource("d.csv", "t", "v")))
    _, spec = lower(s, Estimation("e", params=[]))
    order = [(e["simulation"], e["loss_config"]["observables"][0]["data_column"])
             for e in spec.groups["crossover"]["loss_elements"]][:4]
    assert order == [("M1_0", "x"), ("M1_30", "y"), ("M1_30", "x"), ("M1_125", "x")]


def test_an_assay_that_scores_nothing_is_an_error():
    s = crossover()
    s.assay("a", Measured("x", Obs("x"), DataSource("d.csv", "t", "v")), on={"dose": "999"})
    assert any("assay 'a': scores no simulations" in str(p) for p in validate(s))


def test_describe_shows_each_simulation_whole():
    s = with_assay(crossover())
    text = describe(s, Estimation("e"))
    block = text.split("\nM1_30 ")[1].split("\n\n")[0]
    assert "level       dose=30  ->  mgkg=30" in block
    assert "covariates  weight_kg=8.0" in block
    assert "rules       k_oligo=0.0 (when weight_kg=8.0)" in block
    assert "fitted      1 global (above)" in block
    assert "KI" in text.split("\n\n")[0]            # listed once, in the header
    assert "csf.AB40  vs M1_0 (ratio_pct, pairing=subject)" in block
    assert "pooled over dose" in block
    vehicle = text.split("\nM1_0 ")[1].split("\n\n")[0]
    assert "scored by   nothing" in vehicle and "control for M1_30 (csf), M1_125 (csf)" in vehicle
