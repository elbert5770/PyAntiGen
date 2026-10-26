"""Backend choice (pyantigen.engine.Backend): precedence, fallback, and the machine check with a faked machine."""
import importlib.metadata
from dataclasses import dataclass, field

import pytest

from pyantigen.engine import Backend as B

OK = B.AmiciStatus(True, "1.1.0")
BAD = B.AmiciStatus(False, None, ("the amici package is not installed",), ("pip install 'pyantigen[amici]'",))


def test_stage_defaults_split_short_and_long_stages():
    for stage in ("simulate", "fit"):
        assert B.resolve_backend(stage, env={}, status=OK).backend == "roadrunner"
    for stage in ("multistart", "profile", "fim"):
        r = B.resolve_backend(stage, env={}, status=OK)
        assert (r.backend, r.source) == ("amici", "default")


def test_precedence_argument_environment_settings_spec():
    env = {B.ENV_BACKEND: "roadrunner"}
    assert B.resolve_backend("profile", requested="amici", settings_value="roadrunner", env=env, status=OK).source == "argument"
    assert B.resolve_backend("profile", settings_value="amici", env=env, status=OK).backend == "roadrunner"      # environment beats settings
    assert B.resolve_backend("fit", settings_value="amici", env={}, status=OK).source == "settings"
    assert B.resolve_backend("fit", settings_value="auto", spec_has_gradient=True, env={}, status=OK).source == "spec"
    assert B.resolve_backend("fit", settings_value="roadrunner", spec_has_gradient=True, env={}, status=OK).backend == "roadrunner"


def test_auto_means_no_preference():
    r = B.resolve_backend("profile", requested="auto", settings_value="auto", env={B.ENV_BACKEND: "auto"}, status=OK)
    assert (r.backend, r.source) == ("amici", "default")


def test_a_named_backend_that_cannot_run_is_an_error_with_advice():
    with pytest.raises(B.BackendUnavailable, match=r"pyantigen\[amici\]"):
        B.resolve_backend("fit", requested="amici", env={}, status=BAD)
    with pytest.raises(B.BackendUnavailable):
        B.resolve_backend("fit", spec_has_gradient=True, env={}, status=BAD)


def test_only_the_stage_default_falls_back():
    r = B.resolve_backend("profile", env={}, status=BAD)
    assert r.backend == "roadrunner" and r.notes and "using RoadRunner" in r.notes[0]


def test_bad_names_are_rejected():
    with pytest.raises(ValueError):
        B.resolve_backend("fit", requested="matlab", env={}, status=OK)
    with pytest.raises(ValueError):
        B.resolve_backend("optimise", env={}, status=OK)


def test_apply_to_kwargs():
    amici_named = B.Resolution("amici", "argument")
    out, notes = B.apply_to_optimizer_kwargs({"x": 1}, amici_named, "/c")
    assert out == {"x": 1, "analytic_gradient": {"cache_dir": "/c"}} and notes
    out, _ = B.apply_to_optimizer_kwargs({"analytic_gradient": {"cache_dir": "/mine", "rtol": 1e-8}}, amici_named, "/c")
    assert out["analytic_gradient"] == {"cache_dir": "/mine", "rtol": 1e-8}
    out, _ = B.apply_to_optimizer_kwargs({}, B.Resolution("amici", "default"), None)
    assert out == {}                                                  # a stage default never adds a gradient
    out, notes = B.apply_to_optimizer_kwargs({"analytic_gradient": True}, B.Resolution("roadrunner", "argument"))
    assert out == {} and notes
    src = {"analytic_gradient": True}
    B.apply_to_optimizer_kwargs(src, amici_named, "/c")
    assert src == {"analytic_gradient": True}                          # input untouched


@dataclass
class _Spec:
    optimizer_kwargs: dict = field(default_factory=dict)


def test_configure_spec_returns_a_new_spec_and_exports_the_choice(monkeypatch):
    monkeypatch.delenv(B.ENV_BACKEND, raising=False)
    monkeypatch.setattr(B, "amici_status", lambda *a, **k: OK)
    spec = _Spec({"a": 1})
    new = B.configure_spec(spec, "amici", None, printer=lambda *_: None)
    assert new is not spec and new.optimizer_kwargs["analytic_gradient"] and spec.optimizer_kwargs == {"a": 1}
    import os
    assert os.environ[B.ENV_BACKEND] == "amici"
    monkeypatch.delenv(B.ENV_BACKEND)
    assert B.configure_spec(spec, None, None, printer=lambda *_: None) is spec


def _status(platform="linux", have_amici=True, gcc="g++ (GCC) 13.2.0", tools=("cmake", "swig"), cxx_present=True):
    def which(name):
        if name in tools:
            return "/bin/" + name
        return "/bin/g++" if cxx_present and name == "g++" else None
    return B.amici_status(platform=platform, which=which,
                          find_spec=lambda n: object() if have_amici else None,
                          dist_version=lambda n: "1.1.0", run=lambda cmd: gcc, env={})


def test_machine_check():
    assert _status().available
    s = _status(platform="win32")
    assert not s.available and "WSL" in " ".join(s.advice)
    assert "not installed" in " ".join(_status(have_amici=False).problems)
    assert "gcc 8.5" in " ".join(_status(gcc="g++ (GCC) 8.5.0 20210514").problems)
    assert _status(gcc="Apple clang version 15.0.0").available                  # clang is not held to the gcc floor
    assert "no C++ compiler" in " ".join(_status(cxx_present=False).problems)
    assert any("swig" in p for p in _status(tools=("cmake",)).problems)


def test_amici_status_does_not_import_amici():
    import sys
    before = "amici" in sys.modules
    B.amici_status()
    assert ("amici" in sys.modules) == before


def test_cache_dirs():
    assert B.explicit_cache_dir("/a", {B.ENV_CACHE: "/b"}) == "/a"
    assert B.explicit_cache_dir(None, {B.ENV_CACHE: "/b"}) == "/b"
    assert B.explicit_cache_dir(None, {}) is None
    assert B.default_cache_dir("linux", {}, "/h") == "/h/.cache/pyantigen/amici_models"
    assert B.default_cache_dir("darwin", {}, "/h").startswith("/h/Library/Caches")
    assert B.default_cache_dir("linux", {"XDG_CACHE_HOME": "/x"}, "/h").startswith("/x/")
    assert B.cache_dir_warning("/h/.cache/p", {"SLURM_JOB_ID": "1"}, "/h")
    assert B.cache_dir_warning("/scratch/p", {"SLURM_JOB_ID": "1"}, "/h") is None


# --- sigma policy -----------------------------------------------------------------------------------------

def test_residual_stages_on_amici_get_fixed_sigma():
    for stage in ("multistart", "profile", "fim"):
        assert B.resolve_sigma_floor(stage, "amici") == "fixed"
    assert B.resolve_sigma_floor("fit", "amici") is None            # a plain fit keeps the engine default
    assert B.resolve_sigma_floor("profile", "roadrunner") is None   # until cap is retired


def test_spec_beats_settings_and_cap_is_refused_where_the_residual_form_is_used():
    assert B.resolve_sigma_floor("fit", "roadrunner", "fixed", "cap") == "fixed"
    assert B.resolve_sigma_floor("fit", "roadrunner", None, "fixed") == "fixed"
    assert B.resolve_sigma_floor("fit", "roadrunner", "auto", "auto") is None
    with pytest.raises(B.SigmaPolicyError, match="known sigmas"):
        B.resolve_sigma_floor("profile", "amici", "cap")
    with pytest.raises(B.SigmaPolicyError):
        B.resolve_sigma_floor("multistart", "amici", None, "cap")
    assert B.resolve_sigma_floor("profile", "roadrunner", "cap") == "cap"
    with pytest.raises(ValueError):
        B.resolve_sigma_floor("fit", "amici", "dynamic")


def test_configure_spec_sets_sigma_floor_from_settings(monkeypatch):
    monkeypatch.delenv(B.ENV_BACKEND, raising=False)
    monkeypatch.setattr(B, "amici_status", lambda *a, **k: OK)

    class S:
        SIGMA_FLOOR = "fixed"
    spec = _Spec({"a": 1})
    new = B.configure_spec(spec, None, S, printer=lambda *_: None)
    assert new.optimizer_kwargs == {"a": 1, "sigma_floor": "fixed"} and spec.optimizer_kwargs == {"a": 1}
    spec2 = _Spec({"sigma_floor": "cap"})
    assert B.configure_spec(spec2, None, S, printer=lambda *_: None) is spec2        # the spec's own value wins


def test_profile_backend_is_exported_and_a_default_amici_needs_fixed_sigma():
    from pyantigen.engine import Backend as B
    from pyantigen.engine.Backend import AmiciStatus

    ok = AmiciStatus(available=True, problems=(), advice=())
    quiet = lambda *_: None
    import unittest.mock as mock
    with mock.patch.object(B, "amici_status", return_value=ok):
        env = {}
        assert B.export_profile_backend(None, None, False, quiet, env=env) == "roadrunner"      # default AMICI, sigma not fixed
        assert env[B.ENV_PROFILE_BACKEND] == "roadrunner"
        env = {"PYANTIGEN_SIGMA_FLOOR": "fixed"}
        assert B.export_profile_backend(None, None, False, quiet, env=env) == "amici"
        env = {}
        assert B.export_profile_backend("amici", None, False, quiet, env=env) == "amici"        # named: not second-guessed
        assert B.export_profile_backend("roadrunner", None, False, quiet, env={}) == "roadrunner"


def test_configure_spec_handles_a_spec_whose_optimizer_kwargs_is_derived():
    """The project's Optimization builds optimizer_kwargs in __post_init__ (init=False): dataclasses.replace refuses it."""
    from dataclasses import dataclass, field
    from pyantigen.engine.Backend import configure_spec

    @dataclass
    class Spec:
        n: int
        optimizer_kwargs: dict = field(init=False, default_factory=dict)

        def __post_init__(self):
            self.optimizer_kwargs = {"options": {"maxiter": self.n}}

    import unittest.mock as mock
    from pyantigen.engine import Backend as B
    spec = Spec(5)
    with mock.patch.object(B, "amici_status", return_value=B.AmiciStatus(available=True)):
        out = configure_spec(spec, "amici", None, stage="fit", printer=lambda *_: None)
    assert out is not spec and out.n == 5
    assert out.optimizer_kwargs["analytic_gradient"] and out.optimizer_kwargs["options"] == {"maxiter": 5}
    assert "analytic_gradient" not in spec.optimizer_kwargs
