"""Which solver a stage runs on, RoadRunner or AMICI, and whether this machine can run AMICI.

RoadRunner is the default simulator: it needs no compiler and is quick to start. AMICI compiles each
model to C++ (slow to start, tens of minutes for a large model) and then evaluates a value and its
sensitivities much faster, so it is the choice for the long stages: multistart, profile, FIM. It needs
a C++ toolchain, so it is optional (``pip install pyantigen[amici]``) and it does not run on native
Windows; use WSL there.

The choice is made per stage, in this order (first one that is not ``"auto"`` wins):

1. ``requested``: the ``--backend`` flag of ``Model_run.py``
2. the environment variable ``PYANTIGEN_BACKEND`` (this is how it reaches spawned workers)
3. ``Solver_settings.BACKEND`` of the project
4. ``"amici"`` if the optimization spec already asks for an analytic gradient
5. the stage's default, ``STAGE_DEFAULT``

A backend that was asked for by name (1-4) and cannot run here is an error that says what is missing.
Only the stage default falls back, to RoadRunner, with a note.

Nothing here imports AMICI: ``amici_status`` looks for it, so importing this module costs nothing.
"""
import importlib.metadata
import importlib.util
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field

BACKENDS = ("auto", "roadrunner", "amici")
STAGES = ("simulate", "fit", "multistart", "profile", "fim")
STAGE_DEFAULT = {"simulate": "roadrunner", "fit": "roadrunner",
                 "multistart": "amici", "profile": "amici", "fim": "amici"}

ENV_BACKEND = "PYANTIGEN_BACKEND"
ENV_CACHE = "PYANTIGEN_AMICI_CACHE"

# AMICI builds its models with a C++17 compiler; the cluster's system gcc 8 cannot.
MIN_GCC = (10, 1)


SIGMA_MODES = ("auto", "cap", "fixed")
# Stages that use the residual form of the loss (Levenberg-Marquardt, J^T J as the FIM, one comparable f).
# That needs every sigma known, i.e. sigma_floor="fixed".
RESIDUAL_STAGES = ("multistart", "profile", "fim")


class SigmaPolicyError(ValueError):
    """The sigma rule asked for cannot be used with the backend and stage."""


class BackendUnavailable(RuntimeError):
    """A backend that was asked for by name cannot run on this machine."""


@dataclass(frozen=True)
class AmiciStatus:
    available: bool
    version: str | None = None
    problems: tuple = ()
    advice: tuple = ()


@dataclass(frozen=True)
class Resolution:
    backend: str            # "roadrunner" or "amici"
    source: str             # "argument" | "environment" | "settings" | "spec" | "default"
    notes: tuple = field(default_factory=tuple)


def _check(name, value):
    v = str(value).strip().lower()
    if v not in BACKENDS:
        raise ValueError(f"{name} must be one of {BACKENDS}, got {value!r}")
    return v


def _compiler_version(path, run):
    """(is_gcc, (major, minor)) of the C++ compiler at *path*, or (False, None) if it cannot be read."""
    try:
        out = run([path, "--version"])
    except Exception:
        return False, None
    first = out.splitlines()[0] if out else ""
    if "clang" in out.lower():
        return False, None
    m = re.search(r"(\d+)\.(\d+)\.\d+", first)
    return True, ((int(m.group(1)), int(m.group(2))) if m else None)


def _run_version(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout


def amici_status(platform=None, which=shutil.which, find_spec=importlib.util.find_spec,
                 dist_version=importlib.metadata.version, run=_run_version, env=None):
    """Whether AMICI can be imported and can compile models here. Arguments exist so a test can fake a machine."""
    platform = sys.platform if platform is None else platform
    env = os.environ if env is None else env
    problems, advice = [], []
    if platform == "win32":
        return AmiciStatus(False, None, ("AMICI does not run on native Windows",),
                           ("install WSL (https://learn.microsoft.com/windows/wsl/install), then install "
                            "pyantigen[amici] inside the WSL distribution",))
    version = None
    if find_spec("amici") is None:
        problems.append("the amici package is not installed")
        advice.append("pip install 'pyantigen[amici]'   (or: pyantigen-setup --install-amici)")
    else:
        try:
            version = dist_version("amici")
        except importlib.metadata.PackageNotFoundError:
            version = "unknown"
    cxx = env.get("CXX") or which("g++") or which("c++") or which("clang++")
    if not cxx:
        problems.append("no C++ compiler found")
        advice.append("macOS: xcode-select --install; Linux: install g++ >= 10 with your package manager; "
                      "Hyak: module load gcc/13.2.0 and export CC=gcc CXX=g++")
    else:
        is_gcc, ver = _compiler_version(cxx, run)
        if is_gcc and ver is not None and ver < MIN_GCC:
            problems.append(f"{cxx} is gcc {ver[0]}.{ver[1]}; AMICI needs >= {MIN_GCC[0]}.{MIN_GCC[1]}")
            advice.append("Hyak: module load gcc/13.2.0 and export CC=gcc CXX=g++")
    for tool in ("cmake", "swig"):
        if not which(tool):
            problems.append(f"{tool} not found")
            advice.append(f"pip install {tool}")
    return AmiciStatus(not problems, version, tuple(problems), tuple(advice))


def resolve_backend(stage, requested=None, settings_value=None, spec_has_gradient=False, env=None, status=None):
    """The backend *stage* runs on, as a :class:`Resolution`. Raises :class:`BackendUnavailable` if a named one cannot."""
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}, got {stage!r}")
    env = os.environ if env is None else env
    chosen, source = None, "default"
    levels = (("argument", requested), ("environment", env.get(ENV_BACKEND)), ("settings", settings_value))
    for src, val in levels:
        if val is None:
            continue
        v = _check(src, val)
        if v != "auto":
            chosen, source = v, src
            break
    if chosen is None and spec_has_gradient:
        chosen, source = "amici", "spec"
    if chosen is None:
        chosen = STAGE_DEFAULT[stage]
    notes = []
    if chosen == "amici":
        status = status if status is not None else amici_status()
        if not status.available:
            detail = "; ".join(status.problems) + (". " + " ".join(status.advice) if status.advice else "")
            if source == "default":
                notes.append(f"{stage}: AMICI is the default here but is not usable ({detail}); using RoadRunner")
                chosen = "roadrunner"
            else:
                raise BackendUnavailable(f"backend 'amici' ({source}) is not usable on this machine: {detail}")
    return Resolution(chosen, source, tuple(notes))


def resolve_sigma_floor(stage, backend, spec_value=None, settings_value=None):
    """The sigma rule to set on the spec (``"cap"`` or ``"fixed"``), or None to leave the engine's default.

    The spec's own ``optimizer_kwargs["sigma_floor"]`` is the source of truth; ``Solver_settings.SIGMA_FLOOR``
    fills it in when the spec is silent. Failing both, the AMICI residual stages (:data:`RESIDUAL_STAGES`)
    get ``"fixed"``, since the residual form needs known sigmas; every other stage keeps the engine's default,
    ``"cap"``, until that rule is retired. A residual stage on AMICI with ``"cap"`` is an error.
    """
    for name, v in (("optimizer_kwargs['sigma_floor']", spec_value), ("Solver_settings.SIGMA_FLOOR", settings_value)):
        if v is not None and str(v).strip().lower() not in SIGMA_MODES:
            raise ValueError(f"{name} must be one of {SIGMA_MODES}, got {v!r}")
    chosen = None
    for v in (spec_value, settings_value):
        if v is not None and str(v).strip().lower() != "auto":
            chosen = str(v).strip().lower()
            break
    residual = backend == "amici" and stage in RESIDUAL_STAGES
    if chosen is None and residual:
        chosen = "fixed"
    if chosen == "cap" and residual:
        raise SigmaPolicyError(
            f"the {stage} stage on AMICI minimizes the residual form of the loss, which needs known sigmas: "
            "use sigma_floor='fixed' (or run this stage on the RoadRunner backend)")
    return chosen


def apply_to_optimizer_kwargs(kwargs, resolution, cache_dir=None):
    """The spec's ``optimizer_kwargs`` with the resolved backend applied, and the notes it adds.

    ``amici`` asked for by name turns on ``analytic_gradient`` if the spec lacks it; ``roadrunner`` takes it
    off. A stage default never adds one. *cache_dir* fills ``analytic_gradient["cache_dir"]`` only where the
    spec left it unset. The input dict is not changed.
    """
    out = dict(kwargs or {})
    notes = []
    given = out.get("analytic_gradient")
    if resolution.backend == "roadrunner":
        if given:
            del out["analytic_gradient"]
            notes.append(f"analytic_gradient dropped: backend is roadrunner ({resolution.source})")
        return out, tuple(notes)
    if not given and resolution.source in ("argument", "environment", "settings"):
        given = True
        notes.append(f"analytic_gradient switched on: backend is amici ({resolution.source})")
    if given:
        settings = {} if given is True else dict(given)
        if cache_dir and not settings.get("cache_dir"):
            settings["cache_dir"] = cache_dir
        out["analytic_gradient"] = settings or True
    return out, tuple(notes)


def explicit_cache_dir(setting=None, env=None):
    """The compiled-model directory the user chose (Solver_settings, then the environment), or None."""
    env = os.environ if env is None else env
    return setting or env.get(ENV_CACHE) or None


def default_cache_dir(platform=None, env=None, home=None):
    """Where compiled models go when nothing was chosen: a per-user cache directory, never the project tree."""
    platform = sys.platform if platform is None else platform
    env = os.environ if env is None else env
    home = home or os.path.expanduser("~")
    if platform == "darwin":
        base = os.path.join(home, "Library", "Caches")
    elif platform == "win32":
        base = env.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    else:
        base = env.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")
    return os.path.join(base, "pyantigen", "amici_models")


def cache_dir_warning(path, env=None, home=None):
    """A sentence if *path* sits in a home directory on a cluster, where it is quota-limited; else None."""
    env = os.environ if env is None else env
    home = os.path.realpath(home or os.path.expanduser("~"))
    if not (env.get("SLURM_JOB_ID") or os.path.isdir("/gscratch")):
        return None
    if os.path.realpath(path).startswith(home + os.sep):
        return (f"the compiled-model cache {path} is under your home directory, which has a small quota; "
                f"set {ENV_CACHE} (or Solver_settings.AMICI_CACHE_DIR) to a group or scratch directory")
    return None


def configure_spec(spec, requested=None, settings_module=None, stage="fit", printer=print):
    """*spec* (an ``Optimization``) with the backend for *stage* applied to its ``optimizer_kwargs``.

    This is what ``Model_run.py`` calls. *settings_module* is the project's ``Modules.Solver_settings``; its
    ``BACKEND`` and ``AMICI_CACHE_DIR`` are read if present. A named backend is also exported in
    ``PYANTIGEN_BACKEND`` so spawned workers resolve the same way.
    """
    import dataclasses
    kwargs = getattr(spec, "optimizer_kwargs", None) or {}
    res = resolve_backend(stage, requested, getattr(settings_module, "BACKEND", None),
                          bool(kwargs.get("analytic_gradient")))
    cache = explicit_cache_dir(getattr(settings_module, "AMICI_CACHE_DIR", None))
    new, notes = apply_to_optimizer_kwargs(kwargs, res, cache)
    sigma = resolve_sigma_floor(stage, res.backend, kwargs.get("sigma_floor"),
                                getattr(settings_module, "SIGMA_FLOOR", None))
    if sigma is not None and new.get("sigma_floor") != sigma:
        new = {**new, "sigma_floor": sigma}
        notes = (*notes, f"sigma_floor set to {sigma!r} ({stage} on {res.backend})")
    for n in (*res.notes, *notes):
        printer(f"[backend] {n}")
    printer(f"[backend] {stage}: {res.backend} ({res.source})")
    if res.backend == "amici":
        ag = new.get("analytic_gradient")
        chosen = (ag.get("cache_dir") if isinstance(ag, dict) else None) or cache or default_cache_dir()
        warn = cache_dir_warning(chosen)
        if warn:
            printer(f"[backend] warning: {warn}")
    if requested and str(requested).lower() != "auto":
        os.environ[ENV_BACKEND] = str(requested).lower()
    return dataclasses.replace(spec, optimizer_kwargs=new) if new != kwargs else spec
