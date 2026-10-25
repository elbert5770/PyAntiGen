# Solver backends: RoadRunner and AMICI

| | RoadRunner | AMICI |
|---|---|---|
| install | core dependency (`pip install pyantigen`) | `pip install "pyantigen[amici]"`, then `pyantigen-setup` |
| platforms | Linux, macOS, Windows | Linux, macOS; **Windows only through WSL** |
| start-up | seconds | each model is compiled to C++ once (minutes to tens of minutes), then cached |
| run time | slower; no sensitivities | much faster per evaluation, with exact sensitivities |
| use for | simulate, quick fits | multistart, profile likelihood, FIM |

AMICI needs g++ >= 10.1 (or Apple clang), cmake and swig. `pyantigen-setup` checks all of these and
prints the fix for anything missing (it never uses sudo). `pyantigen-setup --install-amici` pip-installs
the package; it refuses on native Windows and on a cluster login node.

## Choosing

Per stage (`simulate`, `fit`, `multistart`, `profile`, `fim`), first match wins:

1. `Model_run.py --backend roadrunner|amici|auto`
2. environment variable `PYANTIGEN_BACKEND` (reaches spawned workers)
3. `BACKEND` in the project's `Modules/Solver_settings.py`
4. `amici`, if the optimization spec already sets `optimizer_kwargs["analytic_gradient"]`
5. the stage default: RoadRunner for `simulate` and `fit`, AMICI for `multistart`, `profile`, `fim`

`auto` at any level means no preference. A backend named at levels 1-4 that cannot run here stops the run
with the reason and the fix. Only the stage default falls back to RoadRunner, with a note.

`--backend amici` on `--optimize` switches on `analytic_gradient` for a spec that lacks it;
`--backend roadrunner` removes it. `--simulate` is always RoadRunner.

## Compiled-model cache

`Solver_settings.AMICI_CACHE_DIR`, else `PYANTIGEN_AMICI_CACHE`, else a per-user cache directory
(`~/.cache/pyantigen/amici_models`, `~/Library/Caches/...` on macOS). On a cluster, point it at group or
scratch storage: a home directory has a small quota and compiled models are large.

## Not covered yet

This layer chooses and checks the backend. The multistart, profile and FIM stages still run from the
project's `tools/` scripts; moving them into the engine is the next phase. BLAS is not checked.
