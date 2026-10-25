"""``pyantigen-setup``: say whether this machine can run AMICI, and install it where that is safe.

    pyantigen-setup                   report only
    pyantigen-setup --install-amici   pip-install the amici package into the running Python, then report

It never uses sudo or a system package manager; what it cannot do it prints. Native Windows is refused
(use WSL). On a shared cluster's login node an install is refused: do it inside a Slurm job.
"""
import argparse
import json
import os
import socket
import subprocess
import sys

from pyantigen.engine.Backend import (ENV_BACKEND, ENV_CACHE, amici_status, cache_dir_warning, default_cache_dir,
                                      explicit_cache_dir)

AMICI_REQUIREMENT = "amici>=1.1.0"


def on_login_node(env=None):
    env = os.environ if env is None else env
    return (not env.get("SLURM_JOB_ID")) and "login" in socket.gethostname().lower()


def install_amici(run=subprocess.run, env=None):
    """pip-install AMICI. Returns (ok, message)."""
    if sys.platform == "win32":
        return False, "native Windows is not supported; install WSL and run this inside it"
    if on_login_node(env):
        return False, ("this looks like a cluster login node; request a compute node first "
                       "(salloc ...) and run the install there")
    r = run([sys.executable, "-m", "pip", "install", AMICI_REQUIREMENT])
    return r.returncode == 0, ("installed " + AMICI_REQUIREMENT if r.returncode == 0 else "pip install failed")


def report(status, cache):
    lines = [f"Python      {sys.version.split()[0]} ({sys.platform})",
             f"AMICI       {'usable, version ' + str(status.version) if status.available else 'NOT usable'}"]
    lines += [f"  problem   {p}" for p in status.problems]
    lines += [f"  fix       {a}" for a in status.advice]
    lines.append(f"AMICI cache {cache}")
    warn = cache_dir_warning(cache)
    if warn:
        lines.append(f"  warning   {warn}")
    lines.append(f"RoadRunner  always available (core dependency); AMICI is chosen per stage, see {ENV_BACKEND}")
    return "\n".join(lines)


def main(argv=None):
    p = argparse.ArgumentParser(prog="pyantigen-setup", description=__doc__.split("\n\n")[0])
    p.add_argument("--install-amici", action="store_true", help="pip install the amici package, then re-check")
    p.add_argument("--json", action="store_true", help="print the status as JSON")
    args = p.parse_args(argv)
    if args.install_amici:
        ok, msg = install_amici()
        print(msg)
        if not ok:
            return 1
    status = amici_status()
    cache = explicit_cache_dir() or default_cache_dir()
    if args.json:
        print(json.dumps({"available": status.available, "version": status.version, "problems": status.problems,
                          "advice": status.advice, "cache_dir": cache}, indent=2))
    else:
        print(report(status, cache))
    return 0 if status.available else 2


if __name__ == "__main__":
    sys.exit(main())
