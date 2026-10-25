# Which solver runs the optimization: "auto", "roadrunner" or "amici". RoadRunner is quick to start and needs
# nothing extra; AMICI compiles each model (slow once) and then runs multistart and profile much faster, but
# needs `pip install pyantigen[amici]` and does not run on native Windows (use WSL). `pyantigen-setup` checks
# this machine. Overridden by `Model_run.py --backend` and by the PYANTIGEN_BACKEND environment variable.
BACKEND = "auto"

# Where AMICI keeps compiled models. None: PYANTIGEN_AMICI_CACHE, else a per-user cache directory. On a
# cluster point this at group or scratch storage, not at the home directory.
AMICI_CACHE_DIR = None

def make_solver_settings(blocks, abs_tol=1e-10, rel_tol=1e-10, max_steps=20000):
    """Factory for solver settings dicts. Reduces boilerplate across all figure settings."""
    return {
        'integrator': 'cvode',
        'absolute_tolerance': abs_tol,
        'relative_tolerance': rel_tol,
        'maximum_num_steps': max_steps,
        'stiff': True,
        'variable_step_size': True,
        'simulation_blocks': blocks,
    }

def solver_settings_Example(replicate):
    return make_solver_settings(
        {'block1': {'start': 0, 'end': 48, 'n_points': 1000, 'abs_tol':1e-12, 'rel_tol':1e-12}}
    )
