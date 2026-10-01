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
