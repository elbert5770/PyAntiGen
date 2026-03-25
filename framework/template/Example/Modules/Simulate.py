import time

def simulate(r):
    r.setIntegrator('cvode')
    r.integrator.absolute_tolerance = 1e-6
    r.integrator.relative_tolerance = 1e-8
    r.integrator.setValue('stiff', True)
    r.integrator.variable_step_size = True
    print(r.integrator)

    observed_species = ['time','[A_Comp1]','[B_Comp1]', 'pw_interp1', 'pw_interp2']


    print("Running simulation...")
    t0 = time.perf_counter()

    result0 = r.simulate(-10000, 0, 100, observed_species)
    result1 = r.simulate(0, 48, 1000, observed_species)
    
    elapsed = time.perf_counter() - t0
    print(f"Simulation time: {elapsed:.3f} s")
    print(f"CVODE took {len(result1)} steps.")

    return result1
