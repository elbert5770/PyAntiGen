# Update_parameters.py is used in these cases:
# 1. The base parameters change from one treatment to another
# 2. The base parameters change from one replicate to another
# 3. The base parameters are modified by events during the treatment
#  Update_parameters is called after the parameters are set during
#  construction of the RoadRunner model, but before the simulation is run.
#  It is not called during each round of optimization, only at the
#  setup of the optimization problem. If parameters need to be modified
#  at each parameter update in an optimization problem, then
#  contruct a function within Update_opt_parameters.py.
#  The function does not need to return anything, but modifies the parameters in the
#  RoadRunner instance (r).
#  'mode' is optional and is given "Simulator" for a pure simulation
#  and "Optimizer" for an optimization problem. The purpose is to allow for
#  different parameter updates for simulations and optimizations.
#  'replicate' is a dictionary containing the replicate information.
#  'r' is the RoadRunner instance.
#
#  Example:
#
#      def update_parameters(r, replicate):
#          dose = replicate.get("dose", 10)
#          r.A_Comp1 = dose
#          



def update_no_parameters(r, replicate):
    return

def update_flipflop(r, replicate):
    # Ground-truth values used to generate data/Flipflop.csv
    # (see data/make_flipflop_data.py). During optimization these are
    # overwritten every evaluation by the fitted parameters; for --simulate
    # they make the model reproduce the data-generating truth.
    r['k_A_to_B'] = 0.35
    r['k_B_to_C'] = 0.07
    r['SF'] = 1.6
    return

def update_Example(r, replicate):
    
    if replicate["amyloid_positive"]:
        r['k_A_to_B'] = 0.2
        print("Replicate: ", replicate["Label"], " k_A_to_B: ", r['k_A_to_B'])
    else:
        r['k_A_to_B'] = 0.1
        print("Replicate: ", replicate["Label"], " k_A_to_B: ", r['k_A_to_B'])
    return