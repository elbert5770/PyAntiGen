# Update_parameters.py is used in these cases:
# 1. The base parameters change from one treatment to another
# 2. The base parameters change from one replicate to another
# 3. The base parameters are modified by events during the treatment
#  Update_parameters is called after the parameters are set, but before the simulation is run.
#  The function does not need to return anything, but modifies the parameters in the
#  RoadRunner instance (r).
#  Keyword arguments are passed to the function as meta data.
#
#  Example:
#
#      def update_parameters(r, replicate):
#          dose = replicate.get("dose", 10)
#          r.A_Comp1 = dose
#          



def update_no_parameters(r, replicate, mode):
    return {}

def update_Example(r, replicate, mode):
    if mode == "Simulator":
        if replicate["amyloid_positive"]:
            r['k_A_to_B'] = 0.2
    return {}