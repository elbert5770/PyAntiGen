#  Unlike Update_parameters.py, this function is called at each
#  iteration of the optimization problem. The purpose is to modify some 
#  non-optimized parameters that depend upon an optimized parameter.
#  The parameters are modified in the RoadRunner instance (r).
#
#  Use cases: 
#    1. Some parameters depend on parameters that are being optimized.
#    2. Event parameters change depending on experimental design.
#    
#
#  Example:
#
#      def update_opt_parameters_antibody(r, experiment, parameters):
#          drug_name = experiment.get("Drug")
#          if drug_name == "Lecanemab":
#              if 'k_f_DensePlaque_Antibody_Lecanemab' in parameters:
#                  r['k_f_DensePlaque_Antibody'] = parameters['k_f_DensePlaque_Antibody_Lecanemab']
#          elif drug_name == "Aducanumab":
#              if 'k_f_DensePlaque_Antibody_Aducanumab' in parameters:
#                  r['k_f_DensePlaque_Antibody'] = parameters['k_f_DensePlaque_Antibody_Aducanumab']
#          

def update_opt_no_parameters(r, experiment, parameters):
    return