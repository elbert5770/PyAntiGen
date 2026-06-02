#  Unlike Update_parameters.py, this function is called at each
#  iteration of the optimization problem. The purpose is to modify the parameters
#  in the RoadRunner instance (r) based on the current values of the
#  parameters being optimized. 'parameters' is a dictionary of the parameters
#  being optimized. 'parameters' is a dictionary of the parameters
#  that are being optimized.
#
#  Use cases: 
#    1. Some parameters depend on other parameters that are being optimized.
#    2. Events 
#
#  Example 1:
#
#      def update_opt_parameters_example(r, experiment, parameters):
#          r.k_A2_to_B2 = parameters["k_A_to_B"]
#
#  Example 2:
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