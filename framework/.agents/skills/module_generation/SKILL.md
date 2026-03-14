---
name: Generate PyAntiGen Modules
description: Guidelines and rules for generating new PyAntiGen modules
---

# Generate PyAntiGen Modules

When asked to generate new reactions or modules for a new project in the PyAntiGen framework, follow these guidelines carefully. The generated modules will typically be placed in the `modules/` folder of the repository.

## 1. File and Class Structure
- The module must be implemented as a Python class that inherits from `PyAntiGenModule`.
- Import the base class at the top of the file:
  ```python
  from framework.module_base import PyAntiGenModule
  ```
- The class must implement a `build(self)` method.

## 2. Handling Species and Isotopes
- Modules should retrieve their configured `Species` and `No_Isotope_SpeciesList` from `self.config`:
  ```python
  Species = self.config.get('Species')
  No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
  ```
- Determine the correct list of isotopes to iterate over. If the species is in `No_Isotope_SpeciesList`, it should only use the empty string `''` for its isotope label. Otherwise, use `self.model.isotopes`.
  ```python
  Isotopes = [''] if Species in No_Isotope_SpeciesList else self.model.isotopes
  ```

## 3. Formatting Isotope Strings
- Inside the loop over isotopes, build an `Isotope_str` to inject into reactant/product names:
  ```python
  for Isotope in Isotopes:
      Isotope_str = f"_{Isotope}_" if Isotope else "_"
  ```
- If the `Isotope` is `''` (natural abundance/unlabeled), `Isotope_str` will simply be `"_"` (e.g., resulting in `AB40_BrainISF`).
- If the `Isotope` is labeled (e.g., `'13C6Leu'`), `Isotope_str` will be `"_13C6Leu_"` (e.g., resulting in `AB40_13C6Leu_BrainISF`).

## 4. Compartment Naming Convention
- **CRITICAL**: Compartment names **must be CamelCase (or PascalCase) and must not contain any underscores**.
- ✅ **Valid Examples**: `BrainISF`, `TissueVascular`, `SAS`, `BrainVascular`, `Endosomal`.
- ❌ **Invalid Examples**: `Brain_ISF`, `Tissue_Vascular`.

## 5. Adding Reactions
- Use the `self.add_reaction()` method to append reactions to the model.
- **Single Species Format**: Reactants and Products must use the format `[{Species}{Isotope_str}{Compartment}]`.
- **CRITICAL - Multiple Species Array Syntax**: When a reaction has multiple reactants or products (e.g., binding of A and B), you **MUST** format the species as a comma-separated list enclosed in a single pair of brackets, like `f"[{Species1}, {Species2}]"`. Do **not** use string addition like `f"{Species1} + {Species2}"`. 
- **CRITICAL - Variable Definitions**: Do not wrap your Python variables defining species names in brackets natively (e.g., use `APP = f"APP_{Comp}"` instead of `APP = f"[APP_{Comp}]"`). Supply the brackets only when passing to `add_reaction`.
- Ensure that the generated Reaction Names are descriptive and ideally incorporate the rule/flow type, the species, the isotope string, and the compartment(s).

### Reaction Addition Example
```python
Reaction_name = f"FlowWithinTissue_{Species}{Isotope_str}{Comp1}_{Comp2}"
Reactants = f"[{Species}{Isotope_str}{Comp1}]"
Products = f"[{Species}{Isotope_str}{Comp2}]"
Rate_type = "UDF"  # Unidirectional Flow
# CRITICAL: Use parameter names (symbols), NOT hardcoded numerical values.
Rate_eqtn_prototype = "(1-RC_BBB) * Q_BrainISF" 

self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)
```

## 6. Parameter Names vs. Hardcoded Values
- **CRITICAL**: Always use **parameter names (symbols)** in your `Rate_eqtn_prototype` instead of hardcoding numerical values. 
- For instance, if a clearance rate is 4.81e-5, define the equation using the parameter symbol (e.g., `\"kclearAPP\"`) rather than the number `\"4.81e-5\"`.
- This allows external scripts to parse the parameter names and values, enabling easy parameter sweeps and increasing flexibility without changing the Antimony generation code.


## 7. Rate Types and Equation Prototypes
When adding reactions, the `Rate_type` must be carefully specified along with a properly formatted `Rate_eqtn_prototype` that corresponds to the correct units.
- **UDF (Unidirectional Flow)**: `Rate_eqtn_prototype` must have units of **[volume/time]**. Example: `\"(1-RC_BBB) * Q_BrainISF\"`.
- **BDF (Bidirectional Flow)**: Requires two rate prototypes (usually provided as an array or specific format parsed by your module) with units of **[volume/time]**.
- **MA (Mass Action)**: `Rate_eqtn_prototype` has units of **[concentration]^n/time**.
- **RMA (Reversible Mass Action)**: Requires two rate prototypes.
- **custom**: Requires providing the entire rate term with units of **[amount/time]**, where species names in the equation are treated as concentrations.

## 8. Complete Module Example

```python
from framework.module_base import PyAntiGenModule

class Example_FlowsModule(PyAntiGenModule):
    \"\"\"
    Example module for compartmental flow reactions.
    \"\"\"
    def build(self):
        Species = self.config.get('Species')
        No_Isotope_SpeciesList = self.config.get('No_Isotope_SpeciesList', [])
        
        Isotopes = [''] if Species in No_Isotope_SpeciesList else self.model.isotopes

        flow_rates = {
            ('TissueVascular', 'BrainISF'): "(1-RC_BBB) * Q_BrainISF",
        }
        
        for Isotope in Isotopes:
            Isotope_str = f"_{Isotope}_" if Isotope else "_"
            for Comp1, Comp2 in flow_rates.keys():
                Reaction_name = f"ExampleFlow_{Species}{Isotope_str}{Comp1}_{Comp2}"
                Reactants = f"[{Species}{Isotope_str}{Comp1}]"
                Products = f"[{Species}{Isotope_str}{Comp2}]"
                Rate_type = "UDF"
                Rate_eqtn_prototype = flow_rates[(Comp1, Comp2)]
                
                self.add_reaction(Reaction_name, Reactants, Products, Rate_type, Rate_eqtn_prototype)
```

## 9. Reusing and Naming Modules
- **Preserve Existing Modules**: When creating a new module based on logic from an existing one, do not modify or overwrite the existing module. Leave existing code completely intact.
- **Naming Convention**: Create a new file and a new class. The new module's file name and class name should incorporate the new project's name (e.g., `CNS_Flows_<NewProjectName>Module` in `cns_flows_<newprojectname>.py`) to clearly distinguish it from prior versions and prevent accidental regression.
