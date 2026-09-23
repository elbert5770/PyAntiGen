---
name: Convert ODEs to Antimony
description: Guidelines for converting plain text ODEs into Antimony reactions using the PyAntiGen module framework.
---

# Convert ODEs to Antimony
When asked to convert a system of Ordinary Differential Equations (ODEs) into Antimony format and subsequently generate a PyAntiGen module, you must follow a structured pipeline to prevent errors related to compartmental modeling, dimensional inconsistency, and raw text typos.

## Step 1: Intermediate Reaction Extraction
Before writing any Antimony code or Python code, extract the reactions mathematically implied by the ODEs and write them sequentially as an intermediate step.
- **MathML AST Evaluation**: Many DOCX files use Word's built-in MathML (`<m:oMath>`). *Do not* naively flatten `<m:t>` text nodes! Parameters like `k_M2G` are represented as an `<m:sSub>` structure containing a base `<m:e>` and a `<m:sub>`. You must construct a recursive AST parser that evaluates tags like `sSub`, `sSup`, `f` (fractions), and `r` (text runs) separately to retain the mathematical logic! (e.g. Concatenating `sSub` base and sub fields creates unified subscript variables like `kM2G`).
- **Implied Multiplication**: Identify implied mathematical multiplication by checking if variable/parameter boundaries (e.g. `kclearAPP` followed by `APPplasma`) are explicitly adjacent in the evaluated MathML syntax blocks without an explicit operator symbol. Insert a strict `*` operator between adjacent alphanumeric strings during AST evaluation.
- Ensure you output a table format similar to the `Lin2022.docx` format.
- Structure: `Reactants -> Products | Rate expression`
- *Example*: 
  ```text
  APPplasma + BACEplasma -> APP_BACEplasma | konPP * APPplasma * BACEplasma
  ```

## Step 2: Biological and Dimensional Error Checking
Perform a thorough inspection of the generated reaction table and the original ODEs. Compile a list of potential errors before proceeding:
1. **Amount vs. Concentration (Dimensional Consistency)**: Check species prefixes. Species prefixed with `m` are often amounts, while others are concentrations. A binding term between an amount and a concentration directly alters both without scaling, which usually indicates a missing volume division in the concentration's ODE (e.g. `Amount * Concentration` does not yield a pure Rate of Concentration Change).
2. **Volume Differences in Transport**: Transport between compartments (e.g., Plasma and CSF) for species measured in *concentrations* must include a volume ratio (e.g. `(V_csf / V_plasma) * k * [Species]`). If the raw ODEs omit these volume mappings but the variables are concentrations, report this error. (If the species are purely amounts, the symmetry holds without volume factors).
3. **Typos in Raw Text**: Missing multiplication operators (e.g., `kcleaveBACEplasma` instead of `kcleave * BACEplasma`), cross-compartment typos (e.g., `Aoligcsf` used in a `bisf` equation), or injected pseudocode (e.g., `kinfusion if t<=2 h`).
4. **Stoichiometry**: Validate that oligomerization or cleavage processes correctly balance mass.

Provide a `potential_errors.txt` (or similar summary) outlining these inconsistencies.

## Step 3: Generate the PyAntiGen Module
Once errors are identified or corrected (if instructed), proceed to write the PyAntiGen Python module adhering to the "Generate PyAntiGen Modules" skill:
- Implement the `build(self)` method and use `self.add_reaction`.
- Adjust `Rate_eqtn_prototype` to include any requested compartment volume divisions.
- Wrap species names in brackets for array representations in PyAntiGen (e.g. `f"[{Species1}, {Species2}]"`).
- Ensure parameter symbols (not hardcoded values) are used for rate constants and volume factors.
