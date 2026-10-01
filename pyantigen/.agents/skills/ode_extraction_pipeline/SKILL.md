---
description: Pipeline for extracting ODEs from Office MathML in Word documents and formatting them into structured Reaction Tables
---

# ODE Extraction Pipeline from Word MathML

This skill documents the automated pipeline designed to extract algebraic ordinary differential equations encoded as Office MathML (`<m:oMath>`) inside Microsoft Word (`.docx`) files, transform them into machine-readable text, and translate those ODE expressions into structured Reaction Tables suitable for validation against Antimony models.

## Pipeline Steps

### 1. Extract Equations from Word OMML
**Execution:** `python scripts/extract_omml_proper.py <input.docx>`
- Parses the Word archive to extract `word/document.xml`.
- Recursively evaluates MathML nodes (`m:sSub`, `m:f`, `m:sSup`, etc.).
- Binds subscripts natively to their bases (e.g., `<m:e>k</m:e>` and `<m:sub>13</m:sub>` -> `k13`).
- Intelligently injects standard `*` characters between implicitly multiplied, adjacent alphanumeric parameter strings.
- **Output:** `[filename]_extracted_omml.txt`

### 2. Tabulate as Reactions
**Execution:** `python scripts/build_reaction_table_starred.py <input_extracted.txt> [output_table_txt]`
- Identifies mathematical derivatives, isolating the target accumulated variable (e.g., `d(Species)/dt`).
- Splits right-hand side algebraic expressions using sign boundaries while logically handling nested parentheses.
- Groups separated terms having the exact same mathematical rate back together, grouping separated productions and consumptions into a single physical reaction.
- **Format:** `Reactants -> Products | Rate equation`
- **Output:** `[filename]_reactions_table.txt`

### 3. Normalize Nomenclature
**Execution:** `python scripts/map_species_names.py <input_table.txt> [output_mapped.txt] [dictionary_source.py]`
- Bypasses manual string-matching by importing a trusted dictionary (e.g., `name_map` from `scripts/compare_all_species.py`).
- Operates on word boundaries (`\b(key)\b`) inside the reaction table text to cleanly replace source nomenclatures into precise Antimony model equivalents.
- **Output:** `[filename]_reactions_table_mapped.txt`

### 4. Verify and Match
**Execution:** `python scripts/match_reactions.py` *(requires paths inside script or via CLI to be matched to target system)*
- Ingests the normalized `reactions_table_mapped.txt`.
- Parses a compiled target hand-written Python Antimony model file.
- Calculates Jaccard similarity and exact intersections to match documented equations perfectly to implemented lines of code.
- Generates a human-auditable markdown report displaying isolated errors or undocumented mechanics.

## Instructions for AI Agent
When an end-user provides a new Word Document describing mathematical equations to be translated into Antimony or to be verified against an existing codebase, run exactly steps 1 and 2 to obtain the base tabular equations. Only run step 3 if a valid dictionary exists to map the species properly. Finally, evaluate fidelity using step 4.
