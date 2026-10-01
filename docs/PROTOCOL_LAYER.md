# The protocol layer

A Study is a paper. That is a *provenance* unit, and it is the right
top-level one: it has a DOI, a reader can check it, and it owns its control
arm. But it is not a *reuse* unit. Two papers can run nearly the same
experiment -- the same preparation, the same tracer, the same sampling -- and
differ only in which drug went in, with nothing in either file saying so.
What is shared is neither the paper nor the figure; it is the **protocol**.

This document describes the layer being added under Study, one stage at a
time, and what each stage is allowed to disturb. The design argument, the
survey of lab-protocol languages it borrows from, and the errors that
motivated it live in the project note it came from
(`NEXT_AND_PROTOCOL_LAYER.md`, Part 2).

## The shape being built toward

```
Study (one paper, one doi)
  Protocol = Target x Intervention        reusable, citable, completeness-checkable
    Simulation = Protocol x Subject x levels x period
  Assay = analyte x matrix x method  ->  on={simulations}
Optimization = Params x use(study, assays)
```

The three jobs today's `Protocol` does at once:

* **the target** -- what system this is and what compartments it has;
* **the intervention** -- what was done to it, and when;
* **the numerics** -- solver settings and recorded species.

`Assay` is already separate, and that is the part that shows the payoff.

## The rule every stage obeys

**Recording what a paper did must not change what the model computes.**

Documentation that can alter a fit is not documentation. Concretely, for
each stage:

* Nothing in the layer is read by `resolve` or reaches a replicate dict.
* Nothing in it becomes a simulation attribute, so it cannot be selected on
  and cannot silently distinguish two otherwise identical simulations.
* A design that does not use a stage's feature serializes exactly as it did
  before, byte for byte, and therefore keeps its `fingerprint` -- so every
  `Optimization` that pinned that study stays valid. This is why
  `serialize._format_for` writes the *minimum* format a design needs rather
  than the library's current one.

Stage 0 of this work was done outside the library for exactly this reason:
the obvious home for protocol prose, `Protocol.settings`, is merged into
every simulation's attributes by `Study.attributes`, so prose there would
reach the Engine, `select()`, `validate`'s duplicate-simulation check and
`describe`. It is not inert. A sidecar folder of notes is.

## Stages

| stage | what | status |
| --- | --- | --- |
| 0 | Record target / intervention / measurement as prose next to each study, and see whether the vocabulary survives two papers | done, outside the library |
| 1 | `Q` (UCUM quantities) and `Reagent` as value objects; move drug identities there | **this release** |
| 2 | `Target` with typed compartments; the species overlay reads from it | not started |
| 3 | `Intervention` as an ordered action list, with a per-model compiler to events | not started |
| 4 | `Protocol = Target x Intervention`; `describe` prints and diffs methods | not started |
| 5 | Exporters (LabOP, Autoprotocol) and a completeness validator | not started |

## Stage 1: Q and Reagent

### Q

A number with a UCUM unit (`pyantigen.study.quantity`). It exists so a
protocol quantity can be written as the paper said it, converted once where
it enters the design, and checked for dimensional sense before it gets
there.

```python
>>> rate = 16 * Q(1.5, "mL") / Q(144, "h")     # 16 draws of 1.5 mL over 144 h
>>> rate.in_("L/h")
0.00016666666666666666
```

That is a *derivation*, and a derivation that runs is one that cannot be
aliased to another paper's number -- which is how one study's CSF drain rate
came to be used for another's, then rounded to 0.001 L/h, and cost a refit
when it was corrected.

The Engine stays unitless. `in_` is the one function that takes a Q out of
this layer, and it returns a plain float.

Scope: SI base units, the prefixes, and the atoms these protocols use
(litre, minute, hour, day, week, year, percent). Not derived electrical or
mechanical units, not degrees Celsius (an offset, not a factor), and not
anything needing chemistry -- `Q(5, "nmol/L").in_("ug/mL")` is refused,
because that conversion needs a molar mass, which belongs to a named
substance rather than to a unit string. The refusal names the reason.

An exact atom beats a prefix reading, as in UCUM, so `h` is an hour, `d` a
day, `min` a minute and `a` a year.

Equality is semantic and exact: `Q(1, "L") == Q(1000, "mL")`. For computed
quantities use `isclose`.

### Reagent

A named substance, declared once on the Study and referenced wherever used
(`pyantigen.study.reagent`). A reagent appears in two roles -- part of the
target (medium, serum, diet, vehicle) and part of the intervention (drug,
tracer) -- so it is one declaration, not two strings that can disagree.

```python
s.reagent("saline", id="CHEBI:75958")
s.reagent("leucine_13C6", id="CHEBI:15603", label="[U-13C6] L-leucine",
          supplier="Cambridge Isotope Laboratories CLM-2262",
          concentration=Q(7.5, "mg/mL"), vehicle="saline")
s.reagent("MK-0752", id=None, vehicle="water", potencies=[
    Potency("IC50", Q(5, "nmol/L"), "SH-SY5Y CVCL_0019", source="Cook 2010")])
```

A vehicle is itself a reagent, recursively, because a vehicle is a mixture.

A **potency attaches to a (substance, system) pair**, not to the model. A
drug's IC50 in a cell line and its fitted constant in an animal are different
quantities about the same molecule in different systems; today `system` is a
free string, and when `Target` lands it becomes a reference to one. A
recorded potency is never a model parameter: fitted constants stay `Param`s
on an `Optimization`, and a fitted value that disagrees with a published one
is a finding, not an inconsistency for the library to reconcile.

`validate` warns when a reagent has no identifier, the same way it warns
about a missing `doi`, and errors when a vehicle names a reagent that was
never declared. Identifiers stay optional: requiring them would make the
layer unusable for unpublished datasets, which are the ones that most need
the discipline.

### What stage 1 does not do

It does not connect reagents to events, doses or parameters. Nothing
consumes a `Reagent` yet; `describe` prints them and the JSON records them.
That is the whole of it, deliberately -- the identities have somewhere
correct to live before anything depends on them.
