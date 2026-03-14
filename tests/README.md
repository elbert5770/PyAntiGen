# PyAntiGen tests

## SILK integration tests

The tests in `test_silk_all_reactions.py` run scripts and assert on the generated `*_all_reactions.txt` files:

| Script | Generated file | Min reactions |
|--------|----------------|---------------|
| `scripts/Elbert_2022_1a.py` | `generated/Elbert_2022_1a_all_reactions.txt` | 100 |
| `scripts/Bloomingdale_2021_1a.py` | `generated/Bloomingdale_2021_1a_all_reactions.txt` | 25 |
| `scripts/Lin_2022_1b.py` | `generated/Lin_2022_1b_all_reactions.txt` | 70 |

### Where the scripts come from

- **Preferred**: A local copy under **`tests/silk_fixtures/`** (scripts and modules from Elbert_2022_SILK). If present, tests use it and do not need the external repo.
- **Fallback**: If `silk_fixtures` is missing, the Elbert_2022_SILK project must be available:
  - Either as a sibling directory of PyAntiGen: `.../Elbert_2022_SILK`
  - Or set `SILK_PROJECT_PATH` to its root
- PyAntiGen’s `framework` must be on `PYTHONPATH` when the scripts run (tests set this automatically).

### Running

From the PyAntiGen root:

```bash
python -m pytest tests/test_silk_all_reactions.py -v
```

If SILK is not found, the tests are skipped.

### What is checked

- Scripts run successfully and write `generated/<name>_all_reactions.txt`.
- Each line parses as a reaction dict with keys: `Reaction_name`, `Reactants`, `Products`, `Rate_type`, `Rate_eqtn_prototype`.
- `Rate_type` is one of: MA, RMA, UDF, BDF, custom, custom_conc_per_time, custom_amt_per_time.
- Reaction count is at least the expected minimum.
- A set of expected reaction names appears in the file.
