# Canadian Survey on Business Conditions: Analysis & Quarterly Refresh Pipeline

**Aashlesha Khanna** · Data Scientist Application, Business Data Lab, Canadian Chamber of Commerce

Four quarterly extracts of the Canadian Survey on Business Conditions (Q3 2023 – Q2 2024), analysed
end to end, plus a pipeline that absorbs future quarters without manual rework.

---

## The finding, in short

Business expectations improved into Q2 2024: but the improvement is narrower than the national
headline suggests.

1. **Sales, employment and profitability** increase-expectations each hit four-quarter highs
   (**+2.9 / +2.1 / +2.1 pp**). **Investment stayed flat** at 17.6%.
2. **Investment is not a lagging indicator — it is a separate decision.** Two independent
   diagnostics: one business in six gives **no directional answer** on investment (against ~0% on
   employment), and investment is **statistically uncorrelated** with the other three metrics
   (mean r = 0.10 vs 0.32–0.36).
3. **Margin pessimism is entrenched, not emerging.** **12 of 13 industries** expect profitability to
   fall in Q2 2024, and **every** industry expected it in **each of the three prior quarters**. The
   mean balance is improving (−23.9 → −17.1 pp), so Q2 is the least bad of the four; but businesses
   have expected to sell more and earn less for a year, which is a cost-side story.
4. **The improvement was regionally concentrated.** Ontario (**+1.6 pp**) and Quebec (**+2.2 pp**),
   the two largest economies, came in *below* the national +2.9 pp. **Manitoba declined.**

The full details is in the bdl_assignment_report.pdf
---

## How to read this

Seven notebooks in [`src/`](src/), meant to be read in order. **Every notebook ships with its outputs
saved**, so all tables and figures are visible without running anything.

| Notebook | What it covers |
|---|---|
| [01_data_preprocessing](src/01_data_preprocessing.ipynb) | Load, label drift, missing data, abnormalities, validation → clean dataset |
| [02_understanding_data](src/02_understanding_data.ipynb) | Composition, distributions, trends, correlation, and what we do about it |
| [03_analysis_national_expectations](src/03_analysis_national_expectations.ipynb) | **Analysis 1**: the national trend |
| [04_analysis_sector_profitability](src/04_analysis_sector_profitability.ipynb) | **Analysis 2**: profitability balance by industry |
| [05_analysis_regional_dispersion](src/05_analysis_regional_dispersion.ipynb) | **Analysis 3**: who actually experienced the improvement |
| [06_observations_and_conclusions](src/06_observations_and_conclusions.ipynb) | The story, every claim tied to its number, and the limits |
| [07_quarterly_refresh_pipeline](src/07_quarterly_refresh_pipeline.ipynb) | **Part 2**: the pipeline, both modes, guards exercised live |

All shared logic lives in one module, [`src/csbc.py`](src/csbc.py): schema rules, validation, label
normalisation, chart styling, and the pipeline so the notebooks, the tests, and the CLI enforce
exactly the same behaviour.

---

## Quick start

Nothing needs to be run to review the submission. To re-run it:

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m jupyter lab
```

**macOS / Linux**

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pytest -q
.venv/bin/python -m jupyter lab
```

Then run notebooks `01` → `07` in order with **Kernel → Restart Kernel and Run All Cells**.
`python -m pytest -q` should report **15 passed**.

Prefer not to use a virtual environment? `pip install -r requirements.txt` into your existing Python
works identically; the dependencies are just pandas, matplotlib, JupyterLab and pytest.

### Runs anywhere

Every path is derived from the location of `src/csbc.py`, never from the working directory. Unzip the
folder wherever you like; the notebooks work whether Jupyter starts in the project root or in `src/`,
and the CLI works from any directory. There are no absolute paths in the project.

---

## Ingesting a new quarter

The pipeline answers **two different questions**, because a new file is useful in two ways:

| Mode | Question | Needs the history? |
|---|---|---|
| `standalone` | What does this new quarter say **on its own**? | No |
| `append` | How does the **whole series** look now? | Yes |
| `both` | Both, in that order *(default)* | Yes |

```bash
# both (default)
python src/refresh_quarter.py --new-file "data/raw/Data CSBC-Q3 2024.csv"

# inspect the quarter alone, writing nothing to the history
python src/refresh_quarter.py --new-file "data/raw/Data CSBC-Q3 2024.csv" --mode standalone

# append and refresh the trend
python src/refresh_quarter.py --new-file "data/raw/Data CSBC-Q3 2024.csv" --mode append
```

`standalone` runs first by design, so a new file can be inspected **before** any decision to admit it
to the historical series. The script writes nothing and exits non-zero when a guard fires.

**Guards**: each is triggered and shown blocked in notebook `07`, not merely described:

| Attempt | Result |
|---|---|
| Re-ingesting a quarter already in the history | blocked |
| An unrecognised `Business_information` label | blocked |
| A renamed `VALUE` column | blocked |
| A percentage outside 0–100 | blocked |
| An incoming file containing two quarters | blocked |

The reasoning: a renamed column or unmapped category could **silently split a historical series**:
the worst outcome, because the chart still renders and still looks plausible. So the pipeline surfaces
the change rather than guessing. Missing values, by contrast, are expected and are reported and
retained, not treated as a fault.

---

## Key data decisions

| Issue found | Decision | Why |
|---|---|---|
| Two renamed labels in Q2 2024 | Map explicitly in code | An unknown label should fail loudly, not start a parallel series |
| 7 missing `VALUE` cells | Flag, retain, never impute | All 7 sit in the first 18 rows of one file and 6 of 7 are `decrease` an export artifact; and no safe residual exists to back-fill from |
| 122 all-zero groups | Count separately from partial ones | Absent cells, not incomplete answers |
| Shares summing below 100 | Report, never rescale | The residual is real and **metric-specific** (Investment ~15 pp, Employment ~0 pp) |
| Four correlated metrics | No composite index | Averaging them would triple-count one shared factor and present it as three confirmations |

---

## Layout

```text
business_data_lab_assignment/
├── README.md                 this file
├── requirements.txt
├── report/
│   ├── business_conditions_report.tex    the full write-up
│   └── README.md                         how to compile
├── data/
│   ├── raw/                  the 4 supplied quarterly CSVs
│   └── processed/            combined_csbc.csv (written by notebook 01)
├── outputs/                  all figures, metric tables, quality report
├── src/
│   ├── csbc.py               shared: schema, validation, charts, pipeline
│   ├── refresh_quarter.py    pipeline CLI (dual mode)
│   └── 01…07 *.ipynb         the analysis, in run order
└── tests/
    └── test_csbc.py          15 tests
```

---

## Limitations

- **Descriptive, not causal.** Every relationship here is an association across survey cells; the
  margin-squeeze reading is the most plausible explanation of the pattern, not a demonstrated one.
- **No significance testing.** The extract has percentages but no sample sizes, weights or margins of
  error. That is why Analysis 3 ships a volatility column, it is the closest available proxy for how
  much to trust a movement.
- **Four quarters is not a trend.** It supports quarter-to-quarter comparison, not a cycle claim.
- **Expectations, not outcomes.** Whether these expectations were realised is outside this data.
- **Correlations are cross-sectional**, describing how metrics vary together across groups, with four
  periods there are too few for a time-series correlation, so nothing here says which leads which.

## Dependencies

Python 3.10+, pandas, matplotlib, JupyterLab, pytest. Version ranges are in `requirements.txt`.
