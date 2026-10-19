# Where each result of the paper comes from

Paths are relative to `reports/` (written by `python reproduce.py analyse budget halving`).
Unless stated: warm start, margin sampling, round 3 (54 labels), family `tabular+domain` for
the real tabular tasks, `image` for image embeddings, `synthetic` for the controlled factorial.
`python reproduce.py verify` checks the main values automatically (`scripts/verify_paper.py`).

## Figures

| Figure | File | Source | Script |
|---|---|---|---|
| 1 Four questions | `fig_questions.pdf` | `controlled/controlled_variance.csv`, `tabular/availability_by_ir.csv`, `tabular/decision_stability.csv`, `budget_tabular/budget_paired.csv` | `scripts/paper_figures.py` |
| 2 Controlled factorial | `fig_controlled.pdf` | `controlled/fig_controlled.pdf` | written by `sepcv analyse` (`analysis/controlled.py`) |
| 3 Rarity vs SepCV⁺ | `fig_rarity_vs_sepcv.pdf` | `tabular/task_level.csv`, `images/task_level.csv` (RF) | `scripts/paper_figures.py` |
| 4 Exploration budget | `fig_budget.pdf` | `budget_tabular/budget_reliability.csv`, `budget_paired.csv`, `budget_break_even.csv` | `scripts/paper_figures.py` |
| A.5 Dispersion of early estimators | `fig_variance_decomposition.pdf` | simulation | `scripts/fig_variance.py` |
| B.6 Binormal AUPRC | `fig_bayes.pdf` | closed form | `scripts/fig_bayes.py` |
| C.7 Synthetic geometries | `fig_geometries.pdf` | `sepcv.synthetic` | `scripts/fig_geometries.py` |
| E.8 Availability by IR | `fig_availability_ir.pdf` | `tabular/availability_by_ir.csv` | `scripts/paper_figures.py` |
| E.9 Bases × learners | `fig_base_target_tabular_domain.pdf` | `tabular/association.csv` (population `matched`) | `scripts/paper_figures.py` |
| Graphical abstract | `graphical_abstract.pdf/.png/.tiff` | values from `tabular/task_level.csv`, `images/task_level.csv`, `tabular/forecast_logo.csv`, `tabular/decision_stability.csv`, `budget_tabular/budget_paired.csv`, embedded in the script (sources listed in its header) | `scripts/graphical_abstract.py` |

## Tables

| Table | Source |
|---|---|
| 1 Positioning | literature (no computation) |
| 2 Data regimes | `tabular/tasks.csv`, `images/tasks.csv` |
| 3 Overview | summary of the tables and sections below |
| 4 Association | `tabular/association.csv` (row 1, population `own`), `tabular/assoc_paired_delta.csv` (Δρ) |
| 5 Forecasting | `tabular/forecast_logo.csv` (R², sets `SepCV+ + undefined flag (all runs)`, `Undefined flag alone (all runs)`, `Rarity, all runs`, `SepCV+`), `tabular/forecast_paired.csv` (ΔR²) |
| 6 Allocation | `tabular/allocation.csv` (learner `rf`, deploy gain and recovery), `tabular/allocation_paired.csv` |
| D.7 Protocol | `configs/*.json`, `src/sepcv/config.py` |
| E.8 Rules and start | `tabular/robust_matched.csv` |
| E.9 Components of SepCV⁺ | `tabular/sensitivity_k.csv` |
| E.10 Successive halving | `halving_tabular/halving.csv`, `halving_paired.csv` |
| E.11 Claims and evidence | summary |

## Numbers in the text

| Section | Result | Source |
|---|---|---|
| 6.1 | variance shares, observability | `controlled/controlled_variance.csv`, `controlled/controlled_by_ir.csv` |
| 6.1 | availability by IR, real tasks | `tabular/availability_by_ir.csv` |
| 6.2 | jackknife over sources | `tabular/jackknife.csv` |
| 6.2 | random acquisition, cold start | `tabular/robust_matched.csv`, `tabular/forecast_logo.csv` (strategy `random`, start `cold`) |
| 6.2 | images, synthetic | `images/association.csv`, `images/forecast_logo.csv`, `controlled/association.csv` |
| 6.3 | rank and screening stability by positives | `tabular/decision_stability.csv`, `tabular/reliability_by_positives.csv` |
| 6.3 | failure cases | `tabular/failures.csv` |
| 6.4 | exploration-free allocation, break-even | `tabular/allocation_paired.csv`, `budget_tabular/*` |
| 6.4 | improvement objective, next batch | `tabular/allocation.csv` (`imp_*`), `tabular/next_batch.csv` |
| 6.4 | calibrated screening | `tabular/screening.csv`, `tabular/screening_calsize.csv`, `images/screening.csv` |
| 6.4 | successive halving | `halving_tabular/`, `halving_images/` |
| E | S* error | `tabular/sstar_error.csv` |
