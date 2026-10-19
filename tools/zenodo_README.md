# Data, results and reports for "Early performance diagnostics for rare-class active learning"

F. Ros, E. Bertrand-Bonvillain, Y. Nasser — Université d'Orléans, INSA-CVL, PRISME UR 4229.

This deposit contains everything needed to check or reproduce the paper: the input data, the
raw results of every experiment, the analysis reports from which every number of the paper is
taken, the figures, and a snapshot of the software (`sepcv` {VERSION}). The software is
developed at https://github.com/frederic-ros/<repository>.

## Contents

| File | Content |
|---|---|
| `sepcv-{VERSION}.zip` | software snapshot; see its `README.md` and `PAPER_MAP.md` |
| `data_rare27.zip` | 27 imbalanced-learn datasets (`data/rare27/*.xlsx`) |
| `data_domain_cache.zip` | prepared domain datasets (`data/cache/*.npz`, `manifest.json`), except UNSW-NB15 and BAF |
| `data_domain_baf.zip` | Bank Account Fraud, prepared (`data/cache/baf_*.npz`); CC BY-NC-SA 4.0 |
| `data_embeddings.zip` | cached image embeddings, DINOv2 ViT-S/14 and ResNet-18 (`data/embeddings/<dataset>/<backbone>.npz`) |
| `results_controlled.zip` | synthetic factorial: 120 tasks × 5 seeds × 2 starts |
| `results_rare27.zip` | 27 tabular tasks × 10 seeds × 2 starts |
| `results_domains_v2.zip` | 15 domain tasks × 10 seeds × 2 starts |
| `results_images_eswa_tasks.zip` | 42 image tasks × 5 seeds × 2 starts |
| `reports.zip` | all analysis reports (CSV, Markdown, figures) |
| `paper_figures.zip` | the figures of the paper |
| `LICENSES_DATA.md` | sources and licences of the datasets |
| `MANIFEST.sha256` | SHA-256 of every file |

## Results format

Each `results_<experiment>.zip` unpacks to `results/<experiment>/` with:

- `units/<task>__s<seed>__<start>.json`: one unit = one (task, seed, start). For every
  (acquisition rule, learner) trajectory it stores the probes (all early estimators at 38, 46,
  54 and 62 labels), the test AUPRC curve, the final AUPRC and the acquisition order (initial
  labels, then each batch). The labeled set at any round can therefore be rebuilt exactly.
- `units_extra/`: probes at 78, 102 and 126 labels, recomputed from the stored acquisition
  order (`sepcv reprobe`).
- `run_manifest.json`: configuration, configuration hash, software version and environment.

## How the deposit was produced

The trajectories were computed during the study with successive versions of `sepcv`:
the synthetic factorial with version 3.2, the tabular and image experiments with version 3.3
(after missing values were imputed from training medians only), and the post-hoc probes with
version 3.4. Later versions changed the analyses, not the trajectories. The `run_manifest.json`
of each experiment records the version that produced it; version 1.0.0 is the public release
of this 3.x line.

All reports in `reports.zip` and all figures were then regenerated from these results with the
archived software version {VERSION}, by

    python reproduce.py analyse budget halving figures verify

and `verify` confirmed that they reproduce the numbers printed in the paper. Intermediate
reports produced during the study are not archived: they are superseded by these.

## Reproducing

From the archived results (no new trajectories): unpack the software, then unpack the
`data_*.zip` and `results_*.zip` files at its root, install it (`pip install -e .`) and run
the command above. To rerun the experiments themselves: `python reproduce.py all --n-jobs 8`
(several days on 8 cores). The four UNSW-NB15 tasks cannot be rerun from this deposit: their
licence does not allow redistribution, so the prepared files are not included (their results
are). See `LICENSES_DATA.md`.

## Licence

Results, reports and figures: CC BY 4.0. Software: MIT. Datasets: the licence of each
source applies, see `LICENSES_DATA.md`. In particular, `data_domain_baf.zip` is CC BY-NC-SA 4.0
(non-commercial), the prepared credit-card data are under ODbL 1.0, and some image embeddings
derive from datasets restricted to non-commercial research (SVHN, FGVC-Aircraft) or under
ShareAlike licences (Oxford-IIIT Pet, KMNIST).
