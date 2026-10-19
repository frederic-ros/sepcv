# Data sources and licences

The datasets below are redistributed in prepared form (subsampled, binarized, converted to
`.xlsx`/`.npz`) or, for images, as frozen feature embeddings and labels only (no image is
redistributed). The licence of each original source applies to its data; cite the source when
using it. Licences were checked on the source pages in October 2026.

Our own results, reports and figures are under CC BY 4.0; this does not relicense the data.

## data_rare27.zip — imbalanced-learn benchmark (27 datasets)

Collected with `imblearn.datasets.fetch_datasets` (Lemaître et al., 2017), which redistributes
them publicly; the imbalanced-learn documentation states no licence for the data itself.

| Dataset | Original source | Licence |
|---|---|---|
| abalone, abalone_19 | UCI Abalone (id 1) | CC BY 4.0 |
| arrhythmia | UCI Arrhythmia (id 5) | CC BY 4.0 |
| car_eval_34, car_eval_4 | UCI Car Evaluation (id 19) | CC BY 4.0 |
| coil_2000 | UCI Insurance Company Benchmark (COIL 2000) (id 125) | CC BY 4.0 |
| ecoli | UCI Ecoli (id 39) | CC BY 4.0 |
| isolet | UCI ISOLET (id 54) | CC BY 4.0 |
| letter_img | UCI Letter Recognition (id 59) | CC BY 4.0 |
| libras_move | UCI Libras Movement (id 181) | CC BY 4.0 |
| optical_digits | UCI Optical Recognition of Handwritten Digits (id 80) | CC BY 4.0 |
| ozone_level | UCI Ozone Level Detection (id 172) | CC BY 4.0 |
| pen_digits | UCI Pen-Based Recognition of Handwritten Digits (id 81) | CC BY 4.0 |
| satimage | UCI Statlog (Landsat Satellite) (id 146) | CC BY 4.0 |
| sick_euthyroid, thyroid_sick | UCI Thyroid Disease (id 102) | CC BY 4.0 |
| solar_flare_m0 | UCI Solar Flare (id 89) | CC BY 4.0 |
| spectrometer | UCI Low Resolution Spectrometer (id 93) | CC BY 4.0 |
| us_crime | UCI Communities and Crime (id 183) | CC BY 4.0 |
| wine_quality | UCI Wine Quality (id 186) | CC BY 4.0 |
| yeast_me2 | UCI Yeast (id 110) | CC BY 4.0 |
| mammography | Woods et al. (1993); OpenML 310 | no explicit licence (public on OpenML) |
| oil | Kubat et al. (1998); OpenML 311 | no explicit licence (public on OpenML) |
| scene | Boutell et al. (2004), Mulan; OpenML 312 | no explicit licence (public on OpenML) |
| yeast_ml8 | Elisseeff & Weston (2001), Mulan; OpenML 40597 | no explicit licence (public on OpenML) |
| protein_homo | KDD Cup 2004 (Caruana et al.) | no explicit licence |
| webpage | Platt (1998), LIBSVM binary datasets (w7a) | no explicit licence |

UCI pages: `https://archive.ics.uci.edu/dataset/<id>`.

## data_domain_cache.zip and data_domain_baf.zip — domain datasets

| Dataset | Source | Licence | Archive |
|---|---|---|---|
| creditcard | Dal Pozzolo et al. (2015, 2018), Kaggle `mlg-ulb/creditcardfraud` | ODbL 1.0 (database), DbCL 1.0 (contents); the prepared copy is shared under ODbL 1.0 | data_domain_cache.zip |
| taiwan_bankruptcy | Liang et al. (2016), UCI (id 572) | CC BY 4.0 | data_domain_cache.zip |
| polish_bankruptcy_y1–y5 | Zięba et al. (2016), UCI (id 365) | CC BY 4.0 | data_domain_cache.zip |
| ai4i2020 | Matzka (2020), UCI (id 601) | CC BY 4.0 | data_domain_cache.zip |
| aps_failure | Scania CV AB (2016), UCI (id 421) | CC BY 4.0; Copyright (c) 2016 Scania CV AB | data_domain_cache.zip |
| secom | McCann & Johnston (2008), UCI (id 179) | CC BY 4.0 | data_domain_cache.zip |
| baf_base | Jesus et al. (2022), Bank Account Fraud suite, Kaggle `sgpjesus/bank-account-fraud-dataset-neurips-2022` | CC BY-NC-SA 4.0 (non-commercial, ShareAlike) | data_domain_baf.zip |
| unsw_nb15_{analysis, backdoor, shellcode, worms} | Moustafa & Slay (2015), UNSW Canberra, https://research.unsw.edu.au/projects/unsw-nb15-dataset | free for academic research; redistribution not granted | **not deposited** |

**UNSW-NB15.** The prepared files are not included. Their trajectories and results are (in
`results_domains_v2.zip`), so every report and figure can be regenerated. To rerun these four
tasks, download UNSW-NB15 from the source above and prepare the attack-category-versus-normal
tasks as described in the paper; without these files, `sepcv check
configs/domains_v2.json` lists 11 domain tasks instead of 15.

Prepared datasets present in the working cache but not used in the paper (prefixes
`imblearn_`, `standard_`) are not deposited.

## data_embeddings.zip — image embeddings

Embeddings computed with frozen DINOv2 ViT-S/14 (Oquab et al., 2024) and ResNet-18 (He et al.,
2016). Only feature vectors and class labels are redistributed.

| Dataset | Source | Licence of the source |
|---|---|---|
| cifar10, cifar100 | Krizhevsky (2009) | no explicit licence (citation requested) |
| stl10 | Coates et al. (2011); images from ImageNet | no explicit licence |
| svhn | Netzer et al. (2011) | non-commercial use only |
| eurosat | Helber et al. (2019) | MIT |
| gtsrb | Stallkamp et al. (2012) | no named licence ("free to use", citation requested) |
| food101 | Bossard et al. (2014); images from Foodspotting | no explicit licence |
| caltech101 | Fei-Fei et al. (2004), data.caltech.edu | CC BY 4.0 |
| fgvc_aircraft | Maji et al. (2013) | non-commercial research only; photographers keep copyright |
| oxford_iiit_pet | Parkhi et al. (2012) | CC BY-SA 4.0 (embeddings shared under the same licence) |
| dtd | Cimpoi et al. (2014) | research use; no named licence |
| pcam | Veeling et al. (2018) | CC0 1.0 |
| fashion_mnist | Xiao et al. (2017) | MIT |
| kmnist | Clanuwat et al. (2018) | CC BY-SA 4.0 (embeddings shared under the same licence) |
| emnist_balanced | Cohen et al. (2017) | no explicit licence |

The cache also holds Flowers-102 (Nilsback & Zisserman, 2008) and MNIST (LeCun et al., 1998)
embeddings, which the paper excludes (too few rare cases at IR ≥ 49); they are kept so that
the exclusion can be checked.

Embeddings derived from SVHN and FGVC-Aircraft must not be used for commercial purposes.
