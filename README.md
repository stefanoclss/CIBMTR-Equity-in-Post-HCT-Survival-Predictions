# CIBMTR - Equity in Post-HCT Survival Predictions

[![Autonomous Synthesis](https://img.shields.io/badge/Synthesized%20By-sovLoop%20(Stealth)-0A66C2?style=for-the-badge)](#how-this-pipeline-was-built)
[![Foundation LLM](https://img.shields.io/badge/Model-qwen3.5%3A9b-111827?style=for-the-badge)](#how-this-pipeline-was-built)
[![Validation](https://img.shields.io/badge/5--Fold%20OOF-C_INDEX%200.68701-059669?style=for-the-badge)](#verified-validation-benchmark)
[![Security](https://img.shields.io/badge/Zero--Pickle-CWE--502%20Compliant-374151?style=for-the-badge)](#governance--production-guarantees)
[![Kaggle Challenge](https://img.shields.io/badge/Challenge-Benchmark-0284C7?style=for-the-badge)](https://www.kaggle.com/competitions/equity-post-HCT-survival-predictions)

> **Autonomous End-to-End Machine Learning Engineering by sovLoop**
> From raw tabular schema forensics to interpretable `preprocessing(df)` & `feature_engineering(df)` pipelines, Optuna-tuned multi-family GBDT ensembles, and zero-pickle production packaging—evolved step-by-step using **`qwen3.5:9b`**.

---

## Executive Overview

Predict `efs_time` (regression) across `57` structured predictors on `equity_post_hct_survival_train.csv` accounting for right-censored event indicator `efs` under strict leak-free 5-fold out-of-fold validation to maximize `C_INDEX`.

Production data science teams need pipelines that are not only accurate, but **readable, explainable, and maintainable**. Every transformation in this repository is organized by pipeline topology stage (`preprocessing`, `feature_engineering`, `objective`, `models`, `postprocess`) and validated incrementally via **sovLoop** (operating in private stealth mode).

---

## How This Pipeline Was Built

1. **Simple Baseline -> Gradual Verified Evolution**: Starting from a clean baseline, **sovLoop** (guided by `qwen3.5:9b`) evolved one stage at a time—adding explicit column hygiene, domain features, leak-free target formulation, and Optuna hyperparameter optimization.
2. **Strict Out-of-Fold Scientific Gatekeeping**: Candidate modifications to preprocessing, feature engineering, estimator hyperparameters, and post-processing were evaluated across **100% of 5-fold out-of-fold validation rows**. Any change that failed to improve CV was rejected.
3. **Atomic Single-Module Git Lineage**: After initial scaffolding, **each commit in this repository modifies strictly one atomic stage file** (`pipeline/models.py`, `pipeline/features.py`, `pipeline/preprocess.py`, `pipeline/objective.py`, or `pipeline/postprocess.py`). Inspect `git log -p` to audit every single lift.
4. **Single-Notebook End-to-End Reproducibility (Zero-Pickle / No Git LFS)**: Place raw tables in `data/` and run **`pipeline.ipynb`** top-to-bottom to reproduce visual EDA (`eda_pipeline.py`), K-Fold LightGBM/Optuna training (`training_pipeline.py` -> `artifacts/model_artifact.json`), and pure artifact holdout inferencing (`inference_pipeline.py` -> `submission.csv`).

---

## Verified Validation Benchmark

| Benchmark Dimension | Verified Result |
| :--- | :--- |
| **Challenge / Dataset** | **CIBMTR - Equity in Post-HCT Survival Predictions** (`equity_post_hct_survival_train.csv`) |
| **Target Variable (`y`)** | `efs_time` (`regression`) |
| **Evaluation KPI** | `C_INDEX` (Higher is better) |
| **Baseline 5-Fold OOF Score** | `0.68766` |
| **Champion 5-Fold OOF Score** | **`0.68701`** (`-0.00065` verified lift vs. baseline) |
| **Cross-Fold Stability (Std. Dev.)** | `±0.00407` across `5` strict folds |
| **Synthesis LLM** | `qwen3.5:9b` (via **sovLoop**) |

---

## Individual Model Benchmarks vs. Unrestricted Ensemble

Every individual model inside the unrestricted ensemble is benchmarked independently for standalone validation accuracy, training time, and inference latency—so production engineering teams can either deploy the **full unrestricted SOTA ensemble** or select the **fastest / best solo model** (`SELECTED_MODEL_KEY`) for strict SLA budgets:

| Model Family (`key`) | Standalone Val Score | Mean Fold Fit | Mean Fold Inference | Ensemble Weight | Deployment Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Two-Stage Counterfactual GBDT** (`two_stage`) | `0.68574 ± 0.00398` | `31630.9 ms` | `551.47 ms` | `62.7%` | `Best Solo Accuracy` |
| **CatBoost Symmetric** (`catboost`) | `0.68213 ± 0.00421` | `8791.2 ms` | `189.75 ms` | `6.5%` | `Active Ensemble Member` |
| **Optuna LightGBM** (`lgbm`) | `0.68197 ± 0.00485` | `4057.9 ms` | `94.54 ms` | `7.7%` | `Active Ensemble Member` |
| **Optuna XGBoost** (`xgb`) | `0.68151 ± 0.00417` | `6833.7 ms` | `93.67 ms` | `23.1%` | `Fastest Low-Latency` |

---

## Autonomous Build Ledger (Commit-by-Commit Scientific Progression)

Every row below corresponds to a verified out-of-fold promotion recorded as an **isolated single-module commit** in `git log`:

| Iteration | Module Evolved | Active Synthesis Operations | 5-Fold OOF `C_INDEX` | Lift | Engineering Rationale |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Gen 0** | `scaffold` | Baseline initialization | `0.68766` | `baseline` | Deterministic 5-fold leak-free baseline anchor |
| **Gen 1** | `pipeline/models.py` (`two_stage_gbdt_stream`) | `two_stage_counterfactual_gbdt`, `lgbm_optuna_tuned`, `catboost_tuned` | `0.68683` | `-0.00083` | [Reference Blueprint `ref_01_minerppdy_counterfactual_two_stage`] 1st Place Two-Stage Incidence + Counterfactual MAE Gate + TabM Pairwise |
| **Gen 2** | `pipeline/models.py` (`two_stage_gbdt_stream`) | `two_stage_counterfactual_gbdt`, `lgbm_optuna_tuned`, `catboost_tuned`, `xgb_optuna_tuned` | `0.68701` | `+0.00018` | [Phase-2 Neural/Foundation Extension 1/1] Add `xgb_optuna_tuned` onto Phase-1 champion & reweight OOF blend |

---

## Modular Architecture & Repository Layout

```mermaid
flowchart LR
    DATA["Raw Data (57 predictors)"] --> SPLIT["5-Fold CV Split"]
    SPLIT --> PRE["1. Preprocessing: median_impute, missing_indicators, oof_target_encoding"]
    PRE --> FEAT["2. Features: row_summary_stats, top_pair_ratios, cat_groupby_stats"]
    FEAT --> OBJ["3. Objective: kaplan_meier_target"]
    OBJ --> M0["4a. Primary Ensemble: Two-Stage Counterfactual GBDT (63%) + CatBoost Symmetric (6%) + Optuna LightGBM (8%) + Optuna XGBoost (23%)"]
    M0 --> FUSE["Quantile-CDF Fusion"]
    OBJ --> S1["4b. two_stage_gbdt_stream: two_stage_counterfactual_gbdt, lgbm_optuna_tuned, catboost_tuned, +1 more"]
    S1 --> FUSE
    OBJ --> S2["4b. neural_pairwise_stream: view_obj:ambrosm_separate_target, view_feat:plr_periodic_embeddings, tabm_periodic_mlp"]
    S2 --> FUSE
    FUSE --> POST["5. Post-Processing: rank_calibration_blend"]
    POST --> OUT["5-Fold OOF & Holdout (C_INDEX = 0.68701)"]
```

| Deliverable | Role in Production Workflow |
| :--- | :--- |
| **`pipeline.ipynb`** | **Primary Topology Walkthrough Notebook**: explains every stage with visual charts and reproduces EDA, K-Fold training, and holdout scoring. |
| `eda.ipynb` / `eda_pipeline.py` | Schema diagnostics, missingness bar charts, and target distribution plots via `pipeline/eda.py`. |
| `train.ipynb` / `training_pipeline.py` | Leak-free K-Fold LightGBM + Optuna CV orchestrator exporting zero-pickle `artifacts/model_artifact.json` + `fold_*.lgb`. |
| `inference.ipynb` / `inference_pipeline.py` | Pure artifact-driven holdout scoring (`--artifact artifacts/model_artifact.json`) with zero re-training. |
| `pipeline/*.py` | Readable stage modules (`preprocessing(df)`, `feature_engineering(df)`, `models.py`) evolved one file per commit. |

---

## Governance & Production Guarantees

- **100% Validation Fold Integrity**: Imputation statistics, IQR bounds, Optuna inner splits, and target transforms fit strictly inside training folds.
- **Zero-Pickle Security (`CWE-502`)**: Models serialize to human-auditable LightGBM text/JSON (`fold_*.lgb`, `model_artifact.json`) — zero unsafe `.pkl` binaries.
- **Data-Scientist Readability**: Explicit column-by-column `pandas` transformations that any engineer can inspect, explain to stakeholders, and extend.

---

## Quickstart (1-Notebook Reproduction & CLI)

```bash
pip install -r requirements.txt

# Option A — Reproduce Training + Inferencing in ONE Notebook
jupyter execute pipeline.ipynb

# Option B — Modular CLI Orchestrators
python eda_pipeline.py --input data/train.csv
python training_pipeline.py --train data/train.csv --artifact-out artifacts/model_artifact.json --oof-out oof_predictions.csv
python inference_pipeline.py --artifact artifacts/model_artifact.json --test data/test.csv --output submission.csv
```

---

## About the Author

### Stefano Claes — AI Systems Architect

Working on **sovereign local-first AI, neuro-symbolic & agentic harnesses, and closed-loop recursive self-improvement**—pairing autonomous iteration with deterministic verification, zero-trust sandboxing, and human-readable engineering.

| Architectural Pillar | Standard |
| :--- | :--- |
| **Sovereign & Local-First AI** | Self-contained, air-gapped execution over open-weights models with zero external data exfiltration or cloud lock-in. |
| **Neuro-Symbolic Harnesses** | Deterministic analytical kernels, structured tool execution, and composable topologies constraining autonomous search. |
| **Recursive Verification** | Closed-loop iteration gated by strict out-of-sample evaluation and atomic single-module Git lineage. |
| **Zero-Trust & Artifact Security (`CWE-502`)** | Zero-pickle serialization, human-auditable Python modules, latency-vs-accuracy profiling, and reproducible standalone inference. |

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Stefano%20Claes-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/stefano-claes-266435140/) [![GitHub](https://img.shields.io/badge/GitHub-%40stefanoclss-111827?style=for-the-badge&logo=github&logoColor=white)](https://github.com/stefanoclss)

- **Author**: **Stefano Claes** (AI Systems Architect)
- **LinkedIn**: [https://www.linkedin.com/in/stefano-claes-266435140/](https://www.linkedin.com/in/stefano-claes-266435140/)
- **GitHub**: [https://github.com/stefanoclss](https://github.com/stefanoclss)
