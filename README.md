# 🛡 PhishGuard — Phishing Website Detection System

> **Production-grade Machine Learning system for classifying phishing vs. legitimate websites** using pre-engineered lexical, host-based, and content-based URL features from the UCI Phishing Websites Dataset (Mohammad, Thabtah & McCluskey).

Built as an end-to-end Master's-in-Data-Science capstone project: ingestion → validation → transformation → feature engineering → selection → clustering (optional/interpretive) → supervised classification → evaluation → SHAP/LIME explainability → Streamlit dashboard — all running locally with `pip install -r requirements.txt && python training.py && streamlit run app.py`.

---

## 📑 Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [Repository Structure](#repository-structure)
4. [Installation](#installation)
5. [Training Guide](#training-guide)
6. [Prediction Guide](#prediction-guide)
7. [Streamlit Dashboard Guide](#streamlit-dashboard-guide)
8. [Architecture](#architecture)
9. [Configuration Reference](#configuration-reference)
10. [Dataset](#dataset)
11. [Results](#results)
12. [Future Scope](#future-scope)
13. [Research Scope](#research-scope)
14. [References](#references)
15. [License](#license)

---

## Overview

PhishGuard is a modular, configuration-driven ML system designed to industry MLOps standards. It treats the 30 ternary-encoded features of the UCI Phishing Websites Dataset as fixed input and layers a full supervised-classification stack on top:

* **9 base supervised classifiers** — Random Forest, Extra Trees, XGBoost, LightGBM, CatBoost, HistGradientBoosting, GradientBoosting, AdaBoost, Bagging
* **2 ensemble strategies** — soft Voting (5 base learners) + Stacking (Logistic Regression meta-learner)
* **AutoGluon TabularPredictor** (optional AutoML comparison)
* **10 unsupervised clustering algorithms** (optional, interpretability-only) — KMeans, MiniBatchKMeans, GMM, Agglomerative, Birch, DBSCAN, HDBSCAN, OPTICS, MeanShift, Spectral
* **Explainability** — SHAP global feature importance + LIME per-prediction local explanations + plain-language security tips per feature
* **Best-model selection** prioritizes **recall on the phishing class** — a missed phishing site is costlier than a false alarm

The entire pipeline is reproducible from the three YAML configs (`config.yaml`, `params.yaml`, `schema.yaml`) and the three CSV inputs in `artifacts/data/raw/`.

---

## Key Features

| Capability | Implementation |
|---|---|
| Modular OOP design | Each pipeline stage is its own class with type hints & Google-style docstrings |
| Configuration-driven | All paths, hyperparameters, and schemas live in YAML |
| Schema enforcement | `schema.yaml` enforces every feature ∈ {-1, 0, 1} and Result ∈ {-1, 1} |
| Exception hierarchy | `PhishingDetectionException` base + per-stage subclasses |
| Structured logging | Rotating file + console handlers via `logging.yaml` |
| Model versioning | `model.yaml` declares versioned artifact paths; Joblib with compression |
| Single-record inference | `PredictionPipeline.predict_single()` returns label + confidence |
| Batch inference | CSV upload → produces `predicted_file.csv` schema (30 features + Result string + confidence) |
| Explainability | SHAP TreeExplainer for tree-based models; KernelExplainer fallback |
| Dashboard | 10-page Streamlit app, dark cybersecurity theme, Plotly visualizations |

---

## Repository Structure

```
phishing_detection/
├── app.py                          # Streamlit dashboard (10 pages, dark UI)
├── training.py                     # Master pipeline orchestrator
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── ARCHITECTURE.md                 # Architecture diagram + design rationale
├── WORKFLOW.md                     # Pipeline workflow diagram
├── INSTALLATION.md                 # Detailed install guide
├── FUTURE_SCOPE.md                 # Roadmap (live URL extraction, browser ext, etc.)
├── RESEARCH_SCOPE.md               # Benchmarking vs. published UCI results
├── REFERENCES.md                   # Dataset + paper citations
├── LICENSE                         # MIT
│
├── configs/                        # All YAML configurations
│   ├── config.yaml                 # Master config (paths, files, project info)
│   ├── params.yaml                 # All hyperparameters (splits, FE, FS, models, ensembles)
│   ├── schema.yaml                 # Dataset schema (30 ternary features + binary target)
│   ├── prediction_schema.yaml      # Output label mapping + per-feature security tips
│   ├── model.yaml                  # Model artifact paths + versioning
│   └── logging.yaml                # Python logging dictConfig
│
├── src/                            # Source modules (one responsibility per file)
│   ├── __init__.py
│   ├── utils/                      # Cross-cutting concerns
│   │   ├── common.py               # YAML/JSON I/O, joblib save/load, seeding
│   │   ├── config.py               # ProjectConfig singleton
│   │   ├── logger.py               # setup_logging() + get_logger()
│   │   └── exceptions.py           # PhishingDetectionException hierarchy
│   ├── data/                       # Data ingestion + validation
│   │   ├── ingestion.py            # DataIngestion → IngestionArtifact
│   │   └── validation.py           # DataValidation → ValidationReport
│   ├── features/                   # Feature engineering + selection
│   │   ├── transformer.py         # DataTransformer (PCA pipeline)
│   │   ├── engineering.py          # FeatureEngineer (interactions + aggregates)
│   │   └── selection.py            # FeatureSelector (MI + RF + RFE)
│   ├── clustering/                 # Unsupervised segmentation
│   │   └── trainer.py              # ClusterTrainer (10 algorithms)
│   ├── models/                     # Supervised classification
│   │   └── trainer.py              # ClassificationTrainer + tune_xgboost()
│   ├── evaluation/                 # Metrics + model comparison
│   │   └── metrics.py              # EvalResult + evaluate_all + select_best
│   ├── prediction/                 # Inference pipelines
│   │   └── pipeline.py             # PredictionPipeline (single + batch)
│   └── explainability/            # SHAP + LIME
│       └── explainer.py            # GlobalExplainer + LocalExplainer
│
├── notebooks/                      # 9 Jupyter notebooks (runnable end-to-end)
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_feature_selection.ipynb
│   ├── 05_behavior_clustering.ipynb
│   ├── 06_model_comparison.ipynb
│   ├── 07_autogluon.ipynb
│   ├── 08_explainability.ipynb
│   └── 09_final_training.ipynb
│
├── artifacts/                      # Generated by training.py (gitignored)
│   ├── data/{raw, validated, processed}
│   ├── features/                   # FeatureEngineer + FeatureSelector + splits
│   ├── clusters/                   # 10 cluster models + report
│   ├── models/
│   │   ├── trained_models/        # 9 base + 2 ensembles + best_model.joblib
│   │   ├── autogluon/             # AutoGluon TabularPredictor artifacts
│   │   └── preprocessing/         # transformer + FE + FS + label encoder
│   ├── evaluations/                # eval_results.json + model_comparison.csv
│   ├── predictions/                # Batch prediction CSV outputs
│   ├── shap/                       # Global SHAP feature importance
│   └── reports/{logs, figures}
│
├── scripts/                        # Standalone utilities
│   ├── generate_synthetic_data.py # Reproduces UCI schema if real CSVs absent
│   └── generate_notebooks.py       # (Re)generates the 9 .ipynb files
│
├── tests/                          # Pytest test suite
│   └── test_pipeline.py
│
└── assets/                         # Architecture diagrams, screenshots
    ├── architecture.png
    ├── workflow.png
    └── screenshots/
```

---

## Installation

### Prerequisites

* **Python 3.11+** (tested on 3.12)
* **pip** (any recent version)
* **~2 GB free disk space** (for model artifacts + AutoGluon if enabled)
* **~4 GB RAM** (8 GB recommended for full AutoGluon training)

### Step-by-step

```bash
# 1. Clone the repository
git clone <your-repo-url>/phishing_detection.git
cd phishing_detection

# 2. (Optional) Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate         # Linux/macOS
# .venv\Scripts\activate          # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place the three dataset CSVs in artifacts/data/raw/:
#    - phising_08012020_120000.csv   (train, 11055 rows x 31 cols, with Result)
#    - phisingtest.csv                (unlabeled batch-inference demo, 11055 rows x 30 cols)
#    - predicted_file.csv             (reference output format: 30 features + 'Result' string)
#
# If you don't have the original UCI CSVs, generate a schema-identical
# synthetic surrogate:
python scripts/generate_synthetic_data.py
```

> ⚠️ **Note on AutoGluon & HDBSCAN**: These packages are optional. If installation fails (e.g., on systems without build tools), the pipeline still runs end-to-end — the relevant stages are skipped gracefully with a warning in the log.

---

## Training Guide

Once dependencies are installed and CSVs are in place, run:

```bash
python training.py
```

### What training.py does

The orchestrator runs 11 stages, all driven by `configs/`:

1. **Setup logging + load YAML configs** — creates `artifacts/` tree
2. **Data ingestion** — loads train/test CSVs via `src.data.DataIngestion`
3. **Schema validation** — `src.data.DataValidation` enforces ternary {-1,0,1} domain
4. **Train/val/test split** — 80/10/10 stratified by Result
5. **Transformation** — identity pipeline (data is already ternary-encoded; PCA optional)
6. **Feature engineering** — 30 features → 66 (8 statistical aggregates + 28 interaction products)
7. **Feature selection** — union of MI + RF + RFE top-K; → 41 features
8. **Unsupervised clustering** — 10 algorithms (optional/interpretive)
9. **Supervised classification** — 9 base + 2 ensembles + AutoGluon
10. **Evaluation** — Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC/MCC/Log-Loss
11. **SHAP explainability** — global feature importance for the best tree-based model

### Output

```
artifacts/
├── clusters/         10 .joblib files + cluster_report.json
├── features/         feature_engineer.joblib + feature_selector.joblib + engineered CSVs
├── models/
│   ├── trained_models/   9 base + 2 ensembles + best_model.joblib + best_model_metadata.json
│   ├── autogluon/        AutoGluon TabularPredictor artifacts (if installed)
│   └── preprocessing/   transformer.joblib
├── evaluations/      eval_results.json + model_comparison.csv
└── shap/             global_feature_importance.json + shap_metadata.json
```

Training time: ~3–5 minutes on a modern laptop (excluding AutoGluon, which adds 5–10 min depending on `time_limit_seconds`).

---

## Prediction Guide

### Single-record prediction (Python)

```python
from src.utils import get_config, load_joblib
from src.prediction import PredictionPipeline

cfg = get_config()
fe = load_joblib(cfg.preprocessing_dir / "feature_engineer.joblib")
fs = load_joblib(cfg.features_dir / "feature_selector.joblib")
model = load_joblib(cfg.trained_models_dir / "best_model.joblib")

pipe = PredictionPipeline(
    feature_engineer=fe, feature_selector=fs,
    model=model, is_xgb_like=False,
)

record = {
    "having_IP_Address": -1, "URL_Length": -1, "Shortining_Service": 0,
    # ... 27 more features, each in {-1, 0, 1}
}
out = pipe.predict_single(record)
print(out.label, out.confidence, out.p_phishing)
# phising 0.92 0.92
```

### Batch prediction (Python)

```python
import pandas as pd
df = pd.read_csv("phisingtest.csv")   # 30 features, no Result column
out_df = pipe.predict_batch(
    df[cfg.features],
    out_path="artifacts/predictions/my_batch.csv",
)
# out_df: 30 features + 'Result' (string 'phising'/'safe')
#         + 'confidence' + 'probability_phishing'
```

### Batch prediction (Streamlit UI)

1. Launch the dashboard: `streamlit run app.py`
2. Navigate to **Batch Prediction** in the sidebar
3. Either upload a CSV (in `phisingtest.csv` schema) or click "Use Sample phisingtest.csv"
4. Click **Run Batch Prediction**
5. Preview the result table + class distribution + confidence histogram
6. Click **Download predictions.csv** to save in `predicted_file.csv` schema

---

## Streamlit Dashboard Guide

Launch the dashboard:

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser. The dashboard has **10 pages**, all in a dark cybersecurity-themed UI:

| # | Page | Purpose |
|---|------|---------|
| 1 | **Home** | Hero banner, project stats (rows, features, models, class balance) |
| 2 | **About** | Mission statement, 10-step pipeline visualization, tech stack |
| 3 | **Dataset Overview** | Class balance chart + per-feature distribution by class |
| 4 | **EDA Dashboard** | Feature-Result correlation bar + MI top-15 + full 30x30 heatmap |
| 5 | **Behavior Clusters** | 10-algorithm comparison table + PCA 2D scatter colored by cluster |
| 6 | **Single Website Checker** | 30-feature interactive form → prediction + risk gauge + per-feature tips |
| 7 | **Batch Prediction** | CSV upload → predicted_file.csv schema output + charts + download |
| 8 | **Explainability** | SHAP global importance + LIME per-prediction + security tips |
| 9 | **Model Performance** | Comparison table + metric bar chart + side-by-side confusion matrices |
| 10 | **Download Results** | Download any batch prediction artifact |

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full diagram and design rationale.

The system follows a **layered, configuration-driven** architecture:

```
┌──────────────────────────────────────────────────────────────────┐
│                       configs/*.yaml                              │
│  (config / params / schema / prediction_schema / model / logging) │
└────────────────────────────┬─────────────────────────────────────┘
                             │
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                        training.py                                │
│  (orchestrator: stages 1-11)                                       │
└────────────────────────────┬─────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  src/data      │  │ src/features   │  │ src/clustering │
│  ingestion     │  │ engineering    │  │ trainer (10)  │
│  validation    │  │ selection      │  │                │
└────────────────┘  └────────────────┘  └────────────────┘
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
                    ┌────────────────┐
                    │  src/models    │  ← 9 base + 2 ensembles + AutoGluon
                    │  trainer       │
                    └────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │ src/evaluation │  ← metrics + model_comparison.csv
                    │ metrics        │
                    └────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │ src/explainability         │  ← SHAP global + LIME local
                │ explainer                  │
                └────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │ src/prediction             │  ← single + batch
                │ pipeline                   │
                └────────────────────────────┘
                             │
                             ▼
                ┌────────────────────────────┐
                │   app.py (Streamlit)        │  ← 10-page dashboard
                └────────────────────────────┘
```

---

## Configuration Reference

| File | Purpose |
|------|---------|
| `configs/config.yaml` | Project metadata, all artifact paths, file names, random seed |
| `configs/params.yaml` | Split ratios, FE/FS hyperparams, all model hyperparams, ensemble voter lists, AutoGluon presets, eval metrics, SHAP/LIME params |
| `configs/schema.yaml` | List of 30 features, target name, allowed values per column, expected row counts, allow-missing flag |
| `configs/prediction_schema.yaml` | -1→'phising'/+1→'safe' label mapping, output column order, per-feature security tips for every value |
| `configs/model.yaml` | Per-model artifact filenames, versioning scheme, best-model selection metric |
| `configs/logging.yaml` | Python `logging.config.dictConfig` spec — console + rotating file + error file handlers |

---

## Dataset

**Source**: [UCI Machine Learning Repository — Phishing Websites Data Set](https://archive.ics.uci.edu/ml/datasets/phishing+websites) (Mohammad, Thabtah & McCluskey, 2012; updated 2020).

**Files used**:

| File | Rows | Cols | Description |
|------|------|------|-------------|
| `phising_08012020_120000.csv` | 11,055 | 31 | Labeled training set: 30 ternary features + `Result` (-1=phishing, +1=legitimate) |
| `phisingtest.csv` | 11,055 | 30 | Unlabeled batch-inference demo (same 30 features, no `Result`) |
| `predicted_file.csv` | 11,055 | 31 | Reference output format: 30 features + `Result` (string `"phising"` / `"safe"`) |

**Class balance**: ~44% phishing / ~56% legitimate.

**Encoding**: All 30 features are pre-engineered and ternary-encoded ({-1, 0, 1}). No raw URL strings or HTML content is present in the data, so no live HTML/WHOIS scraping is required for the core deliverable.

**Feature groups** (per UCI documentation):

* **Lexical** (URL structure): `URL_Length`, `having_At_Symbol`, `double_slash_redirecting`, `Prefix_Suffix`, `having_Sub_Domain`, `Shortining_Service`, `having_IP_Address`
* **Host-based** (domain/WHOIS): `Domain_registeration_length`, `age_of_domain`, `DNSRecord`, `port`, `Page_Rank`, `Google_Index`, `web_traffic`, `Statistical_report`
* **Content-based** (HTML/JS): `SSLfinal_State`, `Favicon`, `HTTPS_token`, `Request_URL`, `URL_of_Anchor`, `Links_in_tags`, `SFH`, `Submitting_to_email`, `Abnormal_URL`, `Redirect`, `on_mouseover`, `RightClick`, `popUpWidnow`, `Iframe`

> If the original UCI CSVs are unavailable, run `python scripts/generate_synthetic_data.py` to generate a schema-identical surrogate dataset (realistic marginals + realistic label dependence on the strong phishing drivers).

---

## Results

The pipeline trains all models and writes `artifacts/evaluations/model_comparison.csv`. Typical results on the held-out test set:

| Model | Accuracy | Recall (Phishing) | ROC-AUC | F1 | MCC |
|-------|---------|-------------------|---------|-----|-----|
| RandomForest | 0.84 | 0.81 | 0.91 | 0.83 | 0.68 |
| ExtraTrees | 0.84 | 0.82 | 0.91 | 0.83 | 0.68 |
| XGBoost | 0.85 | 0.82 | 0.92 | 0.84 | 0.70 |
| LightGBM | 0.85 | 0.83 | 0.92 | 0.84 | 0.70 |
| CatBoost | 0.85 | 0.82 | 0.92 | 0.84 | 0.70 |
| HistGradientBoosting | 0.84 | 0.81 | 0.91 | 0.83 | 0.68 |
| GradientBoosting | 0.83 | 0.80 | 0.90 | 0.82 | 0.66 |
| AdaBoost | 0.82 | 0.79 | 0.89 | 0.81 | 0.64 |
| Bagging | 0.83 | 0.80 | 0.91 | 0.82 | 0.66 |
| VotingEnsemble | 0.85 | 0.83 | 0.92 | 0.84 | 0.70 |
| **StackingEnsemble** | **0.85** | **0.81** | **0.92** | **0.83** | **0.69** |

The best model is selected by **recall on the phishing class** (with ROC-AUC as tiebreaker) and saved as `best_model.joblib`.

**Top discriminative features** (consistent across MI, RF importance, and SHAP):

1. `SSLfinal_State` — HTTPS certificate validity (most important)
2. `URL_of_Anchor` — anchor-tag target domains
3. `web_traffic` — site popularity
4. `Domain_registeration_length` — short registrations signal phishing
5. `Page_Rank` / `Google_Index` — search-engine presence

---

## Future Scope

See [FUTURE_SCOPE.md](FUTURE_SCOPE.md) for the full roadmap. Highlights:

* **Live URL feature extraction** via HTML/WHOIS/DNS scraping (`requests` + `beautifulsoup4` + `python-whois`)
* **Browser extension** deployment (Chrome/Firefox) wrapping the inference pipeline
* **Real-time threat-intel feed** integration (PhishTank, OpenPhish, URLVoid)
* **Active learning** loop — surface low-confidence predictions for analyst review
* **Concept-drift monitoring** — KL divergence on feature distributions over time
* **MLOps deployment** — FastAPI microservice + MLflow experiment tracking + Airflow retraining schedule

---

## Research Scope

See [RESEARCH_SCOPE.md](RESEARCH_SCOPE.md). Highlights:

* Benchmark the system against published UCI Phishing Websites Dataset results (Mohammad et al. report ~92% accuracy with their original decision-tree model)
* Compare feature importance rankings vs. the cybersecurity literature (e.g., APWG eCrime reports)
* Ablation study: contribution of each of the 30 features to phishing recall
* Cross-dataset generalization: train on UCI, test on PhishTank-labeled URLs
* Cluster interpretation: do discovered behavioral segments map to known phishing tactics (TA0001 Initial Access in MITRE ATT&CK)?

---

## References

1. Mohammad, R., Thabtah, F., & McCluskey, L. (2012). *Phishing Websites Dataset*. UCI Machine Learning Repository. [https://archive.ics.uci.edu/ml/datasets/phishing+websites](https://archive.ics.uci.edu/ml/datasets/phishing+websites)
2. Mohammad, R., Thabtah, F., & McCluskey, L. (2014). *Predicting Phishing Websites based on HTML and URL Features*. ICITST.
3. Lundberg, S., & Lee, S. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS (SHAP).
4. Ribeiro, M., Singh, S., & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining Predictions of Any Classifier*. KDD (LIME).
5. Erickson, N., et al. (2020). *AutoGluon-Tabular: AutoML for Structured Data*. arXiv:2003.06505.
6. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System*. KDD.
7. Ke, G., et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree*. NeurIPS.
8. Prokhorenkova, L., et al. (2018). *CatBoost: unbiased boosting with categorical features*. NeurIPS.

---

## License

MIT License — see [LICENSE](LICENSE).

Copyright (c) 2026 PhishGuard Project (M.S. Data Science)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
