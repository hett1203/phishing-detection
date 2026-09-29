# 🏗 Architecture

> PhishGuard — Phishing Website Detection System
> M.S. Data Science Project

---

## 1. High-level Architecture

The system follows a **layered, configuration-driven** architecture. Every
pipeline stage reads its hyperparameters and I/O paths from YAML configs,
so the code is decoupled from runtime decisions. The Streamlit dashboard
loads pre-trained artifacts at launch time — no model training happens in
the app process.

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Configuration Layer                         │
│  configs/{config,params,schema,prediction_schema,model,logging}.yaml │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │ loaded once via src.utils.get_config()
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Orchestration Layer                         │
│                          training.py (11 stages)                    │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
   ┌───────────────┬───────────────┼───────────────┬───────────────┐
   ▼               ▼               ▼               ▼               ▼
┌────────┐    ┌────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  data  │    │features│    │clustering│    │  models  │    │evaluation│
│ingest- │    │  eng.  │    │ trainer  │    │ trainer  │    │ metrics  │
│ion    │    │ + sel. │    │ (10 alg) │    │ (9+2+AG) │    │ + best   │
└────┬───┘    └───┬────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
     │            │              │                 │               │
     └────────────┴──────────────┴─────────────────┴───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │       Cross-cutting          │
                    │   explainability + prediction │
                    │   (SHAP/LIME + Pipeline)      │
                    └──────────────┬───────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │      Presentation Layer      │
                    │       app.py (Streamlit)     │
                    │      10-page dashboard      │
                    └──────────────────────────────┘
```

---

## 2. Module-by-module responsibility

### `src/utils/` — Cross-cutting concerns

* **`exceptions.py`** — `PhishingDetectionException` base + per-stage
  subclasses (`ValidationError`, `ClusteringError`, `PredictionError`, …).
  All project errors are subclasses, making try/except blocks type-safe.
* **`logger.py`** — `setup_logging()` reads `logging.yaml` and configures
  console + rotating file + error file handlers. Namespaced loggers
  (`phishing_detection.data`, `phishing_detection.models`, …) make it
  easy to filter.
* **`common.py`** — `read_yaml`, `write_json`, `save_joblib`, `load_joblib`,
  `set_seed`, `map_numeric_to_label` (`-1` → `'phising'`), `project_root`,
  `ensure_dir`. The single toolkit used everywhere.
* **`config.py`** — `ProjectConfig` dataclass loads all 6 YAMLs into one
  object and resolves every path against the project root. Singleton via
  `get_config()`.

### `src/data/` — Ingestion + Validation

* **`ingestion.py`** — `DataIngestion` class loads the two raw CSVs and
  returns an `IngestionArtifact` (typed dataclass holding the train + test
  DataFrames, paths, and row counts).
* **`validation.py`** — `DataValidation` enforces `schema.yaml`:
  * Every feature ∈ {-1, 0, 1}
  * Target `Result` ∈ {-1, 1}
  * No missing values
  * Expected row counts match (11055)
  * Writes `artifacts/data/validated/validation_report.json` for auditability.
  Raises `ValidationError` on the first violation.

### `src/features/` — Transformation + Engineering + Selection

* **`transformer.py`** — `DataTransformer` builds a `sklearn.Pipeline`
  with optional `StandardScaler` + `PCA`. For the UCI ternary data this is
  an identity pipeline by default (data is already encoded), but the
  hook is in place for higher-dimensional datasets.
* **`engineering.py`** — `FeatureEngineer` adds **8 statistical aggregates**
  (`risk_score_sum`, `risk_score_mean`, `risk_score_std`, `n_suspicious`,
  `n_legit`, `n_neutral`, `phishing_ratio`, `legit_ratio`) and **top-K
  pairwise interaction products** chosen by mutual information. Expands
  30 → 66 features.
* **`selection.py`** — `FeatureSelector` ranks features by three methods
  (Mutual Information, RandomForest importance, RFE) and selects the
  **union** of the top-K from each. The original 30 features are always
  retained — they are the dataset's identity, expected by the schema.
  Reduces 66 → ~40 features.

### `src/clustering/` — Unsupervised Segmentation (optional/interpretive)

* **`trainer.py`** — `ClusterTrainer` trains **10 clustering algorithms**:
  KMeans, MiniBatchKMeans, GaussianMixture, Agglomerative, Birch, DBSCAN,
  HDBSCAN, OPTICS, MeanShift, Spectral.
* Each algorithm is wrapped in its own try/except so a single failure
  doesn't break the batch.
* Slow O(n²) methods (DBSCAN, OPTICS, Spectral, MeanShift, Agglomerative)
  train on a 3,000-row subsample for speed; the rest use the full set.
* **Selection metric**: silhouette score (primary); Davies-Bouldin (↓)
  and Calinski-Harabasz (↑) reported alongside.
* **Cluster assignments never drive the phishing/safe decision** — they
  are for interpretability (threat-actor profiling, behavioral segmentation).

### `src/models/` — Supervised Classification

* **`trainer.py`** — `ClassificationTrainer` builds and trains:
  * **9 base classifiers**: RandomForest, ExtraTrees, XGBoost, LightGBM,
    CatBoost, HistGradientBoosting, GradientBoosting, AdaBoost, Bagging
  * **2 ensembles**:
    * **VotingEnsemble** — soft voting over RF + ExtraTrees + XGBoost +
      LightGBM + CatBoost, with class-prior-weighted votes
    * **StackingEnsemble** — base learners RF + ExtraTrees + LightGBM +
      CatBoost, with a LogisticRegression meta-learner trained via
      5-fold cross-validation
  * **AutoGluon TabularPredictor** — optional AutoML comparison (skipped
    gracefully if not installed)
* **Label remapping**: XGBoost / LightGBM / CatBoost train on labels
  remapped to {0, 1}; predictions are remapped back to {-1, 1} at
  inference time. The `is_xgb_like` flag on `PredictionPipeline` handles
  this transparently.
* **Optuna tuning** — `tune_xgboost()` runs a TPE sampler study with
  5-fold CV and `roc_auc` as the optimization metric. Disabled by default
  in `params.yaml` (set `n_optuna_trials: 30` to enable).

### `src/evaluation/` — Metrics + Comparison

* **`metrics.py`** — `evaluate_all()` evaluates every trained model on
  the held-out test set using:
  * Accuracy, Precision, Recall, F1
  * **ROC-AUC, PR-AUC** (probability-based)
  * **MCC** (Matthews Correlation Coefficient — robust to class imbalance)
  * **Log-Loss** (calibration-sensitive)
  * **Confusion Matrix** (2×2)
  * **Recall on phishing class** — the primary selection metric
* `select_best()` picks the model with the highest recall on the phishing
  class, tie-broken by ROC-AUC.

### `src/explainability/` — SHAP + LIME

* **`explainer.py`** — Two classes:
  * **`GlobalExplainer`** — uses `shap.TreeExplainer` for tree-based
    models (RF, ExtraTrees, XGBoost, LightGBM, CatBoost, etc.), falls
    back to `shap.KernelExplainer` for non-tree models. Returns
    `mean(|SHAP|)` per feature.
  * **`LocalExplainer`** — uses LIME to produce per-prediction feature
    weights. Includes a `build_security_tip()` helper that maps each
    (feature, value) pair to a plain-language security tip from
    `prediction_schema.yaml`.

### `src/prediction/` — Inference Pipelines

* **`pipeline.py`** — `PredictionPipeline` wraps the FE → FS → Model
  chain so single-record and batch inference share one code path.
  * `predict_single(record_dict)` returns `PredictionOutput(label,
    confidence, p_phishing, p_safe, raw_prediction)`.
  * `predict_batch(dataframe, out_path)` returns a DataFrame in the exact
    `predicted_file.csv` schema: 30 original features + `Result` (string
    `'phising'`/`'safe'`) + `confidence` + `probability_phishing`.

### `app.py` — Presentation Layer (Streamlit)

10-page dashboard with custom CSS for a dark cybersecurity theme. Loads
all artifacts once via `@st.cache_resource` / `@st.cache_data`. No model
training in the app — only inference and visualization.

---

## 3. Design Principles

### SOLID

| Principle | Where it shows up |
|-----------|-------------------|
| **S**ingle Responsibility | One class per pipeline stage; one method per concern |
| **O**pen/Closed | New models added by extending `ClassificationTrainer._factory()` — existing code untouched |
| **L**iskov Substitution | All clustering algorithms return the same `ClusterResult` shape |
| **I**nterface Segregation | `PredictionPipeline` exposes only `predict_single` and `predict_batch` |
| **D**ependency Inversion | All stages depend on the `ProjectConfig` abstraction, not concrete YAML files |

### Configuration-driven

Everything tunable lives in `configs/*.yaml`. The code is the same in
dev, staging, and production — only the configs change. To experiment
with a different number of clusters, edit `params.yaml`; to add a new
output label, edit `prediction_schema.yaml`; no Python changes.

### Type hints + Google-style docstrings

Every public function has type hints and a Google-style docstring with
`Args:` / `Returns:` / `Raises:` / `Notes:` sections.

### Reproducibility

`set_seed(42)` is called at the start of `training.py`. All
non-deterministic algorithms (KMeans, XGBoost, LightGBM, CatBoost,
RandomForest, etc.) accept `random_state=42`. AutoGluon also gets a
fixed seed via the dataset's metadata.

### Graceful degradation

Optional dependencies (`autogluon`, `hdbscan`) are imported lazily and
wrapped in try/except. The pipeline runs end-to-end without them — the
relevant stages are skipped with a warning in the log.

---

## 4. Data Flow

```
  phising_08012020_120000.csv            phisingtest.csv
  (11055 x 31, labeled)                  (11055 x 30, unlabeled)
            │                                       │
            ▼                                       │
   DataIngestion                                     │
            │                                       │
            ▼                                       │
  DataValidation (schema.yaml)                        │
            │                                       │
            ▼                                       │
   train_test_split (80/10/10)                       │
            │                                       │
            ▼                                       │
   DataTransformer (identity)                        │
            │                                       │
            ▼                                       │
   FeatureEngineer (30 → 66 features)                │
            │                                       │
            ▼                                       │
   FeatureSelector (66 → 41 features)                 │
            │                                       │
            ▼                                       │
   ClusterTrainer (10 algorithms, optional)           │
            │                                       │
            ▼                                       │
   ClassificationTrainer (9 base + 2 ensembles)      │
            │                                       │
            ▼                                       │
   evaluate_all (metrics + comparison table)          │
            │                                       │
            ▼                                       │
   select_best (recall on phishing class)             │
            │                                       │
            ▼                                       │
   GlobalExplainer (SHAP global)                       │
            │                                       │
            ▼                                       │
   best_model.joblib + metadata.json                  │
            │                                       │
            └───────────────────┬───────────────────┘
                                ▼
                  PredictionPipeline (single + batch)
                                │
                                ▼
                      app.py (Streamlit)
```

---

## 5. Persistence & Versioning

Every trained model is serialized via `joblib.dump` with compression=3
to `artifacts/models/trained_models/<name>.joblib`. The file paths are
declared in `configs/model.yaml`:

```yaml
versioning:
  current_version: "v1"
  keep_top_n: 3
models:
  RandomForest:  { file: random_forest.joblib, type: sklearn }
  XGBoost:       { file: xgboost.joblib,       type: xgboost }
  ...
```

To roll out a new model version, bump `current_version` to `v2`,
re-train, and the Streamlit app picks up the new `best_model.joblib`
on next launch.

---

## 6. Failure modes & resilience

| Failure | Behavior |
|---------|----------|
| YAML config missing | `ConfigError` raised at startup |
| Train CSV missing | `DataIngestionError` raised at stage 1 |
| Feature value out of {-1,0,1} | `ValidationError` raised at stage 2 |
| Single clustering algorithm fails | Logged as warning, other 9 continue |
| AutoGluon not installed | Stage 8b skipped with warning |
| HDBSCAN not installed | Stage 7 skips HDBSCAN, runs other 9 |
| Single classifier fails to train | Logged as warning, other 8 continue |
| SHAP on ensemble (no TreeExplainer) | Falls back to a tree-based model from the trained set |
| Batch CSV missing columns | Missing engineered columns filled with 0 |
