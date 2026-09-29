# 🔄 Workflow

> PhishGuard — Pipeline Workflow
> Step-by-step data flow through training and inference.

---

## Training Workflow (`python training.py`)

```
┌───────────────────────────────────────────────────────────────┐
│  STAGE 1  —  SETUP & CONFIG LOAD                              │
│  • setup_logging() reads configs/logging.yaml                │
│  • get_config() loads all 6 YAMLs into ProjectConfig         │
│  • create_all_dirs() creates the artifacts/ tree             │
│  • set_seed(42)                                              │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 2  —  DATA INGESTION                                   │
│  • DataIngestion loads train + test CSVs                     │
│  • Output: IngestionArtifact (DataFrames + metadata)         │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 3  —  DATA VALIDATION                                  │
│  • DataValidation checks schema.yaml:                        │
│    - all 30 features present                                  │
│    - every value ∈ {-1, 0, 1}                                │
│    - Result ∈ {-1, 1}                                        │
│    - no missing values                                        │
│    - expected row count (11055)                              │
│  • Writes validation_report.json                              │
│  • Raises ValidationError on first violation                 │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 4  —  TRAIN/VAL/TEST SPLIT                            │
│  • Stratified 80/10/10 split (random_state=42)               │
│  • Persists X_train.csv, X_val.csv, X_test.csv               │
│    and y_train.csv, y_val.csv, y_test.csv                    │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 5  —  DATA TRANSFORMATION                              │
│  • DataTransformer builds identity pipeline (ternary data)    │
│    (Optional PCA available via params.yaml: apply_pca=true) │
│  • Saves transformer.joblib                                  │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 6  —  FEATURE ENGINEERING                              │
│  • mutual_info_classif ranks the 30 features                  │
│  • FeatureEngineer builds:                                    │
│    - 8 statistical aggregates (sum, mean, std, n_suspicious,  │
│      n_legit, n_neutral, phishing_ratio, legit_ratio)        │
│    - Top-K=8 pairwise interaction products                    │
│  • 30 features → 66 features                                  │
│  • Saves feature_engineer.joblib + engineered CSVs           │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 7  —  FEATURE SELECTION                                │
│  • FeatureSelector computes 3 rankings:                       │
│    - Mutual Information (filter)                              │
│    - RandomForest importance (embedded)                       │
│    - RFE with L1-LogisticRegression (wrapper)                │
│  • Selects UNION of top-15 from each method                   │
│  • Always retains the original 30 features                    │
│  • 66 features → 41 selected features                         │
│  • Saves feature_selector.joblib + selection_report.json      │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 8  —  UNSUPERVISED CLUSTERING (optional/interpretive) │
│  • ClusterTrainer trains 10 algorithms:                       │
│    KMeans, MiniBatchKMeans, GMM, Agglomerative, Birch,       │
│    DBSCAN, HDBSCAN, OPTICS, MeanShift, Spectral               │
│  • Slow O(n²) methods subsample to 3000 rows                  │
│  • Each algorithm in its own try/except (fault-tolerant)      │
│  • Computes silhouette / DB / CH for each                     │
│  • Selects best by silhouette score                           │
│  • Saves 10 .joblib files + cluster_report.json              │
│  • Cluster names via per-cluster feature-mean interpretation │
│  • IMPORTANT: clusters NEVER drive the phishing/safe decision │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 9  —  SUPERVISED CLASSIFICATION                        │
│  • ClassificationTrainer trains 9 base models:               │
│    RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost,    │
│    HistGradientBoosting, GradientBoosting, AdaBoost, Bagging  │
│  • Boosting libs (XGB/LGBM/CatBoost) train on {0,1} labels   │
│    (remapped from {-1,1} for XGBoost compatibility)          │
│  • Trains 2 ensembles:                                        │
│    - VotingEnsemble (soft voting, 5 base learners)            │
│    - StackingEnsemble (4 base + LogisticRegression meta)      │
│  • Optional: AutoGluon TabularPredictor (skipped if absent)  │
│  • Saves 9 .joblib + 2 ensemble .joblib files                │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 10  —  EVALUATION & BEST-MODEL SELECTION              │
│  • evaluate_all() scores every model on test set             │
│  • Metrics: accuracy, precision, recall, F1,                  │
│    ROC-AUC, PR-AUC, MCC, Log-Loss, confusion matrix           │
│  • select_best() picks model with highest                     │
│    recall_phishing (tiebreak: roc_auc)                        │
│  • Saves best_model.joblib + best_model_metadata.json         │
│  • Saves model_comparison.csv (sorted table)                  │
└─────────────────────────┬─────────────────────────────────────┘
                          ▼
┌───────────────────────────────────────────────────────────────┐
│  STAGE 11  —  SHAP EXPLAINABILITY                             │
│  • GlobalExplainer uses TreeExplainer on a tree-based model   │
│  • For ensembles (Voting/Stacking) falls back to a tree       │
│    model from the trained set (preferring XGBoost)            │
│  • Computes mean(|SHAP|) per feature                          │
│  • Saves global_feature_importance.json + shap_metadata.json │
└───────────────────────────────────────────────────────────────┘
```

---

## Inference Workflow (single record)

```
   user input (30-feature dict, each ∈ {-1,0,1})
            │
            ▼
   load feature_engineer.joblib (loaded once)
   load feature_selector.joblib (loaded once)
   load best_model.joblib       (loaded once)
            │
            ▼
   FeatureEngineer.transform(dict)
            │ adds 8 statistical aggregates
            │ + 28 interaction products
            ▼
   FeatureSelector.transform(df)
            │ keeps the 41 selected features
            ▼
   best_model.predict_proba(df)
            │ returns P(phishing) and P(safe)
            ▼
   map_numeric_to_label(-1 or 1)
            │ 'phising' or 'safe'
            ▼
   PredictionOutput(
     label, confidence, p_phishing, p_safe, raw
   )
```

---

## Inference Workflow (batch CSV)

```
   uploaded CSV (in phisingtest.csv schema: 30 features, no Result)
            │
            ▼
   PredictionPipeline.predict_batch(df, out_path)
            │
            ▼
   FeatureEngineer.transform(df) → 66 features
            │
            ▼
   FeatureSelector.transform(df) → 41 features
            │
            ▼
   best_model.predict() → hard labels in {-1, 1}
   best_model.predict_proba() → P(phishing) per row
            │
            ▼
   Build output DataFrame:
     • original 30 features
     • Result (string 'phising'/'safe')
     • confidence = max-class probability
     • probability_phishing = P(class = phishing)
            │
            ▼
   Save to artifacts/predictions/batch_pred_<timestamp>.csv
   + Display in Streamlit + offer CSV download
```

---

## Streamlit Launch Workflow

```
   user runs: streamlit run app.py
            │
            ▼
   app.py imports src.utils.get_config()
   ProjectConfig.load() reads all 6 YAMLs
            │
            ▼
   _inject_css() applies dark cybersecurity theme
            │
            ▼
   Sidebar renders 10-page navigation
            │
            ▼
   On first page visit, @st.cache_resource loads:
     - feature_engineer.joblib
     - feature_selector.joblib
     - best_model.joblib
     - best_model_metadata.json
   On first dataset page, @st.cache_data loads:
     - phising_08012020_120000.csv
     - predicted_file.csv (reference)
     - eval_results.json
     - cluster_report.json
     - global_feature_importance.json
            │
            ▼
   User navigates between pages; each page renders
   Plotly charts with dark theme + custom colors
```
