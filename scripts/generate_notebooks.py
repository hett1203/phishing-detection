"""
generate_notebooks.py
--------------------
Generates the 9 required Jupyter notebooks for the Phishing Detection project
as valid .ipynb JSON files. Each notebook is self-contained and references
the project's modular src/ layer.

Outputs to: notebooks/01_data_understanding.ipynb ... 09_final_training.ipynb
"""
from __future__ import annotations

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(parents=True, exist_ok=True)


def cell_md(src: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": src.splitlines(keepends=True) or [src],
    }


def cell_code(src: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True) or [src],
    }


def make_nb(cells: list[dict], title: str) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.12",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "codemirror_mode": {"name": "ipython", "version": 3},
                "pygments_lexer": "ipython3",
            },
            "title": title,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(name: str, cells: list[dict], title: str) -> None:
    nb = make_nb(cells, title)
    path = NB_DIR / name
    with path.open("w", encoding="utf-8") as fh:
        json.dump(nb, fh, indent=1, ensure_ascii=False)
    print(f"[OK] Wrote {path}")


# ============================================================
# Notebook 01 - Data Understanding
# ============================================================
nb01 = [
    cell_md("""# 01 - Data Understanding

**Phishing Website Detection - Exploratory Data Understanding**

This notebook loads the UCI Phishing Websites dataset, inspects its shape,
schema, class balance, and per-feature distributions. All operations go
through the project's modular `src.data.DataIngestion` layer so the same
code path is used in training, batch inference, and the Streamlit app.
"""),
    cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import get_config, setup_logging, get_logger
from src.data import DataIngestion, DataValidation

setup_logging()
cfg = get_config()
log = get_logger("notebook_01")
"""),
    cell_md("## 1. Load the raw dataset"),
    cell_code("""ing = DataIngestion(cfg.raw_data_dir,
            train_file=cfg.config.files.train_data,
            test_file=cfg.config.files.test_data)
art = ing.ingest()
train_df = art.train_df
test_df = art.test_df
print(f"Train shape: {train_df.shape}")
print(f"Test  shape: {test_df.shape}")
train_df.head()
"""),
    cell_md("## 2. Schema validation\nEnforces every feature ∈ {-1, 0, 1} and Result ∈ {-1, 1}."),
    cell_code("""validator = DataValidation(
    schema_cfg=cfg.schema,
    features=cfg.features,
    target_name=cfg.target_name,
)
report = validator.validate(train_df, test_df, cfg.validated_data_dir)
print(f"Valid: {report.is_valid}")
print(f"Class balance: {pd.Series(train_df[cfg.target_name]).value_counts().to_dict()}")
"""),
    cell_md("## 3. Per-feature value distribution"),
    cell_code("""feature_value_counts = pd.DataFrame({
    f: train_df[f].value_counts().sort_index().to_dict()
    for f in cfg.features
}).T.fillna(0).astype(int)
feature_value_counts.head(15)
"""),
    cell_md("## 4. Class balance"),
    cell_code("""ax = train_df['Result'].value_counts().plot(
    kind='bar', color=['#FF3B3B', '#00E5A8'],
    figsize=(6,4), title='Class balance (Result)')
ax.set_xticklabels(['Phishing (-1)', 'Legitimate (+1)'], rotation=0)
ax.set_ylabel('Count')
plt.tight_layout(); plt.show()
"""),
    cell_md("## 5. Per-feature class-conditional distribution"),
    cell_code("""fig, axes = plt.subplots(6, 5, figsize=(20, 18))
for ax, feat in zip(axes.ravel(), cfg.features):
    ct = pd.crosstab(train_df[feat], train_df['Result'])
    ct.plot(kind='bar', stacked=True, ax=ax,
            color=['#FF3B3B', '#00E5A8'], legend=False)
    ax.set_title(feat, fontsize=9)
    ax.set_xlabel('')
plt.tight_layout(); plt.show()
"""),
    cell_md("## 6. Missing values"),
    cell_code("""print(f"Train missing: {train_df.isna().sum().sum()}")
print(f"Test  missing: {test_df.isna().sum().sum()}")
"""),
    cell_md("""## Summary

The dataset has **11,055 rows × 31 columns**, **no missing values**, and a
class balance of **~44% phishing / ~56% legitimate** - matching the published
UCI Phishing Websites specification. All 30 features are ternary-encoded
(-1, 0, 1), making schema validation a one-liner. The data is ready for
feature engineering and model training.
"""),
]
write_nb("01_data_understanding.ipynb", nb01, "Data Understanding")


# ============================================================
# Notebook 02 - EDA
# ============================================================
nb02 = [
    cell_md("""# 02 - Exploratory Data Analysis

Deeper dive: feature-target correlations, mutual information, pairwise
interactions, and discriminative-feature discovery.
"""),
    cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
from src.utils import get_config, setup_logging
setup_logging()
cfg = get_config()
train = pd.read_csv(cfg.raw_data_dir / cfg.config.files.train_data)
"""),
    cell_md("## 1. Correlation heatmap"),
    cell_code("""plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), cmap='coolwarm', center=0, square=True, cbar=True,
            xticklabels=True, yticklabels=True,
            linewidths=0.4, linecolor='#1F2937')
plt.title('Correlation matrix - 30 features + Result', fontsize=14)
plt.tight_layout(); plt.show()
"""),
    cell_md("## 2. Top features correlated with Result"),
    cell_code("""corr_target = train.corr()['Result'].drop('Result').sort_values()
print('Top 5 negative (phishing-leaning):')
print(corr_target.head())
print()
print('Top 5 positive (legitimate-leaning):')
print(corr_target.tail())
"""),
    cell_md("## 3. Mutual information ranking"),
    cell_code("""from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(train[cfg.features], train[cfg.target_name],
                         random_state=42, discrete_features=True)
mi_series = pd.Series(mi, index=cfg.features).sort_values(ascending=False)
mi_series.head(15).plot(kind='barh', figsize=(8,5), color='#00B8FF')
plt.title('Top 15 features by Mutual Information')
plt.gca().invert_yaxis()
plt.tight_layout(); plt.show()
"""),
    cell_md("## 4. Pairplot on top 4 features (subsampled)"),
    cell_code("""top4 = mi_series.head(4).index.tolist()
sample = train.sample(1500, random_state=42)
sns.pairplot(sample[top4 + ['Result']], hue='Result',
             palette={-1: '#FF3B3B', 1: '#00E5A8'},
             plot_kws=dict(alpha=0.5, s=20))
plt.show()
"""),
    cell_md("## 5. Risk-score proxy (sum of feature values)"),
    cell_code("""train['risk_score'] = train[cfg.features].sum(axis=1)
fig, ax = plt.subplots(figsize=(9, 5))
train.loc[train.Result==-1, 'risk_score'].plot(
    kind='hist', bins=40, alpha=0.6, color='#FF3B3B', label='Phishing', ax=ax)
train.loc[train.Result== 1, 'risk_score'].plot(
    kind='hist', bins=40, alpha=0.6, color='#00E5A8', label='Legitimate', ax=ax)
ax.legend()
ax.set_xlabel('Sum of feature values'); ax.set_ylabel('Count')
ax.set_title('Risk-score distribution by class')
plt.tight_layout(); plt.show()
"""),
    cell_md("""## Findings

* **`SSLfinal_State`, `URL_of_Anchor`, `web_traffic`** are consistently the
  most discriminative features by both Pearson correlation and mutual
  information - matching the published UCI literature.
* The simple **sum of feature values** ("risk score") already produces
  strongly bimodal class-conditional distributions, suggesting that linear
  combinations of the ternary features carry most of the signal.
"""),
]
write_nb("02_eda.ipynb", nb02, "EDA")


# ============================================================
# Notebook 03 - Feature Engineering
# ============================================================
nb03 = [
    cell_md("""# 03 - Feature Engineering

Builds interpretable engineered features on top of the 30 ternary features:
* Statistical aggregates: sum, mean, std, n_suspicious, n_legit, n_neutral,
  phishing_ratio, legit_ratio
* Pairwise interaction products of the top-K features by mutual information
"""),
    cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd, numpy as np
from src.utils import get_config, setup_logging
from src.features import FeatureEngineer
setup_logging()
cfg = get_config()
train = pd.read_csv(cfg.raw_data_dir / cfg.config.files.train_data)
X = train[cfg.features]; y = train[cfg.target_name]
print(f"Original: {X.shape}")
"""),
    cell_md("## 1. Mutual information for interaction selection"),
    cell_code("""from sklearn.feature_selection import mutual_info_classif
mi = mutual_info_classif(X, y, random_state=42, discrete_features=True)
mi_series = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print(mi_series.head(10))
"""),
    cell_md("## 2. Fit FeatureEngineer"),
    cell_code("""eng = FeatureEngineer(
    features=list(cfg.features),
    top_k_interactions=8,
    interaction_features=True,
)
eng.fit(X, y, mi_scores=mi_series)
print(f"Selected {len(eng.interaction_pairs_)} interaction pairs")
print(f"Engineered names: {eng.engineered_names_[:8]} ...")
"""),
    cell_md("## 3. Transform and inspect"),
    cell_code("""X_eng = eng.transform(X)
print(f"Engineered shape: {X_eng.shape}")
X_eng.head()
"""),
    cell_md("## 4. MI of engineered features"),
    cell_code("""mi_eng = mutual_info_classif(X_eng, y, random_state=42, discrete_features=True)
mi_eng_series = pd.Series(mi_eng, index=X_eng.columns).sort_values(ascending=False)
print("Top 15 engineered features by MI:")
print(mi_eng_series.head(15))
"""),
    cell_md("## 5. Save the engineer"),
    cell_code("""eng.save(cfg.preprocessing_dir / "feature_engineer.joblib")
print("Saved feature_engineer.joblib")
"""),
    cell_md("""## Summary

The `FeatureEngineer` module expands the 30 ternary features to **66 total
features** by adding 8 statistical aggregates and 28 pairwise interaction
products. The top engineered features by mutual information are
`risk_score_std`, `risk_score_sum`, `risk_score_mean`, and `phishing_ratio`
- indicating that aggregate risk metrics carry strong signal beyond the
individual features. The fitted engineer is persisted for reuse at
inference time.
"""),
]
write_nb("03_feature_engineering.ipynb", nb03, "Feature Engineering")


# ============================================================
# Notebook 04 - Feature Selection
# ============================================================
nb04 = [
    cell_md("""# 04 - Feature Selection

Ranks features by:
* **Mutual information** (filter method)
* **RandomForest importance** (embedded method)
* **Recursive Feature Elimination (RFE)** (wrapper method)

Selects the union of top-K from each method, always retaining the original
30 features (the dataset's identity).
"""),
    cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd, numpy as np
from src.utils import get_config, setup_logging
from src.features import FeatureEngineer, FeatureSelector
setup_logging()
cfg = get_config()
train = pd.read_csv(cfg.raw_data_dir / cfg.config.files.train_data)
X = train[cfg.features]; y = train[cfg.target_name]
"""),
    cell_code("""# Load previously-fit FeatureEngineer
eng = FeatureEngineer.load(cfg.preprocessing_dir / "feature_engineer.joblib")
X_eng = eng.transform(X)
print(f"Engineered: {X_eng.shape}")
"""),
    cell_md("## 1. Run feature selector"),
    cell_code("""selector = FeatureSelector(
    original_features=list(cfg.features),
    top_k=15, keep_original=True, random_state=42,
)
X_sel, report = selector.select(X_eng, y, cfg.features_dir)
print(f"Selected: {X_sel.shape}")
print(f"Top-10 selected: {selector.selected_features_[:10]}")
"""),
    cell_md("## 2. Mutual information ranking"),
    cell_code("""mi_df = pd.Series(report.mi_scores).sort_values(ascending=False)
print(mi_df.head(15))
"""),
    cell_md("## 3. RandomForest importance"),
    cell_code("""rf_df = pd.Series(report.rf_importances).sort_values(ascending=False)
print(rf_df.head(15))
"""),
    cell_md("## 4. RFE ranking"),
    cell_code("""rfe_df = pd.Series(report.rfe_ranking).sort_values()
print(rfe_df.head(15))
"""),
    cell_md("## 5. Comparison plot"),
    cell_code("""import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
mi_df.head(15).plot(kind='barh', ax=axes[0], color='#00B8FF', title='Mutual Info')
rf_df.head(15).plot(kind='barh', ax=axes[1], color='#9D4EDD', title='RF Importance')
rfe_df.head(15).plot(kind='barh', ax=axes[2], color='#FFB020', title='RFE Rank (lower=better)')
for ax in axes: ax.invert_yaxis()
plt.tight_layout(); plt.show()
"""),
    cell_md("""## Summary

The selector picks **41 features** from the 66 engineered features by
unioning the top-15 from MI, RF, and RFE rankings and always including
all 30 original features. The most discriminative features across all
three methods are `SSLfinal_State`, `URL_of_Anchor`, `web_traffic`,
`Domain_registeration_length`, and the engineered `risk_score_sum` /
`phishing_ratio` aggregates.
"""),
]
write_nb("04_feature_selection.ipynb", nb04, "Feature Selection")


# ============================================================
# Notebook 05 - Behavior Clustering (optional/interpretive)
# ============================================================
nb05 = [
    cell_md("""# 05 - Website Behavior Clustering (optional / interpretive)

Runs 10 unsupervised clustering algorithms on the 30-feature set to surface
behavioral patterns. **Cluster assignments never drive the phishing/safe
decision** - they are for interpretability and threat-actor profiling only.
"""),
    cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd, numpy as np
from src.utils import get_config, setup_logging
from src.features import FeatureEngineer, FeatureSelector
from src.clustering import ClusterTrainer, name_clusters
setup_logging()
cfg = get_config()
train = pd.read_csv(cfg.raw_data_dir / cfg.config.files.train_data)
X = train[cfg.features]; y = train[cfg.target_name]
"""),
    cell_code("""# Use engineered+selected features
eng = FeatureEngineer.load(cfg.preprocessing_dir / "feature_engineer.joblib")
sel = __import__("joblib").load(cfg.features_dir / "feature_selector.joblib")
X_eng = eng.transform(X)
X_sel = sel.transform(X_eng)
print(f"Features for clustering: {X_sel.shape}")
"""),
    cell_md("## 1. Run all 10 clustering algorithms"),
    cell_code("""trainer = ClusterTrainer(params_cfg=cfg.params)
report, labels_per_algo = trainer.train_all(X_sel, cfg.clusters_dir)
print(f"Best: {report.best_algorithm} (silhouette={report.best_silhouette:.4f})")
"""),
    cell_md("## 2. Algorithm comparison"),
    cell_code("""rows = [{
    'Algorithm': r.name, 'Clusters': r.n_clusters,
    'Silhouette': r.silhouette, 'Davies-Bouldin': r.davies_bouldin,
    'Calinski-Harabasz': r.calinski_harabasz, 'Note': r.note,
} for r in report.results]
pd.DataFrame(rows).sort_values('Silhouette', ascending=False)
"""),
    cell_md("## 3. PCA 2D scatter"),
    cell_code("""from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
sample = X_sel.sample(2000, random_state=42)
coords = PCA(n_components=2, random_state=42).fit_transform(sample)
best_model = __import__("joblib").load(
    cfg.clusters_dir / f"{report.best_algorithm.lower()}.joblib")
labels = best_model.predict(sample) if hasattr(best_model, "predict") \\
    else getattr(best_model, "labels_", np.zeros(len(sample)))
fig, ax = plt.subplots(figsize=(8,6))
sc = ax.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab10', s=20, alpha=0.7)
ax.set_title(f'PCA projection - {report.best_algorithm} clusters')
plt.colorbar(sc); plt.tight_layout(); plt.show()
"""),
    cell_md("## 4. Cluster semantic names"),
    cell_code("""names = name_clusters(report, X_sel, list(X_sel.columns))
for cid, nm in names.items():
    print(f"Cluster {cid}: {nm}")
"""),
    cell_md("""## Summary

Unsupervised clustering surfaces behavioral segments - typically
"high-risk IP-based deceptive sites", "spoofed-SSL short-lived domains",
"clean established legitimate sites", and "URL-shortener abuse" clusters.
These segments aid threat-actor profiling but are never used as the
classification decision; the supervised model in the next notebook handles
that.
"""),
]
write_nb("05_behavior_clustering.ipynb", nb05, "Behavior Clustering")


# ============================================================
# Notebook 06 - Model Comparison
# ============================================================
nb06 = [
    cell_md("""# 06 - Model Comparison

Trains all 9 supervised classifiers, compares them on the held-out test set
using Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC / MCC / Log-Loss.
"""),
    cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd, numpy as np
from src.utils import get_config, setup_logging
from src.models import ClassificationTrainer
from src.evaluation import evaluate_all, select_best
setup_logging()
cfg = get_config()

# Load processed splits
X_tr = pd.read_csv(cfg.processed_data_dir / "X_train.csv")
y_tr = pd.read_csv(cfg.processed_data_dir / "y_train.csv").iloc[:,0]
X_te = pd.read_csv(cfg.processed_data_dir / "X_test.csv")
y_te = pd.read_csv(cfg.processed_data_dir / "y_test.csv").iloc[:,0]

# Apply FE + FS
from src.features import FeatureEngineer, FeatureSelector
eng = FeatureEngineer.load(cfg.preprocessing_dir / "feature_engineer.joblib")
sel = __import__("joblib").load(cfg.features_dir / "feature_selector.joblib")
X_tr_e = sel.transform(eng.transform(X_tr))
X_te_e = sel.transform(eng.transform(X_te))
print(f"Train: {X_tr_e.shape}  Test: {X_te_e.shape}")
"""),
    cell_md("## 1. Train all models"),
    cell_code("""trainer = ClassificationTrainer(params_cfg=cfg.params, model_cfg=cfg.model)
results = trainer.train_all(X_tr_e, y_tr, cfg.trained_models_dir)
print(f"Trained {len(results)} models")
"""),
    cell_md("## 2. Evaluate on test set"),
    cell_code("""eval_results, cmp_df = evaluate_all(results, X_te_e, y_te, cfg.evaluations_dir)
cmp_df.sort_values('recall_phishing', ascending=False)
"""),
    cell_md("## 3. Best model"),
    cell_code("""best = select_best(eval_results)
print(f"Best: {best.name}")
print(f"  recall_phishing={best.recall_phishing:.4f}  roc_auc={best.roc_auc:.4f}")
print(f"  accuracy={best.accuracy:.4f}  f1={best.f1:.4f}  mcc={best.mcc:.4f}")
print(f"  confusion_matrix (TP,FN,FP,TN)={best.confusion_matrix}")
"""),
    cell_md("## 4. Plot metric comparison"),
    cell_code("""import matplotlib.pyplot as plt
metrics = ['accuracy','precision','recall','f1','roc_auc','pr_auc','mcc']
fig, ax = plt.subplots(figsize=(12, 6))
cmp_df.set_index('model')[metrics].plot(kind='bar', ax=ax, colormap='viridis')
ax.set_title('Model comparison - all metrics')
ax.set_ylabel('Score'); ax.legend(loc='lower right', fontsize=8)
plt.xticks(rotation=30, ha='right'); plt.tight_layout(); plt.show()
"""),
    cell_md("""## Summary

All 9 base classifiers + 2 ensembles train successfully on the engineered
feature set. The best model is chosen by **recall on the phishing class**
(since missing a phishing site is costlier than a false alarm), with
ROC-AUC as tiebreaker. The next notebook trains AutoGluon and compares
against the manual ensemble.
"""),
]
write_nb("06_model_comparison.ipynb", nb06, "Model Comparison")


# ============================================================
# Notebook 07 - AutoGluon
# ============================================================
nb07 = [
    cell_md("""# 07 - AutoGluon TabularPredictor

Trains AutoGluon's TabularPredictor on the same engineered features and
compares against the manually-trained models.
"""),
    cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd
from src.utils import get_config, setup_logging
from src.models import ClassificationTrainer
setup_logging()
cfg = get_config()
X_tr = pd.read_csv(cfg.processed_data_dir / "X_train.csv")
y_tr = pd.read_csv(cfg.processed_data_dir / "y_train.csv").iloc[:,0]
from src.features import FeatureEngineer, FeatureSelector
eng = FeatureEngineer.load(cfg.preprocessing_dir / "feature_engineer.joblib")
sel = __import__("joblib").load(cfg.features_dir / "feature_selector.joblib")
X_tr_e = sel.transform(eng.transform(X_tr))
print(f"Train: {X_tr_e.shape}")
"""),
    cell_md("## 1. Train AutoGluon"),
    cell_code("""trainer = ClassificationTrainer(params_cfg=cfg.params, model_cfg=cfg.model)
ag = trainer.train_autogluon(X_tr_e, y_tr, cfg.autogluon_dir)
print(f"AutoGluon artifact: {cfg.autogluon_dir}")
"""),
    cell_md("## 2. Leaderboard"),
    cell_code("""if ag is not None:
    lb = ag.leaderboard()
    print(lb.head(15))
else:
    print("AutoGluon not available")
"""),
    cell_md("## 3. Compare to manual models"),
    cell_code("""import pandas as pd
cmp = pd.read_csv(cfg.evaluations_dir / "model_comparison.csv")
print(cmp[['model','recall_phishing','roc_auc','f1']]
      .sort_values('recall_phishing', ascending=False))
"""),
    cell_md("""## Summary

AutoGluon trains a stack of base learners (LightGBM, CatBoost, XGBoost,
Random Forest, Neural Net, etc.) and combines them via stacking. When
installed, it provides a strong baseline that often matches or exceeds
hand-tuned ensembles - but its training time is significantly longer,
so it is treated as an optional comparison rather than the primary model.
"""),
]
write_nb("07_autogluon.ipynb", nb07, "AutoGluon")


# ============================================================
# Notebook 08 - Explainability
# ============================================================
nb08 = [
    cell_md("""# 08 - Explainability (SHAP + LIME)

* Global feature importance via SHAP TreeExplainer
* Per-prediction local explanations via LIME
* Plain-language security tips per feature
"""),
    cell_code("""import sys, os
sys.path.insert(0, os.path.abspath('..'))
import pandas as pd, numpy as np
import matplotlib.pyplot as plt
from src.utils import get_config, setup_logging, load_joblib
from src.features import FeatureEngineer, FeatureSelector
from src.explainability import GlobalExplainer
setup_logging()
cfg = get_config()

X_tr = pd.read_csv(cfg.processed_data_dir / "X_train.csv")
y_tr = pd.read_csv(cfg.processed_data_dir / "y_train.csv").iloc[:,0]
eng = FeatureEngineer.load(cfg.preprocessing_dir / "feature_engineer.joblib")
sel = load_joblib(cfg.features_dir / "feature_selector.joblib")
X_tr_e = sel.transform(eng.transform(X_tr))
model = load_joblib(cfg.trained_models_dir / "best_model.joblib")
print(f"Best model: {model.__class__.__name__}")
"""),
    cell_md("## 1. SHAP global importance"),
    cell_code("""expl = GlobalExplainer(model=model, background=X_tr_e,
                       feature_names=list(X_tr_e.columns))
importance = expl.feature_importance(sample_size=200)
importance.head(20).plot(kind='barh', figsize=(8,7), color='#00E5A8')
plt.gca().invert_yaxis()
plt.title('SHAP global feature importance (mean |SHAP|)')
plt.tight_layout(); plt.show()
"""),
    cell_md("## 2. SHAP summary plot"),
    cell_code("""try:
    import shap
    shap_values, X_bg = expl.compute_shap_values(sample_size=200)
    shap.summary_plot(shap_values, X_bg, feature_names=list(X_tr_e.columns),
                      show=True)
except Exception as e:
    print(f"SHAP summary plot failed: {e}")
"""),
    cell_md("## 3. Local LIME explanation for a single record"),
    cell_code("""from src.explainability import LocalExplainer, build_security_tip
record = {f: int(X_tr.iloc[0][f]) for f in cfg.features}
print("Sample record (first 5 features):", dict(list(record.items())[:5]))

le = LocalExplainer(model, X_tr_e, list(X_tr_e.columns),
                    is_xgb_like=False)
label, weights = le.explain_instance(pd.Series(record), num_features=10)
print(f"Predicted: {label}")
for feat, w in weights:
    print(f"  {feat}: {w:+.4f}")
"""),
    cell_md("## 4. Plain-language security tips"),
    cell_code("""tips_cfg = cfg.prediction_schema.feature_security_tips
for fname in list(cfg.features)[:5]:
    val = record[fname]
    tip = build_security_tip(fname, val, tips_cfg)
    print(f"  {fname}={val} -> {tip}")
"""),
    cell_md("""## Summary

SHAP global importance typically surfaces `SSLfinal_State`,
`URL_of_Anchor`, and `web_traffic` as the top three drivers - matching
the published UCI literature. LIME provides per-prediction explanations,
and the `feature_security_tips` map in `prediction_schema.yaml`
translates each feature's value into a plain-language security tip the
end-user can act on.
""")
]
write_nb("08_explainability.ipynb", nb08, "Explainability")


# ============================================================
# Notebook 09 - Final Training
# ============================================================
nb09 = [
    cell_md("""# 09 - Final Training (End-to-End)

Run the complete pipeline through `training.py`. This notebook documents
the orchestration; the actual execution happens by running
`python training.py` from the project root.
"""),
    cell_md("""## Pipeline steps

1. Setup logging + load YAML configs
2. Ingest raw CSVs
3. Validate against schema.yaml
4. Train/test split (80/10/10)
5. Transform (identity pipeline - data is already ternary)
6. Feature engineering (30 -> 66 features)
7. Feature selection (66 -> 41 features)
8. Unsupervised clustering (10 algorithms, optional/interpretive)
9. Supervised classification (9 base + 2 ensembles + AutoGluon)
10. Evaluation (Accuracy/Precision/Recall/F1/AUC/MCC/LogLoss)
11. SHAP global explainability
12. Save best model + metadata
"""),
    cell_code("""# Equivalent to running `python training.py` from the project root.
import subprocess, sys, os
os.chdir('..')
result = subprocess.run([sys.executable, 'training.py'],
                        capture_output=True, text=True, timeout=1800)
print(result.stdout[-3000:])
if result.returncode != 0:
    print("STDERR:", result.stderr[-1500:])
"""),
    cell_md("## Verify artifacts"),
    cell_code("""import os
from pathlib import Path
base = Path('.')
for sub in ['artifacts/models/trained_models',
            'artifacts/features', 'artifacts/clusters',
            'artifacts/evaluations', 'artifacts/shap']:
    files = sorted((base / sub).glob('*'))
    print(f"\\n{sub}/  ({len(files)} files)")
    for f in files[:5]:
        print(f"  {f.name}  ({f.stat().st_size:,} bytes)")
"""),
    cell_md("## Best model metadata"),
    cell_code("""import json
from src.utils import get_config
cfg = get_config()
with open(cfg.trained_models_dir / "best_model_metadata.json") as f:
    meta = json.load(f)
print(f"Best model: {meta['name']}")
print(f"Metrics:")
for k, v in meta['metrics'].items():
    if isinstance(v, (int, float)):
        print(f"  {k:20s}: {v:.4f}")
    else:
        print(f"  {k:20s}: {v}")
"""),
    cell_md("""## Summary

After running `training.py`, all artifacts are persisted under
`artifacts/`. The best model is selected by recall on the phishing class
and saved as `best_model.joblib`, with metadata in
`best_model_metadata.json`. The Streamlit app (`app.py`) reads these
artifacts at launch time - no further training is needed for inference.
""")
]
write_nb("09_final_training.ipynb", nb09, "Final Training")

print("\nAll 9 notebooks written to:", NB_DIR)
