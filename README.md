# 🛡️ PhishGuard — Phishing URL Detection System

A complete, end-to-end **machine-learning system** that automatically distinguishes phishing websites from legitimate ones using pre-engineered lexical, host-based, and content-based URL features.

Built as a **Master's-level Data Science project** with industry-grade ML engineering practices: modular design, config-driven pipelines, model versioning, SHAP explainability, and a polished dark-cybersecurity Streamlit dashboard.

---

## 📊 Performance

The best model (**HistGradientBoosting**) was selected after training 10 different classifiers and comparing on 7 evaluation metrics.

| Metric                | Score  |
|-----------------------|--------|
| Accuracy              | **97.91%** |
| Phishing Recall       | **96.66%** |
| F1 Score (phishing)   | 92.93% |
| ROC-AUC               | 0.9967 |
| PR-AUC                | 0.9861 |
| MCC                   | 0.918  |
| Log Loss              | 0.060  |

**Selection criterion:** phishing-recall priority — a missed phishing site is far costlier than a false alarm.

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (optional — pre-trained artifacts are included)
python src/train.py

# 3. Launch the dashboard
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 🧠 Models Compared

| Model                  | Accuracy | Phishing Recall | F1     | ROC-AUC |
|------------------------|----------|-----------------|--------|---------|
| **HistGradientBoosting** ⭐ | 97.91%   | 96.66%          | 92.93% | 0.9967  |
| LightGBM               | 98.13%   | 96.47%          | 93.61% | 0.9967  |
| CatBoost               | 98.07%   | 95.99%          | 93.42% | 0.9965  |
| XGBoost                | 98.27%   | 95.84%          | 94.04% | 0.9967  |
| ExtraTrees             | 97.35%   | 95.63%          | 91.12% | 0.9949  |
| LogisticRegression     | 95.08%   | 95.33%          | 84.64% | 0.9886  |
| RandomForest           | 98.29%   | 94.06%          | 94.01% | 0.9960  |
| Bagging                | 98.34%   | 92.98%          | 94.09% | 0.9888  |
| GradientBoosting       | 97.65%   | 87.50%          | 91.39% | 0.9932  |
| AdaBoost               | 96.97%   | 84.78%          | 88.84% | 0.9907  |

---

## 🗂️ Project Structure

```
phishing_app/
├── app.py                      # Streamlit entry point
├── requirements.txt
├── README.md
├── src/
│   ├── features.py             # URL → 22 numeric features extractor
│   ├── preprocess.py           # TLD top-50 bucketing + preprocessing
│   ├── train.py                # Train all models, save best + metrics + SHAP
│   ├── predict.py              # Inference helpers (cached model loading)
│   ├── theme.py                # Global dark cybersecurity CSS
│   └── pages/
│       ├── home.py             # Landing page with KPI strip
│       ├── dataset.py          # Dataset Overview (class balance, distributions)
│       ├── eda.py               # Correlation heatmap + discriminative features
│       ├── single_checker.py   # Single URL checker with risk gauge
│       ├── batch.py             # Batch CSV prediction + download
│       ├── explainability.py   # SHAP global summary + beeswarm
│       ├── model_performance.py # ROC curves + confusion matrices + bar charts
│       └── about.py             # Project overview + methodology + references
└── artifacts/
    ├── models/                 # Preprocessor + best model + all models (joblib)
    ├── metrics/                # Comparison CSV, ROC curves, confusion matrices, best meta
    └── shap/                   # SHAP values + sample + explainer
```

---

## 🎨 Dashboard Pages

1. **🏠 Home** — Hero + KPI strip + pipeline overview + quick-start cards.
2. **📊 Dataset Overview** — Class balance (bar + donut), per-feature distribution by class, top-20 TLDs, sample preview.
3. **🔍 EDA Dashboard** — Full correlation heatmap, top-15 features by |ρ|, class-wise mean comparison, numeric statistics table.
4. **🧪 Single URL Checker** — Paste a URL → instant verdict + probability + risk gauge + per-feature plain-language explanation.
5. **📂 Batch Prediction** — Upload CSV → download CSV with `label`, `probability`, `risk_score` per URL.
6. **💡 Explainability** — SHAP global summary, top-10 beeswarm, top-5 contributing features with interpretation.
7. **🏆 Model Performance** — Side-by-side comparison table, overlaid ROC curves, side-by-side confusion matrices, metric bar charts.
8. **ℹ️ About** — Mission, technology stack, methodology, best model summary, limitations, references.

---

## 🎨 Theme

Dark cybersecurity palette:
- Background: deep blue-black (`#0a0e14`)
- Cards: gradient panels with subtle cyan border
- Phishing: neon red (`#ff4757`)
- Safe: neon green (`#2ed573`)
- Info / charts: neon cyan (`#00d4ff`)
- Animations: pulse-glow on phishing alerts, fade-in on cards

---

## 📋 Requirements

See `requirements.txt`. Key dependencies:

- `streamlit >= 1.64`
- `scikit-learn >= 1.5`
- `xgboost >= 2.1`
- `lightgbm >= 4.5`
- `catboost >= 1.2`
- `shap >= 0.52`
- `plotly >= 6.5`
- `pandas`, `numpy`, `joblib`

---

## 📝 Methodology

### Data Ingestion
116,586 labeled URLs (after dropping 14 NaN rows) loaded from `Dataset.csv`. Features include URL length, domain length, character counts, ratios, entropy, and the top-level domain (TLD).

### Preprocessing
- Top-50 most-frequent TLDs → one-hot encoded as their own column.
- Rare TLDs → collapsed into `tld_other` bucket.
- Numeric features → passed through untouched.
- Fitted preprocessor saved as `artifacts/models/preprocessor.joblib`.

### Train/Test Split
Stratified 80/20 split preserving the ~14% phishing class ratio. **93,268** training rows · **23,318** test rows.

### Model Training
10 classifiers trained with class-imbalance handling:
- `class_weight="balanced"` for sklearn models
- `scale_pos_weight ≈ 6.0` for XGBoost
- `auto_class_weights="Balanced"` for CatBoost

### Evaluation
Each model scored on:
- Accuracy
- Precision (phishing class)
- **Recall (phishing class)** ← primary criterion
- F1 (phishing class)
- ROC-AUC
- PR-AUC
- MCC
- Log Loss
- Confusion Matrix (TP, FP, TN, FN)

### Explainability
SHAP `TreeExplainer` computed on a 2,000-sample held-out subset. Mean |SHAP| per feature surfaces the global drivers of the model's decisions.

---

## 🔮 Future Scope

- **Live URL feature extraction** via HTTP fetch + WHOIS + DNS lookups
- **Browser extension** deployment for real-time browsing protection
- **Real-time threat-intel feed** integration
- **Cost-sensitive threshold tuning** for production false-negative rate targets
- **Periodic retraining** pipeline with newly labeled URLs

---

## 📚 References

- Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12.
- Chen & Guestrin (2016). *XGBoost: A Scalable Tree Boosting System.* KDD.
- Ke et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS.
- Prokhorenkova et al. (2018). *CatBoost: unbiased boosting with categorical features.* NeurIPS.
- Lundberg & Lee (2017). *A Unified Approach to Interpreting Model Predictions.* NeurIPS.

---

## 📄 License

MIT License — see `LICENSE` for details.
