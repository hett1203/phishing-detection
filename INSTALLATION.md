# 📦 Installation Guide

> PhishGuard — Phishing Website Detection System
> Complete local installation and verification guide

---

## 1. System Requirements

### Operating System
* Linux (Ubuntu 20.04+ recommended) — fully tested
* macOS 12+ (Monterey and later) — fully tested
* Windows 10/11 with WSL2 recommended (native Windows works but AutoGluon installation may require Visual Studio Build Tools)

### Python
* **Python 3.11 or 3.12** (tested on 3.12.14)
* Python 3.10 will work for the core pipeline but AutoGluon requires 3.11+
* Python 3.13 is **not yet supported** by AutoGluon

### Hardware
* **RAM**: 4 GB minimum, 8 GB recommended (AutoGluon training peaks at ~6 GB)
* **Disk**: 2 GB free for the repo + artifacts; 5 GB if AutoGluon is enabled
* **CPU**: 4+ cores recommended (XGBoost / LightGBM / CatBoost use all available cores via `n_jobs=-1`)
* **GPU**: Not required — all models are CPU-only

### Network
* Internet access during `pip install -r requirements.txt`
* No cloud credentials or API keys needed

---

## 2. Step-by-Step Installation

### Step 1 — Clone or download the project

```bash
git clone <your-repo-url>/phishing_detection.git
cd phishing_detection
```

Or download and extract the project ZIP from `/home/z/my-project/download/phishing_detection.zip`.

### Step 2 — (Recommended) Create a virtual environment

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3 — Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Expected install time**: 5–10 minutes (AutoGluon pulls in many dependencies).

> ⚠️ **If AutoGluon fails to install**: comment out the line
> `autogluon.tabular==1.1.1` in `requirements.txt` and re-run. The rest
> of the pipeline runs without it — AutoGluon training will be skipped
> with a warning in the log.

> ⚠️ **If HDBSCAN fails to install**: comment out the line
> `hdbscan==0.8.39` in `requirements.txt` and re-run. HDBSCAN clustering
> will be skipped gracefully.

### Step 4 — Place the dataset CSVs

The three reference CSVs must live in `artifacts/data/raw/`:

```
artifacts/data/raw/
├── phising_08012020_120000.csv   (11055 rows × 31 cols, with Result target)
├── phisingtest.csv                (11055 rows × 30 cols, unlabeled demo)
└── predicted_file.csv             (11055 rows × 31 cols, reference output format)
```

If you have the original UCI Phishing Websites CSVs, drop them in with these exact filenames.

If not, generate a schema-identical synthetic surrogate:

```bash
python scripts/generate_synthetic_data.py
```

This produces the three CSVs above with realistic feature marginals, a
~44% phishing / ~56% legitimate class balance, and label dependence on
the strong phishing drivers (SSLfinal_State, URL_of_Anchor, web_traffic, …).
The entire pipeline runs identically on the surrogate data.

### Step 5 — Verify the installation

```bash
# Test imports
python -c "from src.utils import get_config; cfg = get_config(); print('Config OK')"

# Test data presence
ls artifacts/data/raw/
# Expected:
# phising_08012020_120000.csv
# phisingtest.csv
# predicted_file.csv
```

### Step 6 — Run training

```bash
python training.py
```

**Expected runtime**: 4–7 minutes (excluding AutoGluon's 5-minute
time_limit). The pipeline logs each stage to console and to
`artifacts/reports/logs/app.log`.

### Step 7 — Launch the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard opens automatically in your default browser at `http://localhost:8501`. If not, navigate there manually.

---

## 3. Troubleshooting

### Problem: `ModuleNotFoundError: No module named 'box'`
**Fix**: `pip install python-box`

### Problem: `ModuleNotFoundError: No module named 'autogluon'`
**Cause**: AutoGluon failed to install (often a build-tools issue on Windows).
**Fix**: Either install AutoGluon manually (`pip install autogluon.tabular`) or leave it uninstalled — the pipeline skips stage 8b gracefully.

### Problem: `RecursionError: maximum recursion depth exceeded`
**Cause**: Joblib serialization of large sklearn trees (Birch, RandomForest).
**Fix**: Already handled in `training.py` via `sys.setrecursionlimit(20000)`. If you still see it, increase to 50000.

### Problem: `MemoryError` during clustering
**Cause**: DBSCAN / OPTICS / Spectral on the full 11,055-row dataset.
**Fix**: Already fixed — these algorithms subsample to 3,000 rows by default. To reduce further, edit `params.yaml` and decrease `clustering.dbscan.min_samples`.

### Problem: `FileNotFoundError: ... 'best_model.joblib'`
**Cause**: Training hasn't completed yet, or it failed mid-way.
**Fix**: Re-run `python training.py` and check `artifacts/reports/logs/app.log` for the failing stage.

### Problem: Streamlit shows `ImportError` on launch
**Fix**: Ensure you're running from the project root directory:
```bash
cd phishing_detection
streamlit run app.py
```

### Problem: Batch prediction shows 0 rows
**Cause**: The uploaded CSV is missing one or more of the 30 required feature columns.
**Fix**: Ensure your CSV has all 30 features as columns. Use `artifacts/data/raw/phisingtest.csv` as a schema reference.

---

## 4. Verification Checklist

After installation, verify each component works:

```bash
# 1. Configs load
python -c "from src.utils import get_config; c = get_config(); print('Configs:', list(c.config.paths.keys())[:3])"

# 2. Data ingestion works
python -c "
from src.data import DataIngestion
from src.utils import get_config
c = get_config()
ing = DataIngestion(c.raw_data_dir, c.config.files.train_data, c.config.files.test_data)
a = ing.ingest()
print(f'Ingested: {a.n_rows_train} train, {a.n_rows_test} test, {a.n_features} features')
"

# 3. Validation passes
python -c "
import pandas as pd
from src.data import DataValidation
from src.utils import get_config
c = get_config()
df = pd.read_csv(c.raw_data_dir / c.config.files.train_data)
test = pd.read_csv(c.raw_data_dir / c.config.files.test_data)
v = DataValidation(c.schema, c.features, c.target_name)
r = v.validate(df, test, c.validated_data_dir)
print(f'Valid: {r.is_valid}, Train rows: {r.n_rows_train}, Features: {r.n_features}')
"

# 4. Training artifacts exist
ls artifacts/models/trained_models/best_model.joblib
ls artifacts/evaluations/model_comparison.csv
ls artifacts/shap/global_feature_importance.json

# 5. Streamlit launches
streamlit run app.py --server.headless true --server.port 8501
# Then in another terminal:
curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8501/
# Expected: HTTP 200
```

---

## 5. Uninstall

```bash
# Deactivate the virtual env
deactivate

# Remove the project directory
cd ..
rm -rf phishing_detection
```

No system-level changes are made — everything lives inside the project directory and the virtual environment.

---

## 6. Next Steps

After installation, see:

* **[Training Guide](README.md#training-guide)** — what `training.py` does stage-by-stage
* **[Prediction Guide](README.md#prediction-guide)** — single-record + batch inference
* **[Streamlit Dashboard Guide](README.md#streamlit-dashboard-guide)** — 10-page tour
* **[Architecture](ARCHITECTURE.md)** — design rationale
* **[Workflow](WORKFLOW.md)** — data-flow diagrams
