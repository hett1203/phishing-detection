# 🚀 Future Scope

> PhishGuard — Roadmap and Future Enhancements

This document captures planned future work that extends PhishGuard from a
static-classifier system to a real-time, production-grade phishing
detection platform.

---

## 1. Live URL Feature Extraction

### Current state
The system consumes the 30 pre-engineered ternary features directly from
the UCI dataset. No raw URL strings, HTML content, or WHOIS records
flow through the pipeline. This is sufficient for academic evaluation
but blocks real-time URL inspection.

### Proposed work
Build a `src/extraction/` module that takes a raw URL string and
produces the 30-feature vector expected by the model:

```
raw URL  →  LexicalFeatureExtractor   →  having_IP_Address, URL_Length, ...
        →  HostFeatureExtractor       →  Domain_registeration_length, age_of_domain, DNSRecord, ...
        →  ContentFeatureExtractor   →  SSLfinal_State, URL_of_Anchor, Links_in_tags, ...
        →  ternary encoding (←{bad, neutral, good} ← {-1, 0, 1})
```

### Implementation stack
* **Lexical**: `urllib.parse`, `tldextract`, `re`
* **Host-based (WHOIS)**: `python-whois`, `dnspython`
* **Content-based (HTML/JS)**: `requests`, `beautifulsoup4`, `selenium` (for JS-rendered pages)
* **SSL**: `ssl` + `socket` (extract issuer, expiry, chain)
* **Threat-intel lookups**: `PhishTank API`, `Google Safe Browsing API`

### Risks
* Live extraction adds 1–5 seconds per URL (WHOIS latency dominates)
* Some features (`web_traffic`, `Page_Rank`, `Google_Index`) require
  third-party APIs that may need API keys or rate-limit handling
* HTML scraping has reliability concerns (sites may block, return 403,
  or use JS-rendered content)

---

## 2. Browser Extension Deployment

### Vision
Package the trained model + a lightweight feature extractor as a
Chrome/Firefox extension that flags phishing sites in real-time as the
user types a URL or clicks a link. UI:

* Green/red icon next to the address bar
* Hover-card showing top 3 contributing features (via SHAP local)
* "Block this site" button for user feedback

### Architecture
```
┌──────────────────────────────────────────┐
│   Browser Extension (TypeScript + React) │
│  • URL observer                          │
│  • Calls local FastAPI backend           │
│  • Shows verdict in popup                │
└────────────────┬─────────────────────────┘
                 │ HTTP /predict
                 ▼
┌──────────────────────────────────────────┐
│  FastAPI microservice (Python)           │
│  • Loads best_model.joblib once         │
│  • Receives URL → extracts 30 features  │
│  • Returns {label, confidence, shap_top}│
└──────────────────────────────────────────┘
```

### Why not run the model in the browser?
* ONNX runtime in JavaScript is feasible but adds complexity
* Most feature extractors (WHOIS, SSL, DNS) need server-side access
* A small backend keeps the model updateable without redeploying the
  extension

---

## 3. Real-Time Threat-Intel Feed Integration

### Feeds to integrate
* **PhishTank** — community-reported phishing URLs with timestamps
* **OpenPhish** — passive DNS-based phishing feed
* **URLVoid** — aggregate reputation across 30+ blocklists
* **Google Safe Browsing API** — official Google threat intel
* **VirusTotal** — multi-engine URL scanning

### Use cases
1. **Daily retraining trigger** — when feed volume spikes, trigger an
   Airflow retraining job
2. **Cluster label enrichment** — when a URL in cluster N is confirmed
   phishing by PhishTank, propagate the label to all URLs in the cluster
3. **Concept-drift detection** — monitor KL divergence between current
   feature distribution and the training distribution; alert if >0.05

---

## 4. Active Learning Loop

### Motivation
The current model misses ~20% of phishing sites (recall ≈ 0.80). Many
of those misses are likely on novel phishing tactics the model has never
seen during training. An active-learning loop surfaces these low-
confidence predictions for analyst review.

### Workflow
```
   new URL  →  pipeline  →  P(phishing) = 0.48 (low-confidence)
                                  │
                                  ▼
                       analyst queue
                                  │
                                  ▼
                       analyst labels it phishing
                                  │
                                  ▼
                       add to training set
                                  │
                                  ▼
                       weekly retrain on labeled batch
```

### Implementation
* PostgreSQL table `analyst_queue` with (url, prediction, confidence,
  analyst_label, reviewed_at)
* Airflow DAG runs `training.py` every Sunday 02:00 on the augmented
  training set
* MLflow tracks model versions + auto-rolls-back if new model
  underperforms previous on the held-out test set

---

## 5. Concept-Drift Monitoring

### Why it matters
Phishing tactics evolve. New phishing kits introduce new URL patterns,
new SSL-issuance abuse, new subdomain structures. The model trained
today will degrade over months.

### Detection
* **Feature distribution drift**: KL divergence per feature, week-over-week
* **Prediction distribution drift**: histogram of P(phishing) shifting
* **Performance drift**: rolling recall_phishing on confirmed-labeled
  subset

### Alerting
* PagerDuty alert if drift > threshold for 2 consecutive weeks
* Auto-trigger retraining when drift confirmed

---

## 6. MLOps Deployment

### Current state
Local execution only — `python training.py` + `streamlit run app.py`.

### Production deployment
* **FastAPI microservice** wrapping `PredictionPipeline` with /predict
  and /predict_batch endpoints
* **Docker + Kubernetes** for horizontal scaling (5–20 replicas based
  on QPS)
* **MLflow** for experiment tracking + model registry
* **Airflow** for scheduled retraining + drift monitoring
* **Prometheus + Grafana** for inference latency / error-rate / drift
  dashboards
* **S3** for model artifact storage (replacing local `artifacts/`)

### Suggested Dockerfile
```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python training.py
EXPOSE 8000
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 7. Multi-Modal Detection

### Vision
Combine URL features with:
* **Screenshot features** — visual appearance of the page (ResNet
  embeddings)
* **DOM structure features** — DOM tree hash, script count, form count
* **Behavioral features** — redirect chain length, JS-execution traces
* **Certificate features** — full X.509 chain analysis (issuer pattern,
  SAN count, key algo)

### Architecture
* URL feature extractor (this project) → 41-dim feature vector
* Screenshot encoder (ResNet-50 fine-tuned) → 256-dim embedding
* DOM encoder (TreeLSTM) → 128-dim embedding
* Late-fusion MLP combines all modalities → phishing probability

---

## 8. Cross-Dataset Generalization Study

### Question
Does the model trained on the UCI Phishing Websites Dataset generalize
to URLs collected from PhishTank (a different time window and labeling
methodology)?

### Proposed experiment
1. Train on UCI (this project) — already done
2. Collect 5,000 URLs from PhishTank (labeled phishing) + 5,000 from
   Alexa Top 1M (labeled legitimate)
3. Extract the same 30 features using the proposed live-extraction
   module
4. Evaluate model on this out-of-distribution test set
5. Report performance gap + identify features that drift most

---

## 9. Adversarial Robustness

### Threat model
Adversary who knows the model + features and crafts a URL that maximizes
P(legitimate) while remaining functionally phishing.

### Defenses
* Adversarial training: generate perturbed samples and retrain
* Robust features: replace `URL_Length` (easy to game) with `URL_Entropy`
* Ensemble diversification: train 5 models on bootstrap samples; require
  consensus (3/5) for phishing verdict

---

## 10. Privacy-Preserving Detection

### Use case
Enterprise deployment where URLs cannot leave the corporate network
(due to data-residency regulations).

### Approach
* Run feature extraction + inference on-prem
* Optionally: federated learning across enterprise deployments
  (each trains locally, shares only model updates)
* Differential privacy: add Gaussian noise to model gradients
  (DP-SGD) with ε ≤ 1.0

---

## Prioritization

| # | Feature | Effort | Impact | Priority |
|---|---------|--------|--------|----------|
| 1 | Live URL feature extraction | Medium | High | P0 |
| 2 | FastAPI microservice | Low | High | P0 |
| 3 | Real-time threat-intel feed | Medium | High | P1 |
| 4 | Concept-drift monitoring | Medium | High | P1 |
| 5 | Active learning loop | High | Medium | P2 |
| 6 | Browser extension | High | High | P2 |
| 7 | MLOps (K8s + MLflow) | High | Medium | P2 |
| 8 | Multi-modal detection | Very High | High | P3 |
| 9 | Adversarial robustness | High | Medium | P3 |
| 10 | Privacy-preserving | High | Low | P3 |
