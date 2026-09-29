# 🔬 Research Scope

> PhishGuard — Research Directions and Publication Opportunities

This document outlines research extensions suitable for academic
publication in cybersecurity / applied-ML venues (e.g., APWG eCrime,
USENIX Security, NDSS, CIKM, KDD Applied Track).

---

## 1. Benchmarking Against Published UCI Results

### Background
The original UCI Phishing Websites Dataset paper (Mohammad, Thabtah &
McCluskey, 2012) reports ~92% accuracy using a single C4.5 decision
tree on the 30 ternary features. Later papers report up to 97% with
ensemble methods on the same dataset.

### Research questions
1. **Does our pipeline match or exceed the published benchmarks?**
   Our best model (StackingEnsemble) achieves ~85% accuracy and 0.81
   recall on the phishing class. The accuracy gap from 92% → 85% may
   be due to:
   * Train/test split methodology (we use 80/10/10 stratified; the
     original uses 60/40 random)
   * Class-balance differences (we preserve the natural ~44/56 split)
   * Our use of engineered features that may have lower variance than
     the raw 30

2. **What's the accuracy ceiling on this dataset?**
   With 5-fold cross-validation across all 10 clustering algorithms and
   all 11 supervised classifiers, we can establish a tight upper bound.

### Proposed experiments
* Re-run with the original 60/40 split + accuracy as primary metric
* Compare 5-fold CV results vs. single 80/10/10 split
* Ablate the engineered features (interactions, aggregates) to isolate
  their contribution

### Expected publication
*"Benchmarking Modern ML Pipelines on the UCI Phishing Websites Dataset:
A Reproducibility Study"* — short paper for APWG eCrime 2026.

---

## 2. Feature Importance vs. Security Literature

### Background
Our SHAP analysis surfaces `SSLfinal_State`, `URL_of_Anchor`, and
`web_traffic` as the top 3 features. The original Mohammad et al. paper
also reports these as the top 3 discriminative features using
information-gain ratio.

### Research questions
1. **Do SHAP/MI rankings agree with published security-literature
   threat-model taxonomies?** (e.g., APWG eCrime annual reports,
   Anti-Phishing Working Group tactics reports)

2. **Are there features that are statistically important but operationally
   useless?** (e.g., `Favicon` may correlate with phishing but is trivial
   for an attacker to spoof)

3. **Are there security-relevant signals absent from the 30-feature set?**
   Candidate missing features:
   * X.509 certificate chain depth
   * Domain-registry country (high-risk jurisdictions)
   * URL lexical entropy
   * HTML DOM depth
   * Number of `<script>` tags with external `src`

### Proposed experiments
* Map each of the 30 features to MITRE ATT&CK techniques
* Compute feature stability across 5 different phishing datasets (UCI,
  PhishTank, OpenPhish, and 2 lab-collected)
* Run SHAP on the lab-collected datasets and compare rankings

### Expected publication
*"Feature Importance Stability Across Phishing Datasets: A SHAP-Based
Comparative Study"* — full paper for USENIX Security 2027.

---

## 3. Cluster Interpretation vs. MITRE ATT&CK

### Background
Our 10 unsupervised clustering algorithms segment websites into
behavioral clusters. KMeans (silhouette=0.036 — low, but expected on
high-dimensional ternary data) produced 5 clusters that the
`name_clusters()` function labeled as "high-risk", "suspicious",
"mixed", "low-trust", and "legitimate".

### Research questions
1. **Do the discovered clusters map to known phishing tactics?**
   For example:
   * Cluster A → "Typosquatting" (high `Prefix_Suffix`, low `Page_Rank`)
   * Cluster B → "IP-based deceptive" (`having_IP_Address` = -1)
   * Cluster C → "URL-shortener abuse" (`Shortining_Service` = -1)
   * Cluster D → "Spoofed SSL" (`SSLfinal_State` = -1 but
     `Domain_registeration_length` = 1)
   * Cluster E → "Clean established sites" (all features +1)

2. **Are cluster assignments stable across time?** A phishing campaign
   that uses IP-based URLs in January may pivot to URL-shortener abuse
   in March. Tracking cluster migration patterns is a research
   contribution.

### Proposed experiments
* Compute cluster assignments for 100k URLs collected monthly over
  12 months
* Map each cluster to MITRE ATT&CK technique IDs (TA0001 Initial Access
  in particular)
* Compute migration rates between clusters month-over-month

### Expected publication
*"Temporal Cluster Migration in Phishing Campaigns: An Unsupervised
Behavioral Analysis"* — full paper for NDSS 2027.

---

## 4. Cross-Dataset Generalization

### Background
Models trained on UCI Phishing Websites often achieve 95%+ accuracy on
the same dataset but degrade to 70% or below when tested on URLs
collected from PhishTank or OpenPhish (different time, different
labeling methodology, different feature distribution).

### Research questions
1. **How well does our model generalize to out-of-distribution URLs?**
   Specifically, what's the accuracy drop when testing on:
   * PhishTank URLs (community-reported, with timestamps)
   * OpenPhish URLs (passive DNS-based)
   * Lab-collected fresh URLs from real phishing emails

2. **Which features drift most?** (KL divergence per feature, sorted)
3. **Can we fine-tune on a small (100-URL) labeled batch from the new
   distribution to recover performance?**

### Proposed experiments
* Train on UCI as in this project (already done)
* Extract same 30 features on 5,000 PhishTank URLs (need live
  extraction module — see FUTURE_SCOPE.md)
* Evaluate model on PhishTank test set
* Compute per-feature KL divergence
* Fine-tune via transfer learning (LogisticRegression on top of frozen
  RF features)

### Expected publication
*"Generalization Gap in Phishing Detection: A Cross-Dataset Study with
Feature-Drift Analysis"* — full paper for KDD Applied Track 2027.

---

## 5. Cost-Sensitive Learning for Operational Deployment

### Background
In production, false negatives (missing a phishing site) cost 100x more
than false positives (blocking a legitimate site). Our pipeline already
selects the best model by recall on the phishing class, but we don't
explicitly model asymmetric costs.

### Research questions
1. **What's the optimal decision threshold for production?** Default is
   0.5, but a threshold of 0.3 may yield higher recall at acceptable FPR.
2. **Can cost-sensitive learning (weighted loss) outperform
   threshold-tuning?**
3. **What's the dollar cost of a single false negative?** (industry data:
   average phishing incident cost is $4.2M per IBM Cost of a Data Breach
   Report 2024)

### Proposed experiments
* Sweep threshold from 0.1 → 0.9 in 0.05 steps; plot ROC + cost curve
* Train CatBoost with `class_weights = {-1: 100, 1: 1}`; compare to
  default + threshold-tuning
* Build a cost-sensitivity dashboard showing expected $ loss per
  threshold choice

### Expected publication
*"Cost-Sensitive Phishing Detection: Threshold Optimization and
Class-Weighting in Production"* — short paper for APWG eCrime 2026.

---

## 6. Federated Learning for Enterprise Deployment

### Background
Large enterprises (banks, governments) cannot share their internal
phishing data with a central model trainer due to data-residency
regulations (GDPR, CCPA, banking secrecy laws).

### Research questions
1. **Can federated learning match centralized learning on this dataset?**
2. **What's the privacy budget (ε) required for differential privacy to
   not degrade recall by more than 2%?**
3. **Can the 30-feature UCI schema be the common vocabulary across
   enterprises?** (vs. each enterprise having its own feature set)

### Proposed experiments
* Simulate 5 "enterprise" clients, each holding 20% of the dataset
* Run 50 rounds of FedAvg with XGBoost (using `xgboost.federated`)
* Compare to centralized training
* Add DP-SGD with ε = {1, 5, 10, 20}; measure recall drop

### Expected publication
*"Federated Phishing Detection: Privacy-Preserving Model Training
Across Enterprises"* — full paper for USENIX Security 2027.

---

## 7. Causal Inference on Phishing Features

### Background
SHAP and LIME identify *correlative* feature importance, not *causal*.
A feature like `URL_Length` may correlate with phishing because
attackers use long URLs — but does shortening the URL cause the model
to classify it as legitimate? No, that's adversarial.

### Research questions
1. **Which features have a causal effect on the model's prediction?**
2. **Can we identify "operationally useful" features (causal + easy to
   measure) vs. "spurious" features (correlative only)?**

### Proposed experiments
* Use DoWhy / causalml libraries to compute average treatment effect
  per feature
* Build a causal DAG over the 30 features
* Compare SHAP ranking vs. causal-effect ranking

### Expected publication
*"From Correlation to Causation: Identifying Operationally Useful
Features in Phishing Detection"* — full paper for CIKM 2027.

---

## 8. Robustness to Adversarial Perturbations

### Background
A phishing attacker who knows the model + features can craft a URL that
maximizes P(legitimate). For example, if `URL_Length` is the top
feature, the attacker can keep the URL short.

### Research questions
1. **What's the smallest perturbation that flips a phishing URL's
   prediction from "phishing" to "safe"?**
2. **Which features are most vulnerable to perturbation?**
3. **Does adversarial training help?**

### Proposed experiments
* Implement a Carlini-Wagner-style attack on the 30-feature input
  (constrained to {-1, 0, 1})
* Compute adversarial accuracy on 1,000 phishing URLs
* Train with adversarial samples; measure robustness gain

### Expected publication
*"Adversarial Robustness of Phishing Classifiers: An Attack-and-Defense
Study"* — full paper for NDSS 2027.

---

## Research Roadmap

| Year | Conference | Topic | Effort | Status |
|------|-----------|-------|--------|--------|
| 2026 Q2 | APWG eCrime | UCI benchmarking reproducibility | Low | Planned |
| 2026 Q3 | APWG eCrime | Cost-sensitive threshold optimization | Low | Planned |
| 2026 Q4 | KDD Applied | Cross-dataset generalization | Medium | Planned |
| 2027 Q1 | CIKM | Causal feature importance | Medium | Planned |
| 2027 Q2 | USENIX Sec | Feature stability across datasets | High | Planned |
| 2027 Q3 | NDSS | Cluster migration in phishing | High | Planned |
| 2027 Q4 | USENIX Sec | Federated learning | Very High | Future |
| 2028 Q1 | NDSS | Adversarial robustness | High | Future |

---

## Collaboration

This research agenda is open to academic collaborators. Contact the
project author via the repository issues page. Each project is designed
to be a self-contained 6-month master's or Ph.D. rotation project with
clearly-scoped deliverables and publication targets.
