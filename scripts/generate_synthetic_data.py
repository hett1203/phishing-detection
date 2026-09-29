"""
generate_synthetic_data.py
-------------------------
Synthesizes three CSVs that exactly match the UCI Phishing Websites Dataset
schema (30 ternary features + binary Result target).

The user's uploaded files were not persisted on the server. This generator
produces a statistically realistic surrogate so the entire pipeline (training,
batch prediction, Streamlit demo) runs end-to-end. When the user supplies the
real UCI CSVs they can simply drop them into artifacts/data/raw/ with the same
filenames and re-run `python training.py` -- no code changes needed.

Output:
  artifacts/data/raw/phising_08012020_120000.csv   # 11055 x 31 (train+label)
  artifacts/data/raw/phisingtest.csv                # 11055 x 30 (inference)
  artifacts/data/raw/predicted_file.csv             # 11055 x 31 (result format)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RNG = np.random.default_rng(42)

FEATURES = [
    "having_IP_Address", "URL_Length", "Shortining_Service", "having_At_Symbol",
    "double_slash_redirecting", "Prefix_Suffix", "having_Sub_Domain",
    "SSLfinal_State", "Domain_registeration_length", "Favicon", "port",
    "HTTPS_token", "Request_URL", "URL_of_Anchor", "Links_in_tags", "SFH",
    "Submitting_to_email", "Abnormal_URL", "Redirect", "on_mouseover",
    "RightClick", "popUpWidnow", "Iframe", "age_of_domain", "DNSRecord",
    "web_traffic", "Page_Rank", "Google_Index", "Links_pointing_to_page",
    "Statistical_report",
]
assert len(FEATURES) == 30

# Features that strongly drive phishing label (per UCI literature)
STRONG_PHISHING_DRIVERS = [
    "SSLfinal_State", "URL_of_Anchor", "web_traffic", "having_Sub_Domain",
    "Prefix_Suffix", "Domain_registeration_length", "Page_Rank",
    "Links_pointing_to_page", "age_of_domain", "DNSRecord",
]


def _sample_features(n: int, rng: np.random.Generator) -> np.ndarray:
    """Sample 30-feature ternary matrix with realistic marginals."""
    X = np.zeros((n, 30), dtype=np.int8)
    for j, _ in enumerate(FEATURES):
        # Each feature has a different marginal distribution
        p_pos = rng.uniform(0.25, 0.65)
        p_neg = rng.uniform(0.15, 0.45)
        # Normalize so p_pos + p_neg <= 1, remainder is 0
        total = p_pos + p_neg
        if total > 0.95:
            scale = 0.95 / total
            p_pos *= scale
            p_neg *= scale
        p_zero = 1.0 - p_pos - p_neg
        col = rng.choice([-1, 0, 1], size=n, p=[p_neg, p_zero, p_pos])
        X[:, j] = col
    return X


def _label_from_features(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Generate Result label with realistic dependence on key features.

    Target ≈ 44% phishing (-1) / 56% legitimate (1) per UCI doc.
    """
    n = X.shape[0]
    # Logit from strong drivers
    logit = np.zeros(n, dtype=np.float64)
    for fname in STRONG_PHISHING_DRIVERS:
        j = FEATURES.index(fname)
        # -1 -> push toward phishing; +1 -> push toward legit
        logit += X[:, j] * rng.uniform(0.6, 1.4)
    # Add noise
    logit += rng.normal(0.0, 1.2, size=n)
    p_phish = 1.0 / (1.0 + np.exp(logit))  # higher logit -> lower phishing prob

    # Calibrate to ~44% phishing (-1) / 56% legitimate (+1)
    target_rate = 0.44
    # Take the top (1 - target_rate)=56% by p_phish as legitimate (1),
    # and the bottom 44% as phishing (-1).
    thr = np.quantile(p_phish, target_rate)
    labels = np.where(p_phish > thr, 1, -1).astype(np.int8)

    # Inject a small amount of label noise for realism
    noise_mask = rng.random(n) < 0.03
    labels[noise_mask] *= -1
    return labels


def _map_label_to_string(y: np.ndarray) -> np.ndarray:
    """Map numeric -1/1 to the reference 'phising'/'safe' strings."""
    return np.where(y == -1, "phising", "safe")


def main(out_dir: str = "artifacts/data/raw") -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n = 11055
    rng_train = np.random.default_rng(2020)
    rng_test = np.random.default_rng(2021)

    # --- Train file ---
    X_train = _sample_features(n, rng_train)
    y_train = _label_from_features(X_train, rng_train)
    df_train = pd.DataFrame(X_train, columns=FEATURES)
    df_train["Result"] = y_train
    train_path = out / "phising_08012020_120000.csv"
    df_train.to_csv(train_path, index=False)
    print(f"[OK] Wrote {train_path}  shape={df_train.shape}  "
          f"phishing={int((y_train == -1).sum())} legit={int((y_train == 1).sum())}")

    # --- Test (unlabeled) file ---
    X_test = _sample_features(n, rng_test)
    df_test = pd.DataFrame(X_test, columns=FEATURES)
    test_path = out / "phisingtest.csv"
    df_test.to_csv(test_path, index=False)
    print(f"[OK] Wrote {test_path}  shape={df_test.shape}")

    # --- Reference predicted file (string labels) ---
    y_pred_ref = _label_from_features(X_test, rng_test)
    df_pred = pd.DataFrame(X_test, columns=FEATURES)
    df_pred["Result"] = _map_label_to_string(y_pred_ref)
    pred_path = out / "predicted_file.csv"
    df_pred.to_csv(pred_path, index=False)
    print(f"[OK] Wrote {pred_path}  shape={df_pred.shape}  "
          f"phising={int((df_pred['Result'] == 'phising').sum())} "
          f"safe={int((df_pred['Result'] == 'safe').sum())}")


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "artifacts/data/raw"
    main(out)
