"""
src/clustering/trainer.py
Unsupervised website-behavior clustering (OPTIONAL - interpretability only).
Compares 10 clustering algorithms; selects the best by silhouette score.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import (
    AgglomerativeClustering, Birch, DBSCAN, KMeans, MeanShift,
    MiniBatchKMeans, OPTICS, SpectralClustering, estimate_bandwidth,
)
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
)
from sklearn.preprocessing import StandardScaler

from src.utils import (
    ClusteringError, get_logger, save_joblib, to_abs, write_json,
)

log = get_logger(__name__)


@dataclass
class ClusterResult:
    name: str
    labels: np.ndarray
    n_clusters: int
    silhouette: Optional[float]
    davies_bouldin: Optional[float]
    calinski_harabasz: Optional[float]
    is_valid: bool = True
    note: str = ""


@dataclass
class ClusterReport:
    results: List[ClusterResult] = field(default_factory=list)
    best_algorithm: str = ""
    best_silhouette: float = -1.0

    def to_dict(self) -> Dict:
        return {
            "results": [
                {
                    "name": r.name,
                    "n_clusters": r.n_clusters,
                    "silhouette": r.silhouette,
                    "davies_bouldin": r.davies_bouldin,
                    "calinski_harabasz": r.calinski_harabasz,
                    "is_valid": r.is_valid,
                    "note": r.note,
                }
                for r in self.results
            ],
            "best_algorithm": self.best_algorithm,
            "best_silhouette": self.best_silhouette,
        }


class ClusterTrainer:
    """Trains all 10 clustering algorithms and selects the best by silhouette."""

    def __init__(self, params_cfg) -> None:
        self.params = params_cfg.clustering
        self.random_state = 42

    def _safe_metrics(self, X: np.ndarray, labels: np.ndarray
                      ) -> tuple[Optional[float], Optional[float],
                                 Optional[float]]:
        """Compute silhouette/DB/CH safely. Returns None if undefined."""
        n_clusters = len(set(labels[labels >= 0])) if len(labels) else 0
        n_samples = X.shape[0]
        if n_clusters < 2 or n_clusters >= n_samples:
            return None, None, None
        # Subsample for speed on silhouette (>=10k rows)
        max_sil_samples = min(2000, n_samples)
        try:
            sil = float(silhouette_score(
                X, labels, sample_size=max_sil_samples,
                random_state=self.random_state))
        except Exception:
            sil = None
        try:
            db = float(davies_bouldin_score(X, labels))
        except Exception:
            db = None
        try:
            ch = float(calinski_harabasz_score(X, labels))
        except Exception:
            ch = None
        return sil, db, ch

    def _scale(self, X: pd.DataFrame) -> np.ndarray:
        # Standardize for distance-based methods
        return StandardScaler().fit_transform(X)

    def train_all(self, X: pd.DataFrame, out_dir: str | Path
                  ) -> tuple[ClusterReport, Dict[str, np.ndarray]]:
        try:
            out = to_abs(out_dir)
            out.mkdir(parents=True, exist_ok=True)

            Xs = self._scale(X)
            # Cap working size for the slow O(n^2) methods (DBSCAN, OPTICS,
            # Spectral, MeanShift, Agglomerative). 3000 keeps runtime < 30s.
            n = Xs.shape[0]
            work_n = min(n, 3000)
            rng = np.random.default_rng(self.random_state)
            work_idx = rng.choice(n, work_n, replace=False) \
                if n > work_n else np.arange(n)
            Xs_work = Xs[work_idx]

            report = ClusterReport()
            labels_per_algo: Dict[str, np.ndarray] = {}

            algos = list(self.params.algorithms)
            log.info("Training %d clustering algorithms...", len(algos))

            # KMeans
            if "KMeans" in algos:
                k = self.params.kmeans.n_clusters
                m = KMeans(n_clusters=k, random_state=self.random_state,
                           n_init=10)
                labels = m.fit_predict(Xs)
                sil, db, ch = self._safe_metrics(Xs, labels)
                save_joblib(m, out / "kmeans.joblib")
                labels_per_algo["KMeans"] = labels
                report.results.append(ClusterResult(
                    "KMeans", labels, k, sil, db, ch))

            # MiniBatchKMeans
            if "MiniBatchKMeans" in algos:
                k = self.params.minibatch_kmeans.n_clusters
                m = MiniBatchKMeans(
                    n_clusters=k, random_state=self.random_state,
                    batch_size=int(self.params.minibatch_kmeans.batch_size),
                    n_init=10)
                labels = m.fit_predict(Xs)
                sil, db, ch = self._safe_metrics(Xs, labels)
                save_joblib(m, out / "minibatch_kmeans.joblib")
                labels_per_algo["MiniBatchKMeans"] = labels
                report.results.append(ClusterResult(
                    "MiniBatchKMeans", labels, k, sil, db, ch))

            # Gaussian Mixture
            if "GaussianMixture" in algos:
                k = self.params.gmm.n_components
                m = GaussianMixture(
                    n_components=k, random_state=self.random_state,
                    covariance_type=self.params.gmm.covariance_type)
                labels = m.fit_predict(Xs)
                sil, db, ch = self._safe_metrics(Xs, labels)
                save_joblib(m, out / "gmm.joblib")
                labels_per_algo["GaussianMixture"] = labels
                report.results.append(ClusterResult(
                    "GaussianMixture", labels, k, sil, db, ch))

            # Agglomerative
            if "AgglomerativeClustering" in algos:
                k = self.params.agglomerative.n_clusters
                m = AgglomerativeClustering(
                    n_clusters=k, linkage=self.params.agglomerative.linkage)
                labels = m.fit_predict(Xs_work)
                # Expand to full size (rest unlabeled -> -1)
                full_labels = np.full(n, -1, dtype=np.int32)
                full_labels[work_idx] = labels
                sil, db, ch = self._safe_metrics(Xs_work, labels)
                save_joblib(m, out / "agglomerative.joblib")
                labels_per_algo["AgglomerativeClustering"] = full_labels
                report.results.append(ClusterResult(
                    "AgglomerativeClustering", full_labels, k, sil, db, ch,
                    note=f"trained on subsample of {work_n} rows"))

            # Birch
            if "Birch" in algos:
                k = self.params.birch.n_clusters
                m = Birch(n_clusters=k,
                          threshold=float(self.params.birch.threshold))
                labels = m.fit_predict(Xs)
                sil, db, ch = self._safe_metrics(Xs, labels)
                save_joblib(m, out / "birch.joblib")
                labels_per_algo["Birch"] = labels
                report.results.append(ClusterResult(
                    "Birch", labels, k, sil, db, ch))

            # DBSCAN (subsampled for speed - O(n^2))
            if "DBSCAN" in algos:
                try:
                    m = DBSCAN(eps=float(self.params.dbscan.eps),
                               min_samples=int(self.params.dbscan.min_samples),
                               n_jobs=-1)
                    labels_sub = m.fit_predict(Xs_work)
                    full_labels = np.full(n, -1, dtype=np.int32)
                    full_labels[work_idx] = labels_sub
                    k = len(set(labels_sub[labels_sub >= 0]))
                    sil, db, ch = self._safe_metrics(Xs_work, labels_sub)
                    save_joblib(m, out / "dbscan.joblib")
                    labels_per_algo["DBSCAN"] = full_labels
                    report.results.append(ClusterResult(
                        "DBSCAN", full_labels, k, sil, db, ch,
                        note=(f"trained on subsample of {work_n} rows; "
                              f"{int((labels_sub == -1).sum())} noise")))
                except Exception as exc:
                    log.warning("DBSCAN failed: %s", exc)

            # HDBSCAN (optional - if available)
            if "HDBSCAN" in algos:
                try:
                    import hdbscan  # type: ignore
                    m = hdbscan.HDBSCAN(
                        min_cluster_size=int(self.params.hdbscan.min_cluster_size),
                        min_samples=int(self.params.hdbscan.min_samples))
                    labels = m.fit_predict(Xs)
                    k = len(set(labels[labels >= 0]))
                    sil, db, ch = self._safe_metrics(Xs, labels)
                    save_joblib(m, out / "hdbscan.joblib")
                    labels_per_algo["HDBSCAN"] = labels
                    report.results.append(ClusterResult(
                        "HDBSCAN", labels, k, sil, db, ch,
                        note=(f"{int((labels == -1).sum())} noise points")))
                except ImportError:
                    log.warning("hdbscan not installed; skipping HDBSCAN")

            # OPTICS (subsampled for speed - O(n^2) worst case)
            if "OPTICS" in algos:
                try:
                    # Use a smaller subsample (2000) - OPTICS is expensive
                    opt_n = min(work_n, 2000)
                    opt_idx = work_idx[:opt_n]
                    Xs_opt = Xs[opt_idx]
                    m = OPTICS(min_samples=int(self.params.optics.min_samples),
                               xi=float(self.params.optics.xi),
                               min_cluster_size=float(
                                   self.params.optics.min_cluster_size),
                               n_jobs=-1)
                    labels_sub = m.fit_predict(Xs_opt)
                    full_labels = np.full(n, -1, dtype=np.int32)
                    full_labels[opt_idx] = labels_sub
                    k = len(set(labels_sub[labels_sub >= 0]))
                    sil, db, ch = self._safe_metrics(Xs_opt, labels_sub)
                    save_joblib(m, out / "optics.joblib")
                    labels_per_algo["OPTICS"] = full_labels
                    report.results.append(ClusterResult(
                        "OPTICS", full_labels, k, sil, db, ch,
                        note=(f"trained on subsample of {opt_n} rows; "
                              f"{int((labels_sub == -1).sum())} noise")))
                except Exception as exc:
                    log.warning("OPTICS failed: %s", exc)

            # MeanShift
            if "MeanShift" in algos:
                bw = self.params.meanshift.bandwidth
                if bw is None:
                    bw = estimate_bandwidth(Xs_work, quantile=0.2)
                    log.info("MeanShift auto-bandwidth=%.3f", bw)
                m = MeanShift(bandwidth=float(bw), bin_seeding=True,
                              n_jobs=-1)
                labels = m.fit_predict(Xs_work)
                full_labels = np.full(n, -1, dtype=np.int32)
                full_labels[work_idx] = labels
                k = len(set(labels[labels >= 0]))
                sil, db, ch = self._safe_metrics(Xs_work, labels)
                save_joblib(m, out / "meanshift.joblib")
                labels_per_algo["MeanShift"] = full_labels
                report.results.append(ClusterResult(
                    "MeanShift", full_labels, k, sil, db, ch,
                    note=f"trained on subsample of {work_n} rows"))

            # Spectral Clustering
            if "SpectralClustering" in algos:
                k = self.params.spectral.n_clusters
                m = SpectralClustering(
                    n_clusters=k, random_state=self.random_state,
                    affinity=self.params.spectral.affinity,
                    n_neighbors=10, n_jobs=-1)
                labels = m.fit_predict(Xs_work)
                full_labels = np.full(n, -1, dtype=np.int32)
                full_labels[work_idx] = labels
                sil, db, ch = self._safe_metrics(Xs_work, labels)
                save_joblib(m, out / "spectral.joblib")
                labels_per_algo["SpectralClustering"] = full_labels
                report.results.append(ClusterResult(
                    "SpectralClustering", full_labels, k, sil, db, ch,
                    note=f"trained on subsample of {work_n} rows"))

            # ---- Best by silhouette ----
            valid = [r for r in report.results
                     if r.silhouette is not None and r.silhouette > 0]
            if valid:
                best = max(valid, key=lambda r: r.silhouette)
                report.best_algorithm = best.name
                report.best_silhouette = best.silhouette
            write_json(out / "cluster_report.json", report.to_dict())

            log.info("Clustering complete. Best=%s (silhouette=%.3f)",
                     report.best_algorithm, report.best_silhouette)
            return report, labels_per_algo
        except Exception as exc:
            raise ClusteringError(f"Clustering failed: {exc}") from exc


def name_clusters(report: ClusterReport, X: pd.DataFrame,
                  feature_names: List[str]) -> Dict[int, str]:
    """Best-effort semantic naming of clusters using per-cluster feature means."""
    if not report.results:
        return {}
    best_res = next((r for r in report.results
                     if r.name == report.best_algorithm), report.results[0])
    if best_res is None:
        return {}

    labels = best_res.labels
    if labels is None or len(labels) != len(X):
        return {}

    df = pd.DataFrame(X).copy()
    df["cluster"] = labels
    df = df[df["cluster"] >= 0]
    if df.empty:
        return {}

    means = df.groupby("cluster").mean()
    # Use known strong phishing drivers to interpret each cluster
    drivers_present = [c for c in [
        "SSLfinal_State", "URL_of_Anchor", "web_traffic",
        "having_Sub_Domain", "Prefix_Suffix", "Domain_registeration_length",
        "Page_Rank", "Links_pointing_to_page",
    ] if c in means.columns]

    names: Dict[int, str] = {}
    for cid, row in means.iterrows():
        avg_score = row[drivers_present].mean() if drivers_present else 0
        if avg_score < -0.4:
            names[cid] = "High-risk phishing cluster"
        elif avg_score < 0:
            names[cid] = "Suspicious sites cluster"
        elif avg_score < 0.4:
            names[cid] = "Mixed / low-trust cluster"
        else:
            names[cid] = "Legitimate established sites"
    return names
