"""
clustering.py — K-means clustering of job description TF-IDF vectors.

Provides:
- fit_kmeans        : fit K-means with safety checks
- get_cluster_info  : cluster size, top terms, top skills
- calculate_silhouette : silhouette score with validity guards
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import N_CLUSTERS, RANDOM_STATE, TOP_K_TERMS
from src.feature_engineering import get_top_tfidf_terms
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# K-means fitting
# ---------------------------------------------------------------------------

def fit_kmeans(
    tfidf_matrix: csr_matrix,
    n_clusters: int = N_CLUSTERS,
    random_state: int = RANDOM_STATE,
) -> tuple[KMeans, np.ndarray]:
    """
    Fit K-means clustering on a TF-IDF matrix.

    Safety checks:
    - n_clusters must be at least 2.
    - n_clusters must be less than the number of samples.
    - Adjusts n_clusters downward if needed and logs a warning.

    Parameters
    ----------
    tfidf_matrix : sparse (n_samples, n_features) TF-IDF matrix
    n_clusters   : desired number of clusters
    random_state : for reproducibility

    Returns
    -------
    kmeans : fitted KMeans object
    labels : cluster label array of shape (n_samples,)
    """
    n_samples = tfidf_matrix.shape[0]

    if n_samples < 2:
        raise ValueError(
            f"Cannot cluster a dataset with fewer than 2 samples (got {n_samples})."
        )

    # Clamp n_clusters to valid range
    effective_k = min(n_clusters, n_samples - 1)
    effective_k = max(effective_k, 2)

    if effective_k != n_clusters:
        logger.warning(
            "Requested %d clusters but dataset has %d samples. "
            "Using %d clusters instead.",
            n_clusters, n_samples, effective_k,
        )

    kmeans = KMeans(
        n_clusters=effective_k,
        random_state=random_state,
        n_init=10,
        max_iter=300,
    )
    labels = kmeans.fit_predict(tfidf_matrix)
    logger.info(
        "K-means fitted: %d clusters, %d samples.", effective_k, n_samples
    )
    return kmeans, labels


def get_cluster_info(
    kmeans: KMeans,
    labels: np.ndarray,
    tfidf_matrix: csr_matrix,
    vectorizer: TfidfVectorizer,
    df: pd.DataFrame,
    top_n: int = TOP_K_TERMS,
) -> list[dict]:
    """
    Build a summary for each cluster.

    Returns a list of dicts (one per cluster) with:
    - cluster_id   : integer label
    - size         : number of documents in this cluster
    - top_terms    : top TF-IDF terms across cluster documents
    - top_skills   : most common skills found in cluster documents
    - categories   : distribution of career categories in this cluster
    """
    n_clusters = kmeans.n_clusters
    results = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        row_indices = np.where(mask)[0].tolist()
        cluster_df = df.iloc[row_indices]

        # Top TF-IDF terms
        top_terms = get_top_tfidf_terms(tfidf_matrix, vectorizer, row_indices, top_n)

        # Top skills (from extracted_skills column if available)
        skill_counter: dict[str, int] = {}
        if "extracted_skills" in cluster_df.columns:
            for skill_list in cluster_df["extracted_skills"]:
                if isinstance(skill_list, list):
                    for skill in skill_list:
                        skill_counter[skill] = skill_counter.get(skill, 0) + 1

        top_skills = sorted(skill_counter, key=skill_counter.get, reverse=True)[:top_n]

        # Category distribution
        category_dist = {}
        if "category" in cluster_df.columns:
            category_dist = cluster_df["category"].value_counts().to_dict()

        results.append({
            "cluster_id": int(cluster_id),
            "size": int(mask.sum()),
            "top_terms": top_terms,
            "top_skills": top_skills,
            "categories": category_dist,
        })

    return results


def calculate_silhouette(
    tfidf_matrix: csr_matrix,
    labels: np.ndarray,
) -> Optional[float]:
    """
    Calculate the silhouette score for the clustering result.

    Returns None if the configuration does not permit a valid calculation:
    - Fewer than 2 clusters
    - Any cluster with fewer than 2 samples

    The silhouette score ranges from -1 (poor) to +1 (excellent).
    Scores above 0.5 are generally considered reasonable.

    Note: For high-dimensional sparse TF-IDF matrices, silhouette scores
    tend to be lower than for dense low-dimensional data.
    """
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)

    if n_clusters < 2:
        logger.warning("Silhouette score requires at least 2 clusters.")
        return None

    # Check that every cluster has at least 2 samples
    for label in unique_labels:
        if np.sum(labels == label) < 2:
            logger.warning(
                "Cluster %d has fewer than 2 samples — silhouette cannot be computed.",
                label,
            )
            return None

    try:
        score = silhouette_score(tfidf_matrix, labels, metric="cosine", sample_size=min(1000, len(labels)))
        return float(score)
    except Exception as e:
        logger.warning("Silhouette score calculation failed: %s", e)
        return None
