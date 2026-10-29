"""
pipeline.py — Shared, cached ML pipeline used by app.py and every page.

Lives in src/ (not app.py) so pages can import it without re-executing the
dashboard script, which would call st.set_page_config twice and st.stop().
"""

import streamlit as st

from src.clustering import calculate_silhouette, fit_kmeans, get_cluster_info
from src.config import N_CLUSTERS
from src.data_loader import get_dataset_stats, load_dataset
from src.feature_engineering import (
    build_skill_cooccurrence,
    build_skill_frequency,
    build_tfidf_matrix,
)
from src.preprocessing import preprocess_dataframe
from src.skill_extractor import extract_skills_from_df


@st.cache_resource(show_spinner="Loading dataset and building ML pipeline…")
def load_pipeline() -> dict:
    """
    Load data, preprocess, extract skills, fit TF-IDF and K-means.

    Cached once per server process, so page navigation never refits models.
    Returns a dict with all shared ML artifacts.
    """
    df, source = load_dataset(prefer_hf=True)

    df = preprocess_dataframe(df)
    df["extracted_skills"] = extract_skills_from_df(df)

    texts = df["clean_description"].tolist()
    vectorizer, tfidf_matrix = build_tfidf_matrix(texts)

    n_clusters = min(N_CLUSTERS, len(df) - 1)
    kmeans, labels = fit_kmeans(tfidf_matrix, n_clusters=n_clusters)
    df["cluster"] = labels

    cluster_info = get_cluster_info(kmeans, labels, tfidf_matrix, vectorizer, df)
    silhouette = calculate_silhouette(tfidf_matrix, labels)

    skill_lists = df["extracted_skills"].tolist()
    cooccurrence = build_skill_cooccurrence(skill_lists)
    skill_frequency = build_skill_frequency(skill_lists)

    return {
        "df": df,
        "source": source,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "kmeans": kmeans,
        "labels": labels,
        "cluster_info": cluster_info,
        "silhouette": silhouette,
        "cooccurrence": cooccurrence,
        "skill_frequency": skill_frequency,
        "stats": get_dataset_stats(df),
    }
