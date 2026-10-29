"""
feature_engineering.py — TF-IDF vectorization and skill co-occurrence.

Provides:
- build_tfidf_matrix : fit TF-IDF on corpus of job descriptions
- transform_user_query : convert a user skill list to the same feature space
- build_skill_cooccurrence : count how often skill pairs appear together
"""

from collections import defaultdict
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import COOCCURRENCE_THRESHOLD, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------

def build_tfidf_matrix(
    texts: list[str],
    max_features: int = TFIDF_MAX_FEATURES,
    ngram_range: tuple[int, int] = TFIDF_NGRAM_RANGE,
) -> tuple[TfidfVectorizer, csr_matrix]:
    """
    Fit a TF-IDF vectorizer on the provided corpus and return the matrix.

    Parameters
    ----------
    texts       : list of pre-cleaned job description strings
    max_features: maximum vocabulary size
    ngram_range : (min_n, max_n) for n-gram extraction

    Returns
    -------
    vectorizer  : fitted TfidfVectorizer (reusable for new inputs)
    matrix      : sparse (n_docs, n_features) TF-IDF matrix
    """
    if not texts:
        raise ValueError("Cannot build TF-IDF matrix from an empty corpus.")

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        min_df=1,           # include terms appearing in at least 1 doc
        sublinear_tf=True,  # apply log normalization to term frequencies
    )
    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def transform_user_query(
    vectorizer: TfidfVectorizer,
    user_skills: list[str],
) -> csr_matrix:
    """
    Transform the user's skill list into the same TF-IDF feature space.

    The skills are joined into a pseudo-document string and transformed
    with the already-fitted vectorizer — no re-fitting occurs.

    Parameters
    ----------
    vectorizer  : fitted TfidfVectorizer from build_tfidf_matrix
    user_skills : list of canonical skill strings

    Returns
    -------
    Sparse (1, n_features) matrix representing the user's profile.
    """
    if not user_skills:
        # Return zero vector of correct shape
        n_features = len(vectorizer.vocabulary_)
        return csr_matrix((1, n_features))

    query_text = " ".join(user_skills)
    return vectorizer.transform([query_text])


def get_feature_names(vectorizer: TfidfVectorizer) -> list[str]:
    """Return the list of feature names from the fitted vectorizer."""
    return vectorizer.get_feature_names_out().tolist()


def get_top_tfidf_terms(
    tfidf_matrix: csr_matrix,
    vectorizer: TfidfVectorizer,
    row_indices: list[int],
    top_n: int = 10,
) -> list[str]:
    """
    Find the top TF-IDF terms across a set of document rows.

    Useful for summarizing a cluster of job descriptions.

    Parameters
    ----------
    tfidf_matrix : the full TF-IDF matrix
    vectorizer   : the fitted vectorizer
    row_indices  : indices of documents to summarize
    top_n        : how many top terms to return

    Returns
    -------
    List of top term strings.
    """
    if not row_indices:
        return []

    feature_names = get_feature_names(vectorizer)
    submatrix = tfidf_matrix[row_indices, :]
    mean_scores = np.asarray(submatrix.mean(axis=0)).flatten()
    top_indices = mean_scores.argsort()[::-1][:top_n]
    return [feature_names[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Skill co-occurrence
# ---------------------------------------------------------------------------

def build_skill_cooccurrence(
    skill_lists: list[list[str]],
    threshold: int = COOCCURRENCE_THRESHOLD,
) -> dict[tuple[str, str], int]:
    """
    Count how often each pair of skills appears in the same job description.

    Parameters
    ----------
    skill_lists : one list of skills per job description
    threshold   : minimum count to include a pair in the result

    Returns
    -------
    dict mapping (skill_a, skill_b) → co-occurrence count
    (only pairs meeting the threshold are included)
    """
    counts: dict[tuple[str, str], int] = defaultdict(int)

    for skills in skill_lists:
        unique_skills = sorted(set(skills))
        for pair in combinations(unique_skills, 2):
            counts[pair] += 1

    # Filter by threshold
    return {pair: count for pair, count in counts.items() if count >= threshold}


def build_skill_frequency(
    skill_lists: list[list[str]],
) -> dict[str, int]:
    """
    Count how many job descriptions each skill appears in.

    Returns dict mapping skill → frequency count.
    """
    freq: dict[str, int] = defaultdict(int)
    for skills in skill_lists:
        for skill in set(skills):
            freq[skill] += 1
    return dict(freq)
