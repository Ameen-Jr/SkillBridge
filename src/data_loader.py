"""
data_loader.py — Dataset loading, validation, and optional HuggingFace download.

Responsibilities:
- Load jobs.csv (HuggingFace) or sample_jobs.csv (synthetic fallback).
- Validate required columns.
- Remove duplicate records.
- Handle missing values safely.
- Attempt auto-download from HuggingFace Parquet URL.
"""

import io
import logging
from pathlib import Path

import pandas as pd
import requests

from src.config import (
    HF_PARQUET_URL,
    JOBS_CSV,
    REQUIRED_COLUMNS,
    SAMPLE_CSV,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(prefer_hf: bool = True) -> tuple[pd.DataFrame, str]:
    """
    Load the job dataset.

    Tries in order:
    1. data/jobs.csv  (HuggingFace download, if prefer_hf=True and file exists)
    2. Auto-download from HuggingFace (if prefer_hf=True)
    3. data/sample_jobs.csv (synthetic fallback)

    Returns
    -------
    df : pd.DataFrame  — cleaned and validated DataFrame
    source : str       — human-readable description of the source used
    """
    if prefer_hf:
        # Try existing cached download
        if JOBS_CSV.exists():
            try:
                df = _load_and_clean(JOBS_CSV)
                return df, f"HuggingFace dataset ({JOBS_CSV.name}, {len(df)} records)"
            except Exception as e:
                logger.warning("Cached jobs.csv failed to load: %s", e)

        # Try live download
        logger.info("Attempting to download HuggingFace dataset …")
        success = download_hf_dataset()
        if success and JOBS_CSV.exists():
            try:
                df = _load_and_clean(JOBS_CSV)
                return df, f"HuggingFace dataset (downloaded, {len(df)} records)"
            except Exception as e:
                logger.warning("Downloaded jobs.csv failed to load: %s", e)

    # Fallback to synthetic sample
    if not SAMPLE_CSV.exists():
        raise FileNotFoundError(
            f"Neither '{JOBS_CSV}' nor '{SAMPLE_CSV}' found. "
            "Please place a valid CSV in the data/ directory."
        )
    df = _load_and_clean(SAMPLE_CSV)
    return df, f"Synthetic sample dataset ({SAMPLE_CSV.name}, {len(df)} records — NOT real job postings)"


def validate_schema(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check that the DataFrame contains all required columns.

    Returns
    -------
    (is_valid, missing_columns)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return len(missing) == 0, missing


def download_hf_dataset() -> bool:
    """
    Attempt to download the HuggingFace Parquet file and save as jobs.csv.

    Returns True on success, False on any failure (network errors, wrong format, …).
    """
    try:
        response = requests.get(HF_PARQUET_URL, timeout=30)
        response.raise_for_status()

        # Read parquet from bytes
        df = pd.read_parquet(io.BytesIO(response.content))

        # Validate before saving
        is_valid, missing = validate_schema(df)
        if not is_valid:
            logger.warning(
                "Downloaded dataset missing columns: %s. Skipping save.", missing
            )
            return False

        JOBS_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(JOBS_CSV, index=False)
        logger.info("HuggingFace dataset saved to %s (%d rows)", JOBS_CSV, len(df))
        return True

    except Exception as e:
        logger.warning("HuggingFace download failed: %s", e)
        return False


def get_dataset_stats(df: pd.DataFrame) -> dict:
    """
    Return a summary statistics dict for display on the dashboard.
    """
    skill_counts = []
    for val in df["job_skill_set"].dropna():
        skills = [s.strip() for s in str(val).split(",") if s.strip()]
        skill_counts.append(len(skills))

    return {
        "total_records": len(df),
        "categories": sorted(df["category"].dropna().unique().tolist()),
        "num_categories": df["category"].nunique(),
        "records_per_category": df["category"].value_counts().to_dict(),
        "avg_skills_per_job": round(sum(skill_counts) / len(skill_counts), 1) if skill_counts else 0,
        "missing_descriptions": int(df["job_description"].isna().sum()),
        "missing_skills": int(df["job_skill_set"].isna().sum()),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_and_clean(path: Path) -> pd.DataFrame:
    """Load CSV or Parquet, validate, deduplicate, and fill NaN."""
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path, dtype=str)

    is_valid, missing = validate_schema(df)
    if not is_valid:
        raise ValueError(
            f"Dataset at '{path}' is missing required columns: {missing}. "
            f"Expected: {REQUIRED_COLUMNS}"
        )

    # Remove duplicate job_id (keep first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset=["job_id"], keep="first")
    dupes_removed = before - len(df)
    if dupes_removed:
        logger.info("Removed %d duplicate job_id entries.", dupes_removed)

    # Drop rows where both description and skills are empty
    mask_empty = df["job_description"].isna() & df["job_skill_set"].isna()
    df = df[~mask_empty].copy()

    # Fill remaining NaN in text columns
    for col in ["job_description", "job_skill_set", "job_title", "category"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)

    # Normalize whitespace
    for col in ["job_description", "job_skill_set", "job_title", "category"]:
        if col in df.columns:
            df[col] = df[col].str.strip()

    if len(df) == 0:
        raise ValueError(
            f"Dataset at '{path}' contains no usable records after cleaning."
        )

    logger.info("Loaded %d records from '%s'.", len(df), path)
    return df.reset_index(drop=True)
