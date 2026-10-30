"""
preprocessing.py — Text cleaning and normalization utilities.

Applies consistent preprocessing to job descriptions and user-entered
skill text before TF-IDF vectorization and skill extraction.
"""

import ast
import re
import unicodedata

import pandas as pd

from src.config import SKILL_ALIASES


# ---------------------------------------------------------------------------
# Text normalization
# ---------------------------------------------------------------------------

def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters to their ASCII equivalents where possible."""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_text(text: str) -> str:
    """
    Lowercase, normalize Unicode, remove special characters, and collapse whitespace.

    Used to preprocess job descriptions before TF-IDF vectorization.
    Does NOT remove all punctuation — preserves tokens like 'c++', 'c#', 'node.js'.
    """
    if not isinstance(text, str):
        return ""
    text = normalize_unicode(text)
    text = text.lower()
    # Replace newlines, tabs, and multiple spaces with a single space
    text = re.sub(r"[\r\n\t]+", " ", text)
    # Remove characters that are not alphanumeric, space, hyphen, dot, slash, or #/+
    text = re.sub(r"[^a-z0-9 \-./#+]", " ", text)
    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_skill_name(skill: str) -> str:
    """
    Normalize a single skill string:
    - Lowercase and strip
    - Apply alias mapping
    - Normalize internal whitespace

    Returns the canonical form of the skill.
    """
    if not isinstance(skill, str):
        return ""
    skill = skill.lower().strip()
    skill = re.sub(r"\s+", " ", skill)
    # Apply alias mapping
    return SKILL_ALIASES.get(skill, skill)


def normalize_skill_list(skills: list[str]) -> list[str]:
    """
    Normalize a list of skill strings: apply aliases, deduplicate, sort.

    Returns a deduplicated list of canonical skill names.
    """
    seen: set[str] = set()
    result: list[str] = []
    for skill in skills:
        normalized = normalize_skill_name(skill)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return sorted(result)


def split_skill_tags(value) -> list[str]:
    """
    Split a skill-tag field into individual raw tags.

    Supports both formats seen in the datasets:
    - comma separated:        "Python, SQL, Docker"
    - Python-list literal:    "['Python', 'SQL', 'Docker']"
    Non-string input returns an empty list.
    """
    if not isinstance(value, str) or not value.strip():
        return []
    text = value.strip()
    if text.startswith("["):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(s).strip() for s in parsed if str(s).strip()]
        except (ValueError, SyntaxError):
            text = text.strip("[]")
    return [s.strip().strip("'\"") for s in text.split(",") if s.strip().strip("'\"")]


def parse_skill_string(skill_string: str) -> list[str]:
    """
    Parse a comma-separated skill string into a normalized list of skill names.

    Example:
        "Python, Machine Learning, SQL" → ["machine learning", "python", "sql"]
    """
    if not skill_string or not isinstance(skill_string, str):
        return []
    return normalize_skill_list(split_skill_tags(skill_string))


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply text cleaning to all text columns in the dataset DataFrame.

    Adds:
    - `clean_description`: cleaned version of job_description
    - `extracted_skills`: parsed list of normalized skills from job_skill_set
    """
    df = df.copy()
    df["clean_description"] = df["job_description"].apply(clean_text)
    df["extracted_skills"] = df["job_skill_set"].apply(parse_skill_string)
    return df


def skills_from_user_input(raw_input: str) -> list[str]:
    """
    Parse user-typed skill input.

    Accepts comma-separated or newline-separated values.
    Returns a normalized, deduplicated list.
    """
    # Accept comma and newline as separators
    parts = re.split(r"[,\n]+", raw_input)
    skills = [p.strip() for p in parts if p.strip()]
    return normalize_skill_list(skills)
