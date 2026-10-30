"""
skill_extractor.py — Technical skill extraction from text.

Uses word-boundary regular expressions to match canonical skill names
against job description text. Avoids false positives by only matching
complete tokens (e.g., "py" will not match inside "copy" or "python").

The skill dictionary and alias mapping are defined in config.py and
can be extended without modifying this module.
"""

import re

import pandas as pd

from src.config import SKILL_ALIASES, SKILL_DICTIONARY
from src.preprocessing import clean_text, normalize_skill_name, split_skill_tags


# ---------------------------------------------------------------------------
# Pre-compiled patterns for performance
# ---------------------------------------------------------------------------

def _build_patterns() -> list[tuple[str, re.Pattern]]:
    """
    Build (canonical_skill, compiled_regex) pairs.

    Sorted by length descending so longer matches (e.g., "machine learning")
    take priority over shorter ones (e.g., "machine").
    """
    patterns = []
    skills_sorted = sorted(SKILL_DICTIONARY, key=len, reverse=True)
    for skill in skills_sorted:
        # Escape special regex characters (e.g., c++, c#, node.js)
        escaped = re.escape(skill)
        if skill[-1].isalnum():
            end_boundary = r"\b"
        else:
            end_boundary = rf"(?![{re.escape(skill[-1])}\w])"
        pattern = re.compile(r"\b" + escaped + end_boundary, re.IGNORECASE)
        patterns.append((skill, pattern))
    return patterns


_SKILL_PATTERNS: list[tuple[str, re.Pattern]] = _build_patterns()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_skills(text: str) -> list[str]:
    """
    Extract recognized technical skills from a piece of text.

    The text is first cleaned (lowercased, punctuation normalized).
    Word-boundary matching ensures partial substring matches are avoided:
    - "py" does NOT match "copy" or "python"
    - "r" does NOT match "react" or "docker"

    Returns a sorted, deduplicated list of canonical skill names.
    """
    if not text or not isinstance(text, str):
        return []

    cleaned = clean_text(text)
    found: set[str] = set()

    for canonical_skill, pattern in _SKILL_PATTERNS:
        if pattern.search(cleaned):
            found.add(canonical_skill)

    return sorted(found)


def extract_skills_from_row(skill_string: str, description: str = "") -> list[str]:
    """
    Extract skills from both the explicit skill_set column and the description.

    Combines results from both sources, deduplicates, and returns sorted list.
    """
    # From the structured skill_set field (comma-separated)
    from_skills_col: set[str] = set()
    if skill_string:
        for raw in split_skill_tags(skill_string):
            norm = normalize_skill_name(raw)
            if norm in SKILL_DICTIONARY:
                from_skills_col.add(norm)

    # From free-text description
    from_desc = set(extract_skills(description))

    combined = from_skills_col | from_desc
    return sorted(combined)


def extract_skills_from_df(
    df: pd.DataFrame,
    skill_col: str = "job_skill_set",
    desc_col: str = "job_description",
) -> pd.Series:
    """
    Apply skill extraction to every row in a DataFrame.

    Returns a pd.Series of lists, one per row.
    """
    return df.apply(
        lambda row: extract_skills_from_row(
            str(row.get(skill_col, "")),
            str(row.get(desc_col, "")),
        ),
        axis=1,
    )


def normalize_user_skills(user_skills: list[str]) -> tuple[list[str], list[str]]:
    """
    Normalize user-entered skills against the skill dictionary.

    Returns
    -------
    recognized   : skills that exist in the canonical dictionary
    unrecognized : skills that could not be matched (after alias resolution)
    """
    recognized: list[str] = []
    unrecognized: list[str] = []

    for skill in user_skills:
        normalized = normalize_skill_name(skill)
        if normalized in SKILL_DICTIONARY:
            recognized.append(normalized)
        else:
            # Try partial match for common abbreviations
            # (only if the result is in the dictionary)
            alias_resolved = SKILL_ALIASES.get(normalized, normalized)
            if alias_resolved in SKILL_DICTIONARY:
                recognized.append(alias_resolved)
            else:
                unrecognized.append(skill)

    return sorted(set(recognized)), unrecognized


def get_all_skills_in_dataset(df: pd.DataFrame) -> list[str]:
    """Return a sorted list of all unique canonical skills found in the dataset."""
    all_skills: set[str] = set()
    for skill_list in df.get("extracted_skills", pd.Series(dtype=object)):
        if isinstance(skill_list, list):
            all_skills.update(skill_list)
    return sorted(all_skills)
