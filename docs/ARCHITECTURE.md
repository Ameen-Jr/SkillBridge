# SkillBridge — Architecture

## Data flow

```mermaid
flowchart TD
    CSV[("data/sample_jobs.csv<br/>(or data/jobs.csv)")] --> DL[data_loader.py<br/>validate · dedupe · fill NaN]
    DL --> PP[preprocessing.py<br/>clean_text · normalize skills]
    PP --> SE[skill_extractor.py<br/>dictionary + alias + regex]
    PP --> TF[feature_engineering.py<br/>TF-IDF 1-2 grams, 500 features]
    SE --> CO[feature_engineering.py<br/>co-occurrence + frequency]
    TF --> KM[clustering.py<br/>K-means k=6 + silhouette]
    PIPE{{pipeline.py<br/>st.cache_resource}}
    DL -.-> PIPE
    KM -.-> PIPE
    CO -.-> PIPE
    TF -.-> PIPE

    U[User skills<br/>pages/1_Profile.py / sidebar] --> NU[normalize_user_skills]
    NU --> M[matching.py<br/>cosine + overlap baseline]
    PIPE --> M
    M --> GAP[skill_gap.py<br/>possessed · missing · frequency rank]
    GAP --> REC[recommendations.py<br/>learning plan from config.py]
    GAP --> RPT[reporting.py<br/>CSV export]
    CO --> NET[skill_network.py<br/>NetworkX → Plotly / PyVis]
    PIPE --> EV[evaluation.py<br/>P/R/F1 · P@K · silhouette]
```

## Layers (match PRD §9)

1. **Data input** — CSV dataset; user skills in `st.session_state.user_skills`.
2. **Processing** — `data_loader`, `preprocessing`, `skill_extractor`.
3. **Features** — TF-IDF matrix, co-occurrence counts (`feature_engineering`).
4. **ML / similarity** — K-means (`clustering`), cosine and overlap (`matching`).
5. **Recommendation** — `skill_gap`, `recommendations`.
6. **Presentation** — `app.py`, `pages/`, `reporting`.

## Key design decisions

- **`src/pipeline.py` holds the cached pipeline.** Pages previously imported it from `app.py`, which re-ran the whole dashboard script and crashed them. The models are fitted once per server process and reused across pages.
- **Schema contract:** every module relies on `job_id, category, job_title, job_description, job_skill_set` (`config.REQUIRED_COLUMNS`) plus derived `clean_description`, `extracted_skills`, `cluster`.
- **Canonical skill names** are lowercase; `aws` is canonical for AWS (alias `amazon web services → aws`).
- **Rule-based vs ML** is kept explicit: only K-means is trained; TF-IDF and cosine are feature/similarity tools; extraction and gap ranking are rules/statistics.
