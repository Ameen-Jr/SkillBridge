# SkillBridge

**ML-Based Career Skill Gap Analysis and Personalized Learning Recommendations**
BTech CSE Microproject · Python · Streamlit · scikit-learn

SkillBridge lets a student enter the technical skills they have, compares them with skills extracted from job descriptions, shows which career roles look most similar, lists the skills they are missing (ranked by how often they appear in the dataset), and links to learning resources.

> **Data notice.** The bundled `data/sample_jobs.csv` is a **synthetic**, manually authored dataset (150 rows, 6 categories). It is **not** real job-posting data. All frequencies and scores describe this dataset only, not the job market. Similarity scores are **not** employment probabilities.

---

## Features (all implemented and covered by tests)

| Page | What it does |
|---|---|
| Overview (`app.py`) | Dataset summary, silhouette score, cluster distribution, pipeline description |
| 1 · My Skill Profile | Pick skills from the dictionary or type them; aliases (`ml`, `js`, `sklearn`…) are normalized; unrecognized entries are reported |
| 2 · Career Matches | Cosine-similarity ranking vs. a skill-overlap baseline, with matching/missing skills per role |
| 3 · Skill Gap | Possessed vs. missing skills for a chosen role, dataset-frequency priority, learning plan, CSV export |
| 4 · Skill Network | Co-occurrence graph (Plotly; PyVis optional) with threshold/layout controls |
| 5 · Model Evaluation | Skill-extraction P/R/F1, Precision@K for both matchers, silhouette, cluster contents |
| 6 · About | Methodology, limitations, setup |

## Methodology (what is ML and what is not)

| Step | Technique | Type |
|---|---|---|
| Skill extraction | Dictionary + alias + word-boundary regex (`src/skill_extractor.py`) | **Rule-based**, not ML |
| Text features | TF-IDF, 1–2-grams, max 500 features | Feature representation |
| Job grouping | K-means, k=6, `random_state=42` | **Unsupervised ML** |
| Career matching | Cosine similarity of the user's skill text vs. job-description vectors, averaged per category over the top 30 jobs | Similarity measure (no training) |
| Baseline | Fraction of a job's skills the user already has | Transparent rule |
| Gap ranking | Frequency of each missing skill among the role's jobs | Descriptive statistic |

K-means clusters are **not** verified career labels. Co-occurrence edges show association, not causation.

## Evaluation results (computed on the bundled synthetic data, October 2026)

| Component | Result | Evaluation set |
|---|---|---|
| Skill extraction | micro-precision 0.969, recall 0.984, F1 0.976 (62 TP, 2 FP, 1 FN) | 30 hand-labelled sentences |
| Career matching – cosine | P@1 = 0.50, P@3 = 1.00 | 20 hand-made skill sets |
| Career matching – overlap baseline | P@1 = 0.85, P@3 = 1.00 | same 20 |
| K-means (k=6) | silhouette (cosine) = 0.091 | 150 jobs |

How to read these honestly:
- The simple overlap baseline ranks the expected category first **more often** than cosine similarity. Neither is clearly better, and the evaluation sets are small.
- The evaluation sets were written by the project author with the dictionary in mind, so skill-extraction scores are optimistic and say little about unseen real-world text.
- The low silhouette (0.091) and mixed cluster contents (e.g. Python and Backend jobs share a cluster) show the synthetic categories overlap heavily in vocabulary.
- Known extraction misses: plural "RESTful APIs" is not matched; "Tailwind CSS" also yields `css`, "system design patterns" also yields `design patterns` (arguably correct, but not in the labels).

Reproduce with `python -m pytest tests -v` and the **Model Evaluation** page. Example exports are in `reports/`.

## Project structure

```
app.py                     Overview dashboard (entry point)
pages/                     1_Profile … 6_About (Streamlit multipage)
src/
  config.py                Paths, hyperparameters, 125-skill dictionary, aliases, resources
  data_loader.py           Load/validate/clean CSV; optional HuggingFace download
  preprocessing.py         Text cleaning, skill-name normalization
  skill_extractor.py       Dictionary/regex skill extraction
  feature_engineering.py   TF-IDF, user-query transform, co-occurrence
  clustering.py            K-means, cluster summaries, silhouette
  matching.py              Cosine matching, overlap baseline, explanations
  skill_gap.py             Gap, demand and ranking
  recommendations.py       Learning plan from curated resources
  skill_network.py         NetworkX graph + Plotly/PyVis rendering
  evaluation.py            Labelled eval sets and metrics
  reporting.py             CSV export
  pipeline.py              Cached shared pipeline (used by app and pages)
tests/                     173 pytest tests (incl. page smoke tests)
data/                      sample_jobs.csv + data/README.md
reports/                   Example exported results
docs/                      Architecture, project report, demo script, viva Q&A
```

## Setup on Windows (PowerShell)

```powershell
cd path\to\SkillBridge
python -m venv .venv
.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
streamlit run app.py
```

Run tests: `python -m pytest tests -v`

Optional network tab: `pip install pyvis` (Plotly graph works without it).

## Dataset

- **Default (bundled):** `data/sample_jobs.csv`, synthetic tech roles, columns `job_id, category, job_title, job_description, job_skill_set`. Everything in this README, the demo script and the evaluation numbers refers to this dataset.
- **Optional real data (opt-in):** the Hugging Face dataset `batuhanmtl/job-skill-set` (≈1,167 rows) is downloaded into `data/jobs.csv` **only** when you set `SKILLBRIDGE_USE_REAL_DATA=1`:

  ```powershell
  $env:SKILLBRIDGE_USE_REAL_DATA = "1"
  streamlit run app.py
  ```

  Verified on a real run: it downloads and passes schema validation. But its 5 categories are **INFORMATION-TECHNOLOGY, BUSINESS-DEVELOPMENT, FINANCE, SALES, HR** (about 220–240 rows each), not tech roles, and its skill tags are mostly non-technical (e.g. `employee relations`, `payroll processing`) and stored as Python-list strings (`['a', 'b']`, now parsed correctly). With the 125-skill tech dictionary, matches are weak (observed: all category scores within 0.13–0.16, silhouette 0.071). The app shows a warning, hides career-matching evaluation examples whose roles are absent, and takes the Profile role list from the loaded data. Check the dataset's licence before using it.

## Deploying as a website (Streamlit Community Cloud, free)

1. Push the repo to GitHub (branch `main`).
2. Go to https://share.streamlit.io, sign in with GitHub, choose *Create app*.
3. Repository `Ameen-Jr/SkillBridge`, branch `main`, main file `app.py`; in *Advanced settings* choose Python 3.12 or 3.13.
4. Deploy. You get a public `*.streamlit.app` link.

The deployed app uses the bundled dataset because `SKILLBRIDGE_USE_REAL_DATA` is not set. Free apps sleep after inactivity and wake on the next visit. Do not set that variable on the deployment unless you want the business-jobs dataset.

## Limitations

- Fixed 125-skill dictionary: unlisted skills (e.g. `google analytics`, `financial modeling`) are ignored.
- Only 47 skills have curated learning links; others get a Coursera search link.
- Small, synthetic, balanced data; frequencies reflect authoring choices.
- Cosine scores depend on vocabulary overlap, not on job suitability.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: src` | Run commands from the project root |
| Dashboard warns about non-tech categories | `SKILLBRIDGE_USE_REAL_DATA` is set; unset it (`Remove-Item Env:SKILLBRIDGE_USE_REAL_DATA`) to use the bundled data |
| Banner says "synthetic sample data" | Expected with the bundled dataset |
| Network tab says PyVis missing | `pip install pyvis` or use the Plotly tab |
| `use_container_width` deprecation warnings | Harmless on current Streamlit; cosmetic |
