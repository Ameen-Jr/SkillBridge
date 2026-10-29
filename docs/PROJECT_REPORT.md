# SkillBridge — BTech Microproject Report

## 1. Abstract
SkillBridge is a Streamlit application that compares a student's technical skills with skills extracted from job descriptions. It uses a dictionary-based extractor, TF-IDF features, K-means clustering and cosine similarity, plus a transparent skill-overlap baseline, to suggest career categories, rank missing skills by dataset frequency and link to learning resources.

## 2. Objectives
See PRD.md §3. All eleven objectives are implemented; export, network and evaluation pages are working.

## 3. Dataset
`data/sample_jobs.csv`: 150 synthetic job records (23–26 per category across 6 categories), average 8.6 skill tags per job. It is synthetic and must not be presented as real labour-market data. An optional Hugging Face dataset can be placed in `data/jobs.csv`; its schema and licence were not verified in this work.

## 4. Methodology
1. **Cleaning** — lowercase, ASCII normalization, preserve `c++`, `c#`, `node.js`.
2. **Skill extraction** — 125-skill dictionary, alias map, word-boundary regexes; skills are the union of the tag column and description matches.
3. **TF-IDF** — unigrams + bigrams, English stop-words, sublinear TF, 500 features.
4. **K-means** — k=6, `n_init=10`, `random_state=42`; k clamped for small datasets; silhouette (cosine) only when every cluster has ≥2 samples.
5. **Matching** — user skills → TF-IDF vector → cosine against all jobs → average of top-30 per category. Baseline: |user ∩ job| / |job|.
6. **Gap analysis** — role skills = 30 most frequent in the category; missing = role − user; priority = frequency.
7. **Network** — co-occurrence counts ≥ threshold, NetworkX layout, Plotly rendering.

## 5. Results (computed, bundled synthetic data)
| Metric | Value |
|---|---|
| Skill extraction micro-P / R / F1 | 0.969 / 0.984 / 0.976 (30 examples) |
| Cosine P@1 / P@3 | 0.50 / 1.00 (20 examples) |
| Overlap baseline P@1 / P@3 | 0.85 / 1.00 (20 examples) |
| Silhouette, K-means k=6 | 0.091 |
| Skill network (threshold 3, ≤40 nodes) | 40 nodes, 308 edges, density 0.395 |

Example for the profile *Python, SQL, pandas, Docker, git, ML*: top cosine category Machine Learning Engineer (0.205); coverage of that role 20.7%; top missing skills deep learning, statistical analysis, PyTorch, NumPy, scikit-learn. See `reports/` for exported CSVs.

## 6. Discussion
The overlap baseline matched or beat cosine similarity on top-1 accuracy, which is plausible because the evaluation profiles are lists of exact dictionary skills and the baseline compares exactly those. The weak silhouette and mixed clusters show the synthetic categories are not separable by vocabulary. Evaluation sets are small and author-written, so results are indicative only.

## 7. Testing
149 pytest tests: unit tests for each module, plus regression tests and a Streamlit AppTest smoke test of all 7 pages with empty, typical and unrecognized profiles. All pass (see README for command). The HuggingFace download path and a manual browser walkthrough were not performed in the build environment.

## 8. Limitations and future work
See README "Limitations" and viva Q15.

## 9. Defects fixed during continuation
Invalid f-string in `app.py`; `from app import` coupling (now `src/pipeline.py`); AWS duplicate canonical name; removed Plotly `titlefont`; IndexError on Career Matches with unmatched skills; empty-demand DataFrame; sidebar accepting unrecognized skills; typo "30–20 examples"; 8 dataset skills missing from the dictionary.
