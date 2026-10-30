# SkillBridge — Viva Questions and Answers

**1. What problem does SkillBridge solve?**
Students cannot easily see how their skills compare with what job descriptions ask for. It extracts skills from job descriptions, matches the student's profile to career categories, lists missing skills ranked by frequency, and links learning resources.

**2. Which parts are machine learning and which are not?**
K-means is the only trained (unsupervised) model. TF-IDF is a feature representation; cosine similarity is a distance measure with no training. Skill extraction (dictionary/regex) and gap ranking (frequency counts) are rule-based/statistical.

**3. Why TF-IDF?**
It weights terms by how distinctive they are across the job corpus, giving fixed-length sparse vectors that K-means and cosine similarity can use. Limitation: it ignores word meaning (no synonyms beyond our alias list).

**4. How does K-means work here, and how good is it?**
It partitions the 150 TF-IDF vectors into k=6 clusters by minimizing within-cluster distance (`random_state=42`). The silhouette score is 0.091 (cosine metric) — weak separation. Clusters mix categories (e.g., Python and Backend developers share one cluster), so they are *not* verified career labels.

**5. How did you pick k=6?**
It equals the number of dataset categories, for easy comparison. We did not tune k; an elbow/silhouette sweep is future work.

**6. Explain cosine similarity matching.**
The user's skills are joined into a pseudo-document, transformed with the already-fitted vectorizer, and compared with every job vector. The top 30 jobs are averaged per category. The score measures vocabulary overlap, not suitability or hiring chance.

**7. Why include a skill-overlap baseline? What did you find?**
A baseline shows whether the more complex method adds value. On our 20 hand-made profiles the baseline achieved P@1 0.85 vs 0.50 for cosine; both reached P@3 1.00. So the simple method is at least as good on this data; we report that honestly. Caveat: overlap favours jobs with few listed skills.

**8. How is skill extraction evaluated?**
30 hand-labelled sentences; micro precision 0.969, recall 0.984, F1 0.976. The sentences were written with the dictionary in mind, so this overstates real-world performance. Known misses: plural "RESTful APIs"; extra `css` from "Tailwind CSS".

**9. How do you avoid false positives like `r` matching in "react"?**
Patterns use word boundaries (`\b`), with special handling for skills ending in symbols (`c++`, `c#`). Ambiguous words such as "go" can still match in ordinary prose — a documented limitation.

**10. How are missing skills prioritized?**
Rank = how many jobs in the selected category mention the skill (ties alphabetical). The UI says "in N jobs (x%) in this dataset" — it does not claim market-wide demand.

**11. What does the skill network show?**
Nodes are skills, edges mean the two skills appeared in the same job at least *threshold* times. It shows association within the dataset, not causation.

**12. What defects did you find when continuing the project?**
(a) An invalid f-string crashed the dashboard; (b) pages imported `app.py`, re-running it and crashing every page — fixed by moving the cached pipeline to `src/pipeline.py`; (c) AWS was split into two names, halving its counts and breaking learning-resource lookup; (d) a Plotly property removed in newer versions broke the network page; (e) an index error when no entered skill is in the vocabulary; (f) empty-demand DataFrame lost its columns. Each has a regression test.

**13. Why is the dataset synthetic and what are the consequences?**
No real, licensed dataset could be verified. Frequencies reflect the author's design, not the labour market, so conclusions apply only to this dataset. The app supports swapping in a real CSV with the same columns. We also tried the public Hugging Face dataset (opt-in): it loads, but its categories are business functions (IT, HR, Finance, Sales, Business Development) with mostly non-technical skills, so it is unsuitable for tech-role matching without a different dictionary — a documented limitation.

**14. How is the app kept fast?**
`st.cache_resource` in `src/pipeline.py` fits TF-IDF and K-means once per server process; pages reuse it.

**15. How would you improve it?**
Real licensed data, a larger labelled evaluation set (ideally annotated by someone other than the author), semantic embeddings, tuning k, and an extractor that handles plurals/synonyms.
