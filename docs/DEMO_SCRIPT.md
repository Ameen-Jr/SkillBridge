# SkillBridge — Demonstration Script (~8 minutes)

**Before you start:** activate the venv, run `streamlit run app.py`, confirm the dashboard shows the *synthetic data* banner. Say it out loud: the data is synthetic, so every number describes this dataset only.

1. **Overview (1 min).** Point to 150 records, 6 categories, 125 recognized skills, silhouette 0.091. Explain the 8-step pipeline cards. Mention the silhouette is low and why (categories share vocabulary).
2. **My Skill Profile (1.5 min).** Open *Type Manually*, enter `Python, SQL, pandas, Docker, git, ml`, click *Parse & Save*. Show that `ml` became *Machine Learning*. Type `zzzunknown` to show the unrecognized-skill warning, then restore the real skills.
3. **Career Matches (2 min).** Expected top category for this profile: Machine Learning Engineer (avg cosine ≈ 0.205). Open the match card to show matching and missing skills. Open *Method Comparison*: the overlap baseline ranks Python/Backend/ML almost equally — discuss that the two methods can disagree. Stress: scores are similarities, not hiring probabilities. Download the CSV.
4. **Skill Gap (1.5 min).** Select *Machine Learning Engineer*. Coverage is about 21%. Top missing skills by dataset frequency: deep learning (65.4% of ML jobs), statistical analysis, PyTorch, NumPy, scikit-learn. Expand a skill to show resources, then download the CSV.
5. **Skill Network (1 min).** Lower/raise the threshold. Python, Git, Docker are the hubs. Read the info box: co-occurrence ≠ causation.
6. **Model Evaluation (1 min).** Skill extraction F1 0.976 on 30 hand-labelled examples; cosine P@1 0.50 vs overlap P@1 0.85, both P@3 1.00. Read the transparency notice: small hand-made sets, optimistic.

**Edge-case demo (optional):** clear the profile → pages show a friendly warning; enter an unknown skill → Career Matches explains that no similarity could be computed.

**Expected numbers** were produced with the bundled dataset on 2026-10-30; they should reproduce exactly because all random states are fixed.
