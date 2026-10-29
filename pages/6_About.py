"""
6_About.py — About page for SkillBridge.

Covers:
- Project description and objectives
- Technology stack
- Dataset attribution
- ML methodology explanation
- Limitations and future scope
- Local setup instructions (Windows PowerShell)
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

st.set_page_config(page_title="About — SkillBridge", page_icon="ℹ️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #c9d1d9; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
h1 { background: linear-gradient(90deg, #58a6ff, #3fb950); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
h2, h3 { color: #e6edf3 !important; }
code { background: rgba(33,38,45,0.9) !important; border: 1px solid #30363d; border-radius: 4px; }
pre { background: #161b22 !important; border: 1px solid #30363d !important; border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.markdown("# ℹ️ About SkillBridge")
st.markdown("""
**SkillBridge** is a BTech Computer Science and Engineering microproject that uses machine learning
and natural language processing to help students identify career skill gaps and discover personalized
learning pathways.
""")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🎯 Objectives")
    st.markdown("""
    1. Extract and normalize technical skills from job descriptions.
    2. Apply TF-IDF vectorization to represent jobs numerically.
    3. Use K-means clustering to discover groups of similar roles.
    4. Match student skill profiles to career paths using cosine similarity.
    5. Identify skills students have and skills they are missing.
    6. Rank missing skills by their demand in the dataset.
    7. Recommend transparent, configurable learning resources.
    8. Visualize skill relationships in an interactive network graph.
    9. Evaluate ML components using real computed metrics.
    10. Provide downloadable analysis results.
    """)

    st.markdown("### 🔬 Methodology")
    st.markdown("""
    **Data Preparation:** Job descriptions are loaded, validated, deduplicated,
    and normalized (lowercased, punctuation-cleaned).

    **Skill Extraction:** A configurable dictionary of ~80 canonical technical skills
    with alias mapping (e.g., ML → Machine Learning) is matched against cleaned text
    using word-boundary regular expressions to avoid false positives.

    **TF-IDF Vectorization:** scikit-learn's `TfidfVectorizer` converts each job
    description into a 500-dimensional sparse vector using 1–2-gram features.

    **K-means Clustering:** Jobs are grouped into 6 clusters (adjustable).
    Reproducibility is ensured via `random_state=42`. Silhouette score is computed
    when valid (≥2 clusters, ≥2 samples per cluster).

    **Career Matching:** The user's skill list is transformed into the same TF-IDF
    space and ranked against all job records by cosine similarity.
    A skill-overlap ratio is computed as a baseline for comparison.

    **Skill Gap Analysis:** Missing skills are identified by comparing the user's
    skills to the most common skills in the selected career category. Missing skills
    are ranked by their frequency in the category's job records.
    """)

with col2:
    st.markdown("### 🛠️ Technology Stack")
    tech_data = {
        "Component": [
            "Language", "Web Interface", "Data Processing",
            "NLP & Vectorization", "Clustering", "Similarity",
            "Visualization", "Network Graph", "Testing",
        ],
        "Technology": [
            "Python 3.13", "Streamlit ≥1.35", "Pandas ≥2.2, NumPy ≥1.26",
            "scikit-learn TfidfVectorizer", "scikit-learn KMeans",
            "scikit-learn cosine_similarity", "Plotly ≥5.22",
            "NetworkX + Plotly (PyVis optional)", "Pytest ≥8",
        ],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(tech_data), use_container_width=True, hide_index=True)

    st.markdown("### 📋 Dataset Attribution")
    st.markdown("""
    **Bundled Dataset:** `data/sample_jobs.csv` is a synthetic 150-row dataset
    created for demonstration purposes. It is NOT based on real job postings.

    **Optional Dataset:** [`batuhanmtl/job-skill-set`](https://huggingface.co/datasets/batuhanmtl/job-skill-set)
    — ~1,170 rows created by BATUHAN MUTLU and published on Hugging Face.
    Please check the dataset page for the current license before use.

    ⚠️ Synthetic data must not be presented as real labour-market data.
    """)

    st.markdown("### ⚠️ Limitations")
    st.markdown("""
    - Skill extraction relies on a fixed dictionary and misses skills not in the vocabulary.
    - The synthetic dataset is small and does not represent real market diversity.
    - Cosine similarity scores are similarity measures, NOT employment probability estimates.
    - K-means clusters are not verified career labels — they reflect TF-IDF feature patterns.
    - Skill demand frequencies are based only on the loaded dataset, not the full job market.
    - The evaluation set is hand-crafted (30–20 examples) and is not an industry benchmark.
    - Learning resources are curated manually and may become outdated.
    """)

    st.markdown("### 🚀 Future Enhancements")
    st.markdown("""
    - Résumé upload and automatic skill extraction from PDF.
    - Semantic embeddings (BERT/Sentence-BERT) for richer skill matching.
    - Personalized multi-step learning roadmaps.
    - Skill progression tracking across sessions.
    - Integration with live course catalogue APIs.
    - Larger, regularly updated real-world job dataset.
    """)

st.markdown("---")

st.markdown("### 💻 Local Setup (Windows PowerShell)")
st.code("""
# 1. Clone or download the project
cd d:\\Coding\\SkillBridgeML

# 2. Create a virtual environment
python -m venv .venv
.venv\\Scripts\\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py

# 5. Run tests
python -m pytest tests\\ -v
""", language="powershell")

st.markdown("### 📁 Project Structure")
st.code("""
SkillBridgeML/
├── app.py                  # Dashboard (Page 1)
├── requirements.txt
├── README.md
├── PRD.md
├── data/
│   ├── sample_jobs.csv     # Synthetic demo dataset
│   └── README.md
├── src/
│   ├── config.py           # Central configuration
│   ├── data_loader.py      # Dataset loading & validation
│   ├── preprocessing.py    # Text cleaning
│   ├── skill_extractor.py  # Skill dictionary matching
│   ├── feature_engineering.py  # TF-IDF, co-occurrence
│   ├── clustering.py       # K-means
│   ├── matching.py         # Cosine + overlap matching
│   ├── skill_gap.py        # Gap analysis
│   ├── recommendations.py  # Learning resources
│   ├── skill_network.py    # Network graph
│   ├── evaluation.py       # Metrics & eval sets
│   └── reporting.py        # CSV export
├── pages/
│   ├── 1_Profile.py
│   ├── 2_Career_Matches.py
│   ├── 3_Skill_Gap.py
│   ├── 4_Skill_Network.py
│   ├── 5_Model_Evaluation.py
│   └── 6_About.py
├── tests/                  # Pytest test suite
└── docs/                   # Reports, diagrams, viva Q&A
""", language="text")

st.markdown("---")
st.markdown("""
**SkillBridge** — BTech CSE Microproject  
Built with ❤️ using Python, Streamlit, and Scikit-learn  
*All similarity scores are similarity measures, not employment probability estimates.*
""")
