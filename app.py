"""
app.py — SkillBridge: ML-Based Career Skill Gap Analysis System.

Entry point for the Streamlit multi-page application.
This file renders the Overview Dashboard (Page 1) and bootstraps
the shared ML pipeline used by all other pages.

Architecture:
- @st.cache_resource caches the loaded dataset and fitted TF-IDF + K-means models.
- st.session_state stores the user's skill profile across page navigation.
- All analysis modules are imported from src/.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src/ is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.config import SKILL_DICTIONARY
from src.pipeline import load_pipeline

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="SkillBridge — Career Skill Gap Analyzer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "SkillBridge — BTech CSE Microproject. ML-Based Career Skill Gap Analysis.",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS — premium dark theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #161b22 100%);
    color: #c9d1d9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #30363d;
}

[data-testid="stSidebar"] .css-1d391kg {
    padding-top: 1rem;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(33, 38, 45, 0.8);
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 1rem;
    backdrop-filter: blur(10px);
}

[data-testid="stMetricValue"] {
    color: #58a6ff !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
}

[data-testid="stMetricLabel"] {
    color: #8b949e !important;
    font-size: 0.85rem !important;
}

/* Info boxes */
.stAlert {
    border-radius: 8px;
    border-left: 4px solid #58a6ff;
}

/* Headers */
h1 {
    background: linear-gradient(90deg, #58a6ff, #3fb950);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 2.2rem !important;
}

h2, h3 {
    color: #e6edf3 !important;
    font-weight: 600 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
    transition: all 0.2s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff);
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(56, 139, 253, 0.4);
}

/* Expanders */
.streamlit-expanderHeader {
    background: rgba(33, 38, 45, 0.6);
    border: 1px solid #30363d;
    border-radius: 8px;
    color: #c9d1d9 !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(33, 38, 45, 0.5);
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #30363d;
}

.stTabs [data-baseweb="tab"] {
    color: #8b949e;
    font-weight: 500;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #1f6feb, #388bfd) !important;
    color: white !important;
    border-radius: 6px;
}

/* Data frames */
[data-testid="stDataFrame"] {
    border: 1px solid #30363d;
    border-radius: 8px;
}

/* Skill tags */
.skill-tag {
    display: inline-block;
    background: rgba(56, 139, 253, 0.15);
    border: 1px solid rgba(56, 139, 253, 0.4);
    color: #58a6ff;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.8rem;
    margin: 2px;
    font-weight: 500;
}

.skill-tag-missing {
    background: rgba(248, 81, 73, 0.15);
    border-color: rgba(248, 81, 73, 0.4);
    color: #f85149;
}

.skill-tag-possessed {
    background: rgba(63, 185, 80, 0.15);
    border-color: rgba(63, 185, 80, 0.4);
    color: #3fb950;
}

/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(31, 111, 235, 0.1) 0%, rgba(63, 185, 80, 0.1) 100%);
    border: 1px solid rgba(56, 139, 253, 0.3);
    border-radius: 16px;
    padding: 2rem;
    margin-bottom: 1.5rem;
}

/* Pipeline steps */
.pipeline-step {
    background: rgba(33, 38, 45, 0.7);
    border: 1px solid #30363d;
    border-left: 3px solid #58a6ff;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "user_skills" not in st.session_state:
    st.session_state.user_skills = []

if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🎯 SkillBridge")
    st.markdown("*ML-Based Career Skill Gap Analysis*")
    st.markdown("---")

    # Quick skill entry in sidebar
    st.markdown("### ⚡ Quick Skills")
    sidebar_skills = st.text_area(
        "Enter your skills (comma-separated):",
        value=", ".join(st.session_state.user_skills),
        height=80,
        key="sidebar_skill_input",
        placeholder="e.g. Python, SQL, React",
    )

    if st.button("Update Profile", key="sidebar_update_btn"):
        from src.preprocessing import skills_from_user_input
        from src.skill_extractor import normalize_user_skills
        recognized, unrecognized = normalize_user_skills(skills_from_user_input(sidebar_skills))
        st.session_state.user_skills = recognized
        st.success(f"✅ {len(recognized)} skills saved.")
        if unrecognized:
            st.warning("Not recognized (skipped): " + ", ".join(unrecognized[:5]))

    if st.session_state.user_skills:
        st.markdown("**Current skills:**")
        tags = " ".join(
            f'<span class="skill-tag">{s.title()}</span>'
            for s in st.session_state.user_skills[:8]
        )
        if len(st.session_state.user_skills) > 8:
            tags += f' <span class="skill-tag">+{len(st.session_state.user_skills) - 8} more</span>'
        st.markdown(tags, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Navigation**")
    st.markdown("""
    - 🏠 **Overview** (this page)
    - 👤 **1 · My Skill Profile**
    - 🎯 **2 · Career Matches**
    - 📊 **3 · Skill Gap Analysis**
    - 🕸️ **4 · Skill Network**
    - 📈 **5 · Model Evaluation**
    - ℹ️ **6 · About**
    """)

    st.markdown("---")
    st.caption("BTech CSE Microproject · Python 3.13 · Streamlit")


# ---------------------------------------------------------------------------
# Main Dashboard — Overview (Page 1)
# ---------------------------------------------------------------------------

st.markdown("# 🎯 SkillBridge")
st.markdown("### ML-Based Career Skill Gap Analysis & Personalized Learning Recommendations")

# Hero banner
st.markdown("""
<div class="hero-banner">
<h4 style="color: #58a6ff; margin: 0 0 0.5rem 0;">Welcome to SkillBridge</h4>
<p style="color: #c9d1d9; margin: 0; line-height: 1.6;">
SkillBridge uses <strong>TF-IDF vectorization</strong>, <strong>K-means clustering</strong>, and
<strong>cosine similarity</strong> to analyze job descriptions and help you identify the technical skills
needed for your target career role. Enter your skills and get personalized, data-driven recommendations.
</p>
</div>
""", unsafe_allow_html=True)

# Load pipeline
try:
    pipeline = load_pipeline()
    stats = pipeline["stats"]
    load_error = None
except Exception as e:
    load_error = str(e)
    st.error(f"❌ Failed to load dataset: {e}")
    st.info("Please ensure `data/sample_jobs.csv` exists in the project directory.")
    st.stop()

# Dataset source notice
source = pipeline["source"]
if "Synthetic" in source:
    st.warning(
        f"⚠️ **Using synthetic sample data** ({source}). "
        "This dataset is for demonstration only and does not represent real job postings. "
        "See `data/README.md` for instructions on adding real data."
    )
else:
    st.success(f"✅ Dataset loaded: {source}")

st.markdown("---")

# Key metrics row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("📋 Job Records", f"{stats['total_records']:,}")
with col2:
    st.metric("🏷️ Career Categories", stats['num_categories'])
with col3:
    st.metric("🔧 Recognized Skills", len(SKILL_DICTIONARY))
with col4:
    sil = pipeline.get("silhouette")
    st.metric("📐 Silhouette Score", f"{sil:.3f}" if sil is not None else "N/A")

st.markdown("---")

# Two-column layout for charts
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("### 📊 Dataset Distribution")
    cat_counts = pd.DataFrame([
        {"Category": cat, "Records": count}
        for cat, count in stats["records_per_category"].items()
    ]).sort_values("Records", ascending=True)

    fig_bar = px.bar(
        cat_counts,
        x="Records",
        y="Category",
        orientation="h",
        color="Records",
        color_continuous_scale="Blues",
        template="plotly_dark",
    )
    fig_bar.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font={"color": "#c9d1d9", "family": "Inter"},
        coloraxis_showscale=False,
        margin={"t": 10, "b": 10},
        height=320,
    )
    fig_bar.update_traces(marker_line_color="#30363d", marker_line_width=0.5)
    st.plotly_chart(fig_bar, use_container_width=True)

with col_right:
    st.markdown("### 🧩 Cluster Distribution")
    cluster_info = pipeline["cluster_info"]
    cluster_df = pd.DataFrame([
        {"Cluster": f"Cluster {c['cluster_id']}", "Jobs": c['size']}
        for c in cluster_info
    ])

    fig_pie = px.pie(
        cluster_df,
        values="Jobs",
        names="Cluster",
        hole=0.45,
        template="plotly_dark",
        color_discrete_sequence=px.colors.sequential.Blues_r,
    )
    fig_pie.update_layout(
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        font={"color": "#c9d1d9", "family": "Inter"},
        margin={"t": 10, "b": 10},
        height=320,
        legend={"font": {"color": "#c9d1d9"}},
    )
    st.plotly_chart(fig_pie, use_container_width=True)

st.markdown("---")

# ML Pipeline Overview
st.markdown("### ⚙️ Analysis Pipeline")
_sil = pipeline.get("silhouette")
silhouette_text = f"{_sil:.3f}" if _sil is not None else "N/A"
steps = [
    ("1. Data Loading & Validation", "Loads job descriptions from CSV, validates schema, handles missing values."),
    ("2. Text Preprocessing", "Cleans text: lowercasing, punctuation normalization, whitespace collapse."),
    ("3. Skill Extraction", "Identifies technical skills using a word-boundary dictionary matcher with alias resolution."),
    ("4. TF-IDF Vectorization", f"Converts {stats['total_records']} job descriptions into {len(pipeline['vectorizer'].vocabulary_)}-dimensional feature vectors."),
    ("5. K-Means Clustering", f"Groups similar jobs into {pipeline['kmeans'].n_clusters} clusters (silhouette = {silhouette_text})."),
    ("6. Cosine Similarity Matching", "Ranks career profiles by similarity to the user's skill vector."),
    ("7. Skill Gap Analysis", "Identifies possessed vs. missing skills and ranks them by dataset demand."),
    ("8. Learning Recommendations", "Maps missing skills to curated learning resources."),
]

for title, desc in steps:
    st.markdown(f"""
    <div class="pipeline-step">
    <strong style="color: #58a6ff;">{title}</strong><br>
    <span style="color: #8b949e; font-size: 0.9rem;">{desc}</span>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Getting started
st.markdown("### 🚀 Getting Started")
gcol1, gcol2, gcol3 = st.columns(3)

with gcol1:
    st.markdown("""
    **Step 1 — Enter Your Skills**
    Go to **👤 My Skill Profile** to enter your current technical skills.
    The system will recognize and normalize them automatically.
    """)

with gcol2:
    st.markdown("""
    **Step 2 — View Career Matches**
    Navigate to **🎯 Career Matches** to see ranked career paths based on your skills,
    with similarity scores and matching explanations.
    """)

with gcol3:
    st.markdown("""
    **Step 3 — Analyze the Gap**
    Go to **📊 Skill Gap Analysis** to see exactly which skills you need to learn,
    how in-demand they are, and where to learn them.
    """)

st.markdown("---")
st.caption(
    "SkillBridge — BTech CSE Microproject | "
    "Built with Python 3.13, Streamlit, Scikit-learn, Plotly | "
    "All similarity scores are similarity measures, not employment probability estimates."
)
