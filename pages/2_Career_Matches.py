"""
2_Career_Matches.py — Career Matches page.

Shows ranked career matches using:
1. Cosine similarity (primary TF-IDF method)
2. Skill overlap baseline (simple Jaccard-like ratio)

Both methods are compared side by side.
Scores are labelled as similarity measures — NOT employment probabilities.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import load_pipeline
from src.feature_engineering import transform_user_query
from src.matching import (
    aggregate_matches_by_category,
    aggregate_overlap_by_category,
    build_match_explanation,
    cosine_match,
    skill_overlap_match,
)
from src.reporting import export_career_matches_csv

st.set_page_config(page_title="Career Matches — SkillBridge", page_icon="🎯", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #c9d1d9; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
h1 { background: linear-gradient(90deg, #58a6ff, #3fb950); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
h2, h3 { color: #e6edf3 !important; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700 !important; }
.match-card { background: rgba(33,38,45,0.8); border: 1px solid #30363d; border-radius: 12px; padding: 1rem; margin: 0.5rem 0; }
.skill-tag { display: inline-block; background: rgba(56,139,253,0.15); border: 1px solid rgba(56,139,253,0.4); color: #58a6ff; padding: 2px 8px; border-radius: 16px; font-size: 0.78rem; margin: 2px; }
.skill-tag-missing { background: rgba(248,81,73,0.15); border-color: rgba(248,81,73,0.4); color: #f85149; }
.skill-tag-ok { background: rgba(63,185,80,0.15); border-color: rgba(63,185,80,0.4); color: #3fb950; }
.stButton > button { background: linear-gradient(135deg, #1f6feb, #388bfd); color: white; border: none; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

if "user_skills" not in st.session_state:
    st.session_state.user_skills = []

st.markdown("# 🎯 Career Matches")
st.markdown("Your skills matched against career profiles in our dataset.")

# Guard — no skills
if not st.session_state.user_skills:
    st.warning("⚠️ You haven't entered any skills yet. Go to **👤 My Skill Profile** first.")
    st.stop()

# Load pipeline
try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

df = pipeline["df"]
vectorizer = pipeline["vectorizer"]
tfidf_matrix = pipeline["tfidf_matrix"]
user_skills = st.session_state.user_skills

# Transform user vector
user_vec = transform_user_query(vectorizer, user_skills)

# Run both matching methods
cosine_df = cosine_match(user_vec, tfidf_matrix, df, top_k=30)
overlap_df = skill_overlap_match(user_skills, df, top_k=30)

# Aggregate by category
cosine_cat = aggregate_matches_by_category(cosine_df)
overlap_cat = aggregate_overlap_by_category(overlap_df)

st.markdown("---")

# Summary metrics
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.metric("Skills in Profile", len(user_skills))
with mc2:
    top_cat = cosine_cat.iloc[0]["category"] if not cosine_cat.empty else "N/A"
    st.metric("Top Match", top_cat)
with mc3:
    top_score = cosine_cat.iloc[0]["avg_score"] if not cosine_cat.empty else 0
    st.metric("Top Score", f"{top_score:.3f}")
with mc4:
    if not cosine_cat.empty:
        best_skills = cosine_cat.iloc[0].get("sample_skills", [])
        best_skills = best_skills if isinstance(best_skills, list) else []
        match_pct = round(100 * len(set(user_skills) & set(best_skills)) / len(best_skills)) if best_skills else 0
        st.metric("Skill Coverage", f"{match_pct:.0f}%")
    else:
        st.metric("Skill Coverage", "—")

if cosine_cat.empty:
    st.warning(
        "None of your skills overlap with the vocabulary of the job descriptions in this dataset, "
        "so no similarity scores could be computed. Add more skills on **👤 My Skill Profile**."
    )

st.markdown("---")

# Tabs: Category view vs. Job-level view vs. Comparison
tab1, tab2, tab3 = st.tabs(["🏷️ By Career Category", "📄 Individual Jobs", "⚖️ Method Comparison"])

with tab1:
    st.markdown("### Career Category Rankings (Cosine Similarity)")
    st.caption("Similarity score: higher = more vocabulary overlap with your skills. This is NOT an employment probability.")

    if cosine_cat.empty:
        st.warning("No matches found. Try adding more skills to your profile.")
    else:
        # Bar chart
        fig = px.bar(
            cosine_cat,
            x="avg_score",
            y="category",
            orientation="h",
            color="avg_score",
            color_continuous_scale="Blues",
            template="plotly_dark",
            labels={"avg_score": "Avg Cosine Score", "category": "Career Category"},
            text=cosine_cat["avg_score"].round(3),
        )
        fig.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
            font={"color": "#c9d1d9", "family": "Inter"},
            coloraxis_showscale=False, height=350,
            margin={"t": 10},
        )
        fig.update_traces(textposition="outside", textfont_color="#c9d1d9")
        st.plotly_chart(fig, use_container_width=True)

        # Expandable match cards
        st.markdown("### 📋 Match Details")
        for _, row in cosine_cat.iterrows():
            cat = row["category"]
            score = row["avg_score"]
            sample_skills = row.get("sample_skills", [])

            explanation = build_match_explanation(sample_skills, user_skills, score, method="cosine")
            matching = explanation["matching_skills"]
            missing = explanation["missing_skills"][:8]

            with st.expander(f"**{cat}** — Score: {score:.3f} | {explanation['overlap_pct']}% skill overlap"):
                ecol1, ecol2 = st.columns(2)
                with ecol1:
                    st.markdown("**✅ Matching Skills:**")
                    if matching:
                        st.markdown("".join(f'<span class="skill-tag skill-tag-ok">{s.title()}</span>' for s in matching), unsafe_allow_html=True)
                    else:
                        st.caption("No direct skill matches in top-scoring job.")
                with ecol2:
                    st.markdown("**❌ Key Missing Skills:**")
                    if missing:
                        st.markdown("".join(f'<span class="skill-tag skill-tag-missing">{s.title()}</span>' for s in missing), unsafe_allow_html=True)
                    else:
                        st.caption("You cover most skills in this role!")
                st.caption(explanation["explanation"])

with tab2:
    st.markdown("### Individual Job Record Matches (Top 10)")
    if cosine_df.empty:
        st.warning("No matches found.")
    else:
        display_df = cosine_df.head(10)[["job_title", "category", "cosine_score"]].copy()
        display_df.columns = ["Job Title", "Category", "Cosine Score"]
        display_df["Cosine Score"] = display_df["Cosine Score"].round(4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download
        csv_bytes = export_career_matches_csv(cosine_df, user_skills)
        st.download_button(
            label="⬇️ Download Career Matches CSV",
            data=csv_bytes,
            file_name="career_matches.csv",
            mime="text/csv",
            key="download_matches",
        )

with tab3:
    st.markdown("### ⚖️ Cosine Similarity vs. Skill Overlap Baseline")
    st.markdown("""
    Two methods are compared:
    - **Cosine Similarity**: TF-IDF vocabulary match — captures context beyond exact skill names.
    - **Skill Overlap**: Fraction of the role's skills you already possess — directly interpretable.
    """)

    if cosine_cat.empty or overlap_cat.empty:
        st.warning("Not enough data for comparison.")
    else:
        # Merge the two
        compare = cosine_cat[["category", "avg_score"]].rename(columns={"avg_score": "Cosine Score"})
        overlap_renamed = overlap_cat[["category", "avg_score"]].rename(columns={"avg_score": "Overlap Score"})
        merged = compare.merge(overlap_renamed, on="category", how="outer").fillna(0)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name="Cosine Similarity",
            x=merged["category"],
            y=merged["Cosine Score"],
            marker_color="#58a6ff",
        ))
        fig2.add_trace(go.Bar(
            name="Skill Overlap",
            x=merged["category"],
            y=merged["Overlap Score"],
            marker_color="#3fb950",
        ))
        fig2.update_layout(
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#161b22",
            font={"color": "#c9d1d9", "family": "Inter"},
            height=400,
            legend={"font": {"color": "#c9d1d9"}},
            xaxis={"tickangle": -20},
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("**Interpretation:**")
        st.info(
            "When cosine similarity and skill overlap agree on the top category, "
            "that recommendation is more reliable. Disagreements may indicate that "
            "the job descriptions use different vocabulary than your skill list."
        )
