"""
3_Skill_Gap.py — Skill Gap Analysis page.

For a selected career role, shows:
- Skills the user already possesses
- Missing skills with dataset demand frequency
- Prioritized learning plan with resource links
- CSV export
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import load_pipeline
from src.config import CAREER_CATEGORIES
from src.recommendations import build_learning_plan, get_skill_display_name, get_resources_for_skill
from src.reporting import export_skill_gap_csv
from src.skill_gap import full_gap_analysis

st.set_page_config(page_title="Skill Gap Analysis — SkillBridge", page_icon="📊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #c9d1d9; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
h1 { background: linear-gradient(90deg, #58a6ff, #3fb950); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
h2, h3 { color: #e6edf3 !important; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700 !important; }
.skill-tag { display: inline-block; background: rgba(56,139,253,0.15); border: 1px solid rgba(56,139,253,0.4); color: #58a6ff; padding: 2px 8px; border-radius: 16px; font-size: 0.78rem; margin: 2px; }
.skill-tag-missing { background: rgba(248,81,73,0.15); border-color: rgba(248,81,73,0.4); color: #f85149; }
.skill-tag-ok { background: rgba(63,185,80,0.15); border-color: rgba(63,185,80,0.4); color: #3fb950; }
.resource-card { background: rgba(33,38,45,0.7); border: 1px solid #30363d; border-radius: 8px; padding: 0.6rem 1rem; margin: 0.3rem 0; }
.stButton > button { background: linear-gradient(135deg, #1f6feb, #388bfd); color: white; border: none; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

if "user_skills" not in st.session_state:
    st.session_state.user_skills = []
if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

st.markdown("# 📊 Skill Gap Analysis")
st.markdown("See exactly which skills you need for your target role, how in-demand they are, and where to learn them.")

if not st.session_state.user_skills:
    st.warning("⚠️ No skills in your profile. Go to **👤 My Skill Profile** first.")
    st.stop()

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

df = pipeline["df"]

# Role selector
st.markdown("---")
available_categories = sorted(df["category"].unique().tolist())

default_idx = 0
if st.session_state.selected_category and st.session_state.selected_category in available_categories:
    default_idx = available_categories.index(st.session_state.selected_category)

selected_cat = st.selectbox(
    "🎯 Select target career role:",
    options=available_categories,
    index=default_idx,
    key="gap_category_select",
)
st.session_state.selected_category = selected_cat

# Run gap analysis
user_skills = st.session_state.user_skills
gap_result = full_gap_analysis(user_skills, selected_cat, df)
gap = gap_result["gap"]
ranked = gap_result["ranked"]
learning_plan = build_learning_plan(ranked, max_skills=10)

st.markdown("---")

# Summary metrics
mc1, mc2, mc3, mc4 = st.columns(4)
with mc1:
    st.metric("Skills You Have", len(gap["possessed"]))
with mc2:
    st.metric("Skills to Learn", len(gap["missing"]))
with mc3:
    st.metric("Coverage", f"{gap['overlap_pct']}%")
with mc4:
    st.metric("Jobs Analyzed", gap_result["n_jobs_in_cat"])

st.markdown("---")

col_chart, col_skills = st.columns([1, 1.3])

with col_chart:
    st.markdown("### 📈 Skill Coverage")
    possessed = len(gap["possessed"])
    missing = len(gap["missing"])

    if possessed + missing > 0:
        fig_donut = go.Figure(data=[go.Pie(
            labels=["Possessed", "Missing"],
            values=[possessed, missing],
            hole=0.55,
            marker_colors=["#3fb950", "#f85149"],
        )])
        fig_donut.update_layout(
            template="plotly_dark",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#161b22",
            font={"color": "#c9d1d9", "family": "Inter"},
            legend={"font": {"color": "#c9d1d9"}},
            height=280,
            margin={"t": 10},
            annotations=[{
                "text": f"{gap['overlap_pct']}%",
                "font": {"size": 24, "color": "#58a6ff"},
                "showarrow": False,
            }],
        )
        st.plotly_chart(fig_donut, use_container_width=True)
    else:
        st.info("No skills data available for this role.")

    # Missing skills demand bar chart
    if not ranked.empty:
        st.markdown("### 📊 Missing Skill Demand")
        st.caption("Frequency = number of jobs in this category requiring this skill (from our dataset).")
        top_missing = ranked.head(10)
        fig_bar = px.bar(
            top_missing,
            x="frequency",
            y="skill",
            orientation="h",
            color="frequency",
            color_continuous_scale="Reds",
            template="plotly_dark",
            labels={"frequency": "Jobs Requiring Skill", "skill": "Skill"},
        )
        fig_bar.update_layout(
            plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
            font={"color": "#c9d1d9", "family": "Inter"},
            coloraxis_showscale=False, height=300,
            margin={"t": 10},
        )
        fig_bar.update_yaxes(tickmode="linear")
        st.plotly_chart(fig_bar, use_container_width=True)

with col_skills:
    st.markdown("### ✅ Skills You Already Have")
    if gap["possessed"]:
        st.markdown("".join(
            f'<span class="skill-tag skill-tag-ok">✓ {s.title()}</span>'
            for s in gap["possessed"]
        ), unsafe_allow_html=True)
    else:
        st.info("None of your skills directly matched this role's skill profile.")

    st.markdown("### ❌ Missing Skills (Prioritized)")
    if gap["missing"]:
        for _, row in ranked.iterrows():
            skill = row["skill"]
            freq = row["frequency"]
            pct = row["pct_of_jobs"]
            rank = int(row["priority_rank"])
            badge_color = "#f85149" if rank <= 3 else "#e3b341" if rank <= 6 else "#8b949e"
            st.markdown(
                f'<span style="color:{badge_color}; font-weight:600;">#{rank}</span> '
                f'<span class="skill-tag skill-tag-missing">{skill.title()}</span> '
                f'<span style="color:#8b949e; font-size:0.8rem;">— in {freq} jobs ({pct}%)</span>',
                unsafe_allow_html=True,
            )
    else:
        st.success("🎉 You already have all the skills for this role!")

st.markdown("---")

# Learning Plan
st.markdown("### 📚 Personalized Learning Plan")
st.markdown(
    f"Top {len(learning_plan)} skills to learn for **{selected_cat}**, "
    "ranked by how often they appear in job descriptions in our dataset."
)

if not learning_plan:
    st.info("Nothing to learn — you already cover all role skills!")
else:
    for item in learning_plan:
        skill = item["skill"]
        rank = item["priority_rank"]
        reason = item["priority_reason"]
        resources = item["resources"]

        with st.expander(f"**#{rank} {get_skill_display_name(skill)}** — {reason}"):
            st.markdown("**📖 Learning Resources:**")
            for res in resources:
                icon = "📚" if res["type"] == "docs" else "🎓" if res["type"] == "course" else "💡"
                badge = res["type"].upper()
                st.markdown(
                    f'<div class="resource-card">{icon} <a href="{res["url"]}" target="_blank" style="color:#58a6ff;">'
                    f'<strong>{res["title"]}</strong></a> '
                    f'<span style="background:#21262d;border:1px solid #30363d;padding:1px 6px;border-radius:4px;'
                    f'font-size:0.7rem;color:#8b949e;">{badge}</span></div>',
                    unsafe_allow_html=True,
                )

st.markdown("---")

# CSV Export
csv_bytes = export_skill_gap_csv(gap_result, learning_plan)
st.download_button(
    label="⬇️ Download Skill Gap Analysis CSV",
    data=csv_bytes,
    file_name=f"skill_gap_{selected_cat.replace(' ', '_').lower()}.csv",
    mime="text/csv",
    key="download_gap",
)
st.caption(
    "Export includes: possessed skills, missing skills, demand frequency, priority rank, and resource links."
)
