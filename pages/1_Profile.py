"""
1_Profile.py — My Skill Profile page.

Allows users to:
- Enter existing technical skills via multi-select or free-text.
- Review recognized vs. unrecognized skills.
- Select career interests.
- Persist skills to session state for use by other pages.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import CAREER_CATEGORIES, SKILL_DICTIONARY
from src.pipeline import load_pipeline
from src.preprocessing import skills_from_user_input
from src.skill_extractor import normalize_user_skills

st.set_page_config(page_title="My Skill Profile — SkillBridge", page_icon="👤", layout="wide")

# Apply shared CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #c9d1d9; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
h1 { background: linear-gradient(90deg, #58a6ff, #3fb950); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
h2, h3 { color: #e6edf3 !important; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700 !important; }
.skill-tag { display: inline-block; background: rgba(56,139,253,0.15); border: 1px solid rgba(56,139,253,0.4); color: #58a6ff; padding: 3px 10px; border-radius: 20px; font-size: 0.8rem; margin: 2px; }
.skill-tag-missing { background: rgba(248,81,73,0.15); border-color: rgba(248,81,73,0.4); color: #f85149; }
.skill-tag-ok { background: rgba(63,185,80,0.15); border-color: rgba(63,185,80,0.4); color: #3fb950; }
.stButton > button { background: linear-gradient(135deg, #1f6feb, #388bfd); color: white; border: none; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "user_skills" not in st.session_state:
    st.session_state.user_skills = []
if "selected_category" not in st.session_state:
    st.session_state.selected_category = None

st.markdown("# 👤 My Skill Profile")
st.markdown("Enter your technical skills and career interests. Your profile will be used for career matching and gap analysis.")

st.markdown("---")

col_input, col_preview = st.columns([1.2, 1])

with col_input:
    st.markdown("### ✍️ Enter Your Skills")

    # Tab: multi-select or free-text
    tab_select, tab_text = st.tabs(["📋 Pick from List", "⌨️ Type Manually"])

    with tab_select:
        st.markdown("Select your skills from the recognized skill list:")
        sorted_skills = sorted(SKILL_DICTIONARY)
        selected_from_list = st.multiselect(
            "Skills",
            options=sorted_skills,
            default=[s for s in st.session_state.user_skills if s in sorted_skills],
            format_func=lambda x: x.title(),
            label_visibility="collapsed",
            key="multiselect_skills",
        )

        if st.button("✅ Use Selected Skills", key="btn_use_selected"):
            st.session_state.user_skills = sorted(set(selected_from_list))
            st.success(f"✅ {len(st.session_state.user_skills)} skills saved to your profile.")
            st.rerun()

    with tab_text:
        st.markdown("Type your skills, separated by commas or new lines:")
        raw_text = st.text_area(
            "Skills",
            height=200,
            placeholder="e.g.\nPython\nMachine Learning\nSQL, React, Docker",
            value=", ".join(st.session_state.user_skills),
            label_visibility="collapsed",
            key="freetext_skills",
        )

        if st.button("✅ Parse & Save Skills", key="btn_parse_text"):
            parsed = skills_from_user_input(raw_text)
            recognized, unrecognized = normalize_user_skills(parsed)
            st.session_state.user_skills = recognized

            if recognized:
                st.success(f"✅ Recognized {len(recognized)} skills.")
            if unrecognized:
                st.warning(
                    f"⚠️ {len(unrecognized)} skill(s) not recognized: "
                    f"{', '.join(unrecognized[:5])}{'…' if len(unrecognized) > 5 else ''}. "
                    "These were not added. Try the pick-from-list tab to find the correct name."
                )
            st.rerun()

    st.markdown("---")

    st.markdown("### 🎯 Career Interests (Optional)")
    st.markdown("Select a role from the loaded dataset to explore:")
    try:
        available_categories = sorted(load_pipeline()["df"]["category"].dropna().unique().tolist())
    except Exception:
        available_categories = list(CAREER_CATEGORIES)  # fallback if data cannot load
    current = st.session_state.selected_category
    interest = st.selectbox(
        "Career Interest",
        options=["No preference"] + available_categories,
        index=(available_categories.index(current) + 1) if current in available_categories else 0,
        label_visibility="collapsed",
        key="career_interest_select",
    )
    if st.button("Save Career Interest", key="btn_save_interest"):
        st.session_state.selected_category = None if interest == "No preference" else interest
        st.success(f"✅ Career interest set to: {interest}")


with col_preview:
    st.markdown("### 📋 Your Current Profile")

    if not st.session_state.user_skills:
        st.info("👈 Enter your skills on the left to see your profile here.")
    else:
        st.metric("Skills in Profile", len(st.session_state.user_skills))

        recognized, _ = normalize_user_skills(st.session_state.user_skills)

        st.markdown("**Recognized Skills:**")
        tags_html = "".join(
            f'<span class="skill-tag skill-tag-ok">✓ {s.title()}</span>'
            for s in recognized
        )
        st.markdown(tags_html, unsafe_allow_html=True)

        if st.session_state.selected_category:
            st.markdown(f"**Career Interest:** `{st.session_state.selected_category}`")

        st.markdown("---")
        st.markdown("**Skill Categories in Your Profile:**")

        # Categorize skills roughly by domain
        domains = {
            "Languages": {"python", "java", "javascript", "typescript", "c++", "c#", "go", "rust", "scala", "r", "swift", "kotlin", "php", "ruby"},
            "Web": {"html", "css", "react", "angular", "vue", "next.js", "node.js", "django", "flask", "fastapi"},
            "Data & ML": {"machine learning", "deep learning", "scikit-learn", "tensorflow", "pytorch", "pandas", "numpy", "data analysis", "statistical analysis"},
            "Cloud & DevOps": {"aws", "azure", "google cloud platform", "docker", "kubernetes", "ci/cd", "linux", "git"},
            "Databases": {"sql", "postgresql", "mysql", "mongodb", "redis", "elasticsearch"},
            "Other": set(),
        }

        user_set = set(st.session_state.user_skills)
        domain_counts = {}
        matched = set()
        for domain, domain_skills in domains.items():
            if domain == "Other":
                continue
            count = len(user_set & domain_skills)
            if count > 0:
                domain_counts[domain] = count
                matched |= (user_set & domain_skills)

        other_count = len(user_set - matched)
        if other_count > 0:
            domain_counts["Other"] = other_count

        if domain_counts:
            import plotly.express as px
            dc_df = [{"Domain": k, "Count": v} for k, v in domain_counts.items()]
            import pandas as pd
            fig = px.bar(
                pd.DataFrame(dc_df), x="Domain", y="Count",
                color="Count", color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig.update_layout(
                plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
                font={"color": "#c9d1d9", "family": "Inter"},
                coloraxis_showscale=False, height=250,
                margin={"t": 10, "b": 10},
            )
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.markdown("### 💡 Quick Add Suggestions")
    st.markdown("Popular skill sets by career path:")

    quick_sets = {
        "🐍 Python Dev": ["python", "django", "flask", "postgresql", "docker", "git"],
        "📊 Data Analyst": ["python", "sql", "pandas", "tableau", "statistical analysis", "excel"],
        "🤖 ML Engineer": ["python", "machine learning", "scikit-learn", "pytorch", "pandas", "numpy"],
        "🌐 Full-Stack": ["javascript", "react", "node.js", "postgresql", "html", "css"],
    }

    qcols = st.columns(2)
    for i, (label, skills) in enumerate(quick_sets.items()):
        with qcols[i % 2]:
            if st.button(label, key=f"quick_{i}"):
                st.session_state.user_skills = sorted(set(st.session_state.user_skills) | set(skills))
                st.success(f"Added {label} skills to your profile.")
                st.rerun()

st.markdown("---")
st.markdown("**Next step:** Navigate to **🎯 Career Matches** to see your matched career paths →")
