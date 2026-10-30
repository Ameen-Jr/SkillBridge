"""
5_Model_Evaluation.py — Model Evaluation page.

Shows actual computed metrics for all ML components:
- Dataset and preprocessing statistics
- K-means clustering metrics (silhouette score, cluster info)
- Skill extraction evaluation (precision, recall, F1)
- Career matching Precision@K (cosine vs. overlap baseline)
- Evaluation limitations (clearly stated)

All metrics are computed from real data — none are fabricated.
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import load_pipeline
from src.evaluation import (
    evaluate_career_matching,
    evaluate_overlap_matching,
    evaluate_skill_extraction,
)
from src.reporting import export_evaluation_csv

st.set_page_config(page_title="Model Evaluation — SkillBridge", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #c9d1d9; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
h1 { background: linear-gradient(90deg, #58a6ff, #3fb950); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
h2, h3 { color: #e6edf3 !important; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700 !important; }
.stButton > button { background: linear-gradient(135deg, #1f6feb, #388bfd); color: white; border: none; border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 📈 Model Evaluation")
st.markdown("Actual computed metrics for all SkillBridge ML components. No scores are fabricated.")

st.warning(
    "**Evaluation Transparency Notice:** All metrics below are computed from small, "
    "hand-crafted evaluation sets. Results are indicative of system behavior on these "
    "examples but should not be generalized as definitive benchmarks."
)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

df = pipeline["df"]
vectorizer = pipeline["vectorizer"]
tfidf_matrix = pipeline["tfidf_matrix"]
stats = pipeline["stats"]
cluster_info = pipeline["cluster_info"]
silhouette = pipeline["silhouette"]

st.markdown("---")

# Run evaluations (cached with st.cache_data)
@st.cache_data(show_spinner="Computing evaluation metrics…")
def run_evaluations(_vectorizer, _tfidf_matrix, _df):
    skill_eval = evaluate_skill_extraction()
    cosine_eval = evaluate_career_matching(_vectorizer, _tfidf_matrix, _df, top_k=3)
    overlap_eval = evaluate_overlap_matching(_df, top_k=3)
    return skill_eval, cosine_eval, overlap_eval

skill_eval, cosine_eval, overlap_eval = run_evaluations(vectorizer, tfidf_matrix, df)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🗄️ Dataset Stats",
    "🔧 Skill Extraction",
    "🎯 Career Matching",
    "🧩 Clustering",
])

with tab1:
    st.markdown("### Dataset & Preprocessing Statistics")

    dc1, dc2, dc3, dc4 = st.columns(4)
    dc1.metric("Total Records", stats["total_records"])
    dc2.metric("Categories", stats["num_categories"])
    dc3.metric("Avg Skills/Job", stats["avg_skills_per_job"])
    dc4.metric("Missing Descriptions", stats["missing_descriptions"])

    st.markdown("#### Records per Category")
    cat_df = pd.DataFrame([
        {"Category": cat, "Records": cnt}
        for cat, cnt in stats["records_per_category"].items()
    ])
    fig = px.bar(cat_df, x="Category", y="Records", color="Records",
                 color_continuous_scale="Blues", template="plotly_dark",
                 labels={"Records": "Number of Jobs"})
    fig.update_layout(plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
                      font={"color": "#c9d1d9"}, coloraxis_showscale=False,
                      height=300, margin={"t": 10})
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Data Source")
    st.info(f"**Source used:** {pipeline['source']}")


with tab2:
    st.markdown("### Skill Extraction Evaluation")
    st.markdown(
        f"Evaluated on **{skill_eval['n_examples']} hand-labelled examples**. "
        "Metric: micro-averaged precision, recall, and F1."
    )

    ec1, ec2, ec3 = st.columns(3)
    ec1.metric("Precision", f"{skill_eval['micro_precision']:.3f}",
               help="Of the skills extracted, what fraction were correct?")
    ec2.metric("Recall", f"{skill_eval['micro_recall']:.3f}",
               help="Of all correct skills, what fraction were found?")
    ec3.metric("F1 Score", f"{skill_eval['micro_f1']:.3f}",
               help="Harmonic mean of precision and recall.")

    # Confusion summary
    conf_col1, conf_col2, conf_col3 = st.columns(3)
    conf_col1.metric("True Positives", skill_eval["total_tp"])
    conf_col2.metric("False Positives", skill_eval["total_fp"])
    conf_col3.metric("False Negatives", skill_eval["total_fn"])

    st.caption(skill_eval.get("limitation", ""))

    st.markdown("#### Per-Example Results")
    per_ex_df = pd.DataFrame(skill_eval["per_example"])[
        ["text", "precision", "recall", "f1", "tp", "fp", "fn"]
    ]
    per_ex_df.columns = ["Text Snippet", "Precision", "Recall", "F1", "TP", "FP", "FN"]
    st.dataframe(per_ex_df, use_container_width=True, hide_index=True)

    # Download
    eval_csv = export_evaluation_csv(skill_eval)
    st.download_button(
        "⬇️ Download Evaluation CSV",
        data=eval_csv,
        file_name="skill_extraction_evaluation.csv",
        mime="text/csv",
        key="download_eval",
    )


with tab3:
    st.markdown("### Career Matching Evaluation")
    st.markdown(
        f"Evaluated on **{cosine_eval['n_examples']} hand-crafted skill→role examples**. "
        "Metric: Precision@K (whether the expected category appears in top K results)."
    )

    if cosine_eval.get("n_skipped"):
        st.warning(
            f"{cosine_eval['n_skipped']} of the hand-crafted examples were skipped because their expected "
            "role is not a category in the loaded dataset. Metrics cover only the remaining examples."
        )
    if cosine_eval["n_examples"] == 0:
        st.info("None of the evaluation examples apply to the loaded dataset's categories, "
                "so Precision@K cannot be computed. The bundled sample dataset supports all examples.")
    else:
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Cosine P@1", f"{cosine_eval['precision_at_1']:.3f}",
                   help="Fraction of examples where the top category matched.")
        mc2.metric(f"Cosine P@3", f"{cosine_eval.get('precision_at_3', 0):.3f}")
        mc3.metric("Overlap P@1", f"{overlap_eval['precision_at_1']:.3f}")
        mc4.metric(f"Overlap P@3", f"{overlap_eval.get('precision_at_3', 0):.3f}")

        # Comparison chart
        compare_data = pd.DataFrame({
            "Method": ["Cosine Similarity", "Skill Overlap Baseline"],
            "Precision@1": [cosine_eval["precision_at_1"], overlap_eval["precision_at_1"]],
            f"Precision@3": [cosine_eval.get("precision_at_3", 0), overlap_eval.get("precision_at_3", 0)],
        })
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(name="Precision@1", x=compare_data["Method"], y=compare_data["Precision@1"], marker_color="#58a6ff"))
        fig2.add_trace(go.Bar(name="Precision@3", x=compare_data["Method"], y=compare_data["Precision@3"], marker_color="#3fb950"))
        fig2.update_layout(barmode="group", template="plotly_dark", plot_bgcolor="#0d1117", paper_bgcolor="#161b22",
                           font={"color": "#c9d1d9"}, height=320, margin={"t": 10},
                           legend={"font": {"color": "#c9d1d9"}})
        st.plotly_chart(fig2, use_container_width=True)

        _c1, _o1 = cosine_eval["precision_at_1"], overlap_eval["precision_at_1"]
        if _o1 > _c1:
            st.info(f"On this evaluation set the skill-overlap baseline (P@1 = {_o1:.2f}) ranked the expected "
                    f"category first more often than cosine similarity (P@1 = {_c1:.2f}). The simple baseline "
                    "is competitive; neither method should be presented as clearly superior.")
        elif _c1 > _o1:
            st.info(f"On this evaluation set cosine similarity (P@1 = {_c1:.2f}) ranked the expected category "
                    f"first more often than the overlap baseline (P@1 = {_o1:.2f}).")
        else:
            st.info(f"Both methods achieve the same P@1 ({_c1:.2f}) on this evaluation set.")

        st.caption(cosine_eval.get("limitation", ""))

        # Per-example detail
        with st.expander("View per-example matching results"):
            per_match = pd.DataFrame(cosine_eval["per_example"])
            per_match = per_match.rename(columns={
                "expected": "Expected", "top_predictions": "Top Predictions",
                "hit_at_1": "Hit@1", "hit_at_k": "Hit@3",
            })
            st.dataframe(
                per_match[["Expected", "Top Predictions", "Hit@1", "Hit@3"]],
                use_container_width=True, hide_index=True,
            )


with tab4:
    st.markdown("### K-Means Clustering")

    kc1, kc2, kc3 = st.columns(3)
    kc1.metric("Clusters (K)", pipeline["kmeans"].n_clusters)
    kc2.metric("Total Records", stats["total_records"])
    if silhouette is not None:
        kc3.metric("Silhouette Score", f"{silhouette:.4f}",
                   help="Ranges -1 to +1. Higher = better-separated clusters.")
    else:
        kc3.metric("Silhouette Score", "N/A")

    if silhouette is not None:
        st.info(
            f"**Silhouette score: {silhouette:.4f}**  "
            "Score > 0.5 suggests reasonable cluster separation. "
            "TF-IDF matrices are high-dimensional and sparse, "
            "so scores are typically lower than for dense low-dimensional data."
        )

    st.markdown("#### Cluster Details")
    for cinfo in cluster_info:
        with st.expander(f"Cluster {cinfo['cluster_id']} — {cinfo['size']} jobs"):
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                st.markdown("**Top TF-IDF Terms:**")
                st.markdown(", ".join(cinfo["top_terms"][:8]) or "—")
                st.markdown("**Top Skills:**")
                st.markdown(", ".join(cinfo["top_skills"][:8]) or "—")
            with ccol2:
                st.markdown("**Category Distribution:**")
                if cinfo["categories"]:
                    cat_series = pd.Series(cinfo["categories"]).sort_values(ascending=False)
                    for cat, count in cat_series.items():
                        st.write(f"• {cat}: {count} jobs")
                else:
                    st.write("No category data.")

    st.caption(
        "K-means clusters are NOT verified career labels. They reflect patterns "
        "in the TF-IDF feature space and should be interpreted with this in mind."
    )
