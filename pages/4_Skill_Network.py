"""
4_Skill_Network.py — Interactive Skill Co-occurrence Network page.

Visualizes relationships between technical skills as a graph.
Nodes = skills, Edges = co-occurrence in the same job description.

Primary renderer: Plotly (always available).
Optional renderer: PyVis HTML (if pyvis is installed).

Important: edge connections represent co-occurrence, NOT causal relationships.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import load_pipeline
from src.skill_network import (
    build_network_graph,
    get_network_stats,
    get_plotly_network_figure,
    get_pyvis_html,
)

st.set_page_config(page_title="Skill Network — SkillBridge", page_icon="🕸️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
.stApp { background: linear-gradient(135deg, #0d1117 0%, #161b22 100%); color: #c9d1d9; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #161b22 0%, #0d1117 100%); border-right: 1px solid #30363d; }
h1 { background: linear-gradient(90deg, #58a6ff, #3fb950); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 700 !important; }
h2, h3 { color: #e6edf3 !important; }
[data-testid="stMetricValue"] { color: #58a6ff !important; font-weight: 700 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("# 🕸️ Skill Co-occurrence Network")
st.markdown("An interactive graph showing which technical skills appear together in job descriptions.")

st.info(
    "**How to read this graph:** Each node is a skill. An edge (connection) between two skills "
    "means they appeared together in the same job description at least N times (N = threshold below). "
    "**A connection does not mean one skill causes a need for the other.** "
    "Larger nodes appear in more job records. Brighter edges indicate stronger co-occurrence."
)

try:
    pipeline = load_pipeline()
except Exception as e:
    st.error(f"Pipeline error: {e}")
    st.stop()

cooccurrence = pipeline["cooccurrence"]
skill_frequency = pipeline["skill_frequency"]

st.markdown("---")

# Controls
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns(3)
with ctrl_col1:
    threshold = st.slider(
        "Co-occurrence Threshold",
        min_value=1,
        max_value=max(10, max(cooccurrence.values()) if cooccurrence else 10),
        value=3,
        step=1,
        help="Minimum number of times two skills must co-occur to show a connection.",
        key="network_threshold",
    )
with ctrl_col2:
    layout_choice = st.selectbox(
        "Graph Layout",
        options=["spring", "kamada_kawai", "circular"],
        index=0,
        help="Algorithm used to position nodes.",
        key="network_layout",
    )
with ctrl_col3:
    max_nodes = st.slider(
        "Max Nodes",
        min_value=10,
        max_value=60,
        value=40,
        step=5,
        help="Limit the number of skills shown for performance.",
        key="network_max_nodes",
    )

# Build graph
graph = build_network_graph(cooccurrence, skill_frequency, threshold=threshold, max_nodes=max_nodes)
stats = get_network_stats(graph)

# Stats row
sc1, sc2, sc3, sc4 = st.columns(4)
with sc1:
    st.metric("Skill Nodes", stats["nodes"])
with sc2:
    st.metric("Connections", stats["edges"])
with sc3:
    st.metric("Graph Density", f"{stats['density']:.4f}")
with sc4:
    st.metric("Threshold", threshold)

st.markdown("---")

# Visualization tabs
vtab1, vtab2 = st.tabs(["📊 Plotly (Interactive)", "🌐 PyVis (if available)"])

with vtab1:
    fig = get_plotly_network_figure(graph, layout=layout_choice)
    st.plotly_chart(fig, use_container_width=True)

with vtab2:
    pyvis_html = get_pyvis_html(graph)
    if pyvis_html:
        st.components.v1.html(pyvis_html, height=600, scrolling=True)
        st.success("✅ PyVis is available — showing enhanced interactive graph above.")
    else:
        st.info(
            "PyVis is not installed in this environment. "
            "The Plotly graph on the previous tab provides full interactivity. "
            "To install PyVis: `pip install pyvis`"
        )

st.markdown("---")

# Top connected skills
if stats["top_connected"]:
    st.markdown("### 🔗 Most Connected Skills")
    st.markdown("Skills with the most connections to other skills in the network:")
    cols = st.columns(min(5, len(stats["top_connected"])))
    for i, (skill, degree) in enumerate(stats["top_connected"][:5]):
        with cols[i]:
            st.metric(skill, f"{degree} links")

st.markdown("---")
st.caption(
    "Edges represent skill co-occurrence in job descriptions. "
    "Data source: the loaded dataset. "
    "Network layout uses NetworkX spring/Kamada-Kawai/circular positioning."
)
