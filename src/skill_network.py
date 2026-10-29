"""
skill_network.py — Interactive skill co-occurrence network visualization.

Builds a NetworkX graph from skill co-occurrence data and returns
Plotly figure objects for display in Streamlit.

Edges represent co-occurrence within the same job description.
A co-occurrence is NOT evidence of a causal relationship between skills.

PyVis is attempted as an optional enhancement; Plotly is the primary renderer.
"""

import logging
from typing import Optional

import networkx as nx
import numpy as np
import plotly.graph_objects as go

from src.config import MAX_NETWORK_NODES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_network_graph(
    cooccurrence: dict[tuple[str, str], int],
    skill_frequency: dict[str, int],
    threshold: int = 3,
    max_nodes: int = MAX_NETWORK_NODES,
) -> nx.Graph:
    """
    Build a NetworkX undirected graph from skill co-occurrence data.

    Parameters
    ----------
    cooccurrence    : dict mapping (skill_a, skill_b) → co-occurrence count
    skill_frequency : dict mapping skill → number of job records it appears in
    threshold       : minimum co-occurrence count to include an edge
    max_nodes       : cap on the number of nodes for performance

    Returns
    -------
    NetworkX Graph with node attributes: frequency, label
    and edge attributes: weight (co-occurrence count)
    """
    G = nx.Graph()

    # Filter edges by threshold
    filtered = {pair: count for pair, count in cooccurrence.items() if count >= threshold}

    if not filtered:
        logger.warning("No skill pairs meet the co-occurrence threshold of %d.", threshold)
        return G

    # Collect all skill nodes that appear in the filtered edges
    node_set: set[str] = set()
    for skill_a, skill_b in filtered:
        node_set.add(skill_a)
        node_set.add(skill_b)

    # Limit to top-N skills by frequency if needed
    if len(node_set) > max_nodes:
        top_skills = sorted(node_set, key=lambda s: skill_frequency.get(s, 0), reverse=True)[:max_nodes]
        node_set = set(top_skills)
        filtered = {
            (a, b): c for (a, b), c in filtered.items()
            if a in node_set and b in node_set
        }

    # Add nodes with attributes
    for skill in node_set:
        G.add_node(
            skill,
            frequency=skill_frequency.get(skill, 0),
            label=skill.title(),
        )

    # Add edges
    for (skill_a, skill_b), count in filtered.items():
        if skill_a in node_set and skill_b in node_set:
            G.add_edge(skill_a, skill_b, weight=count)

    logger.info(
        "Skill network built: %d nodes, %d edges (threshold=%d).",
        G.number_of_nodes(), G.number_of_edges(), threshold,
    )
    return G


# ---------------------------------------------------------------------------
# Plotly visualization
# ---------------------------------------------------------------------------

def get_plotly_network_figure(
    graph: nx.Graph,
    layout: str = "spring",
    title: str = "Skill Co-occurrence Network",
) -> go.Figure:
    """
    Create an interactive Plotly figure from a NetworkX graph.

    Node size reflects skill frequency.
    Edge width reflects co-occurrence count.

    Parameters
    ----------
    graph  : NetworkX Graph from build_network_graph
    layout : one of "spring", "kamada_kawai", "circular"
    title  : figure title

    Returns
    -------
    Plotly Figure object
    """
    if graph.number_of_nodes() == 0:
        # Return an empty placeholder figure
        fig = go.Figure()
        fig.update_layout(
            title="No skill network to display (try lowering the co-occurrence threshold)",
            plot_bgcolor="#0d1117",
            paper_bgcolor="#0d1117",
            font={"color": "#c9d1d9"},
        )
        return fig

    # Compute layout positions
    seed = 42
    if layout == "kamada_kawai":
        try:
            pos = nx.kamada_kawai_layout(graph)
        except Exception:
            pos = nx.spring_layout(graph, seed=seed)
    elif layout == "circular":
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=seed, k=1.5)

    # Build edge traces
    edge_traces = []
    max_weight = max((d.get("weight", 1) for _, _, d in graph.edges(data=True)), default=1)

    for u, v, data in graph.edges(data=True):
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = data.get("weight", 1)
        opacity = 0.2 + 0.6 * (weight / max_weight)
        width = 0.5 + 3.0 * (weight / max_weight)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line={"width": width, "color": f"rgba(88, 166, 255, {opacity:.2f})"},
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Build node trace
    node_x, node_y, node_text, node_hover, node_size, node_color = [], [], [], [], [], []
    frequencies = [graph.nodes[n].get("frequency", 1) for n in graph.nodes()]
    max_freq = max(frequencies) if frequencies else 1

    for node in graph.nodes():
        x, y = pos[node]
        freq = graph.nodes[node].get("frequency", 1)
        degree = graph.degree(node)

        node_x.append(x)
        node_y.append(y)
        node_text.append(node.title())
        node_hover.append(
            f"<b>{node.title()}</b><br>"
            f"Appears in {freq} jobs<br>"
            f"Connected to {degree} other skills"
        )
        node_size.append(12 + 28 * (freq / max_freq))
        # Color intensity by frequency
        node_color.append(freq)

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        textfont={"size": 10, "color": "#c9d1d9"},
        hovertext=node_hover,
        hoverinfo="text",
        marker={
            "size": node_size,
            "color": node_color,
            "colorscale": "Viridis",
            "colorbar": {
                "title": {"text": "Job Frequency", "font": {"color": "#c9d1d9"}},
                "thickness": 12,
                "tickfont": {"color": "#c9d1d9"},
            },
            "line": {"width": 1, "color": "#30363d"},
            "opacity": 0.9,
        },
        showlegend=False,
    )

    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title={
            "text": title,
            "font": {"size": 16, "color": "#c9d1d9"},
        },
        showlegend=False,
        hovermode="closest",
        plot_bgcolor="#0d1117",
        paper_bgcolor="#161b22",
        margin={"b": 20, "l": 5, "r": 5, "t": 50},
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        height=600,
        font={"color": "#c9d1d9"},
    )
    return fig


# ---------------------------------------------------------------------------
# Optional PyVis fallback
# ---------------------------------------------------------------------------

def get_pyvis_html(graph: nx.Graph, bgcolor: str = "#0d1117") -> Optional[str]:
    """
    Attempt to generate PyVis interactive HTML.

    Returns the HTML string on success, or None if PyVis is not available
    or the graph is empty.

    This is an optional enhancement — the Plotly renderer is always available.
    """
    if graph.number_of_nodes() == 0:
        return None

    try:
        from pyvis.network import Network  # type: ignore

        net = Network(height="550px", width="100%", bgcolor=bgcolor, font_color="#c9d1d9")
        net.from_nx(graph)

        # Improve visual settings
        net.set_options("""
        {
          "physics": {
            "forceAtlas2Based": {
              "gravitationalConstant": -50,
              "centralGravity": 0.005,
              "springLength": 130
            },
            "solver": "forceAtlas2Based"
          },
          "interaction": {"hover": true},
          "nodes": {"font": {"size": 12}}
        }
        """)

        html = net.generate_html()
        return html

    except ImportError:
        logger.info("PyVis not installed — using Plotly network renderer.")
        return None
    except Exception as e:
        logger.warning("PyVis rendering failed: %s", e)
        return None


def get_network_stats(graph: nx.Graph) -> dict:
    """Return summary statistics about the network for display."""
    if graph.number_of_nodes() == 0:
        return {"nodes": 0, "edges": 0, "density": 0.0, "top_connected": []}

    density = nx.density(graph)
    degree_sorted = sorted(graph.degree(), key=lambda x: x[1], reverse=True)
    top_connected = [(node.title(), deg) for node, deg in degree_sorted[:10]]

    return {
        "nodes": graph.number_of_nodes(),
        "edges": graph.number_of_edges(),
        "density": round(density, 4),
        "top_connected": top_connected,
    }
