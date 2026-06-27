from __future__ import annotations

import numpy as np
import rustworkx as rx

from ._base import render_plot

def network_edge_stat(
    network: rx.PyDiGraph,
    **kwargs,
):
    """Plot edge score distributions by source and target node type.

    Use this function to inspect correlation score distributions across network
    edge categories.

    Anti-Patterns
    -------------
    - Do NOT pass a generic NetworkX graph. The input must be a
      ``rustworkx.PyDiGraph`` whose nodes expose ``type`` and whose edges expose
      score attributes such as ``cor_score``.
    - Do NOT use this function to summarize regression scores only. The current
      plot displays correlation score violins.

    Parameters
    ----------
    network : rustworkx.PyDiGraph
        Regulatory network whose nodes provide a ``type`` attribute and whose
        edge data may provide ``cor_score`` and ``regr_score`` attributes.
    **kwargs
        Additional rendering options passed to :func:`snapatac2.pl.render_plot`,
        such as ``show``, ``interactive``, ``out_file``, and ``scale``.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns a Plotly figure when ``show=False`` and ``out_file=None``;
        otherwise renders or saves the plot and returns ``None``.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> import rustworkx as rx
    >>> import snapatac2 as snap
    >>> graph = rx.PyDiGraph()
    >>> peak = graph.add_node(SimpleNamespace(type="peak"))
    >>> gene = graph.add_node(SimpleNamespace(type="gene"))
    >>> graph.add_edge(peak, gene, SimpleNamespace(cor_score=0.7, regr_score=0.2))
    >>> fig = snap.pl.network_edge_stat(graph, show=False)
    >>> fig.update_layout(title="Network edge scores")
    """
    from collections import defaultdict
    import plotly.graph_objects as go

    scores = defaultdict(lambda: defaultdict(lambda: []))

    for fr, to, data in network.edge_index_map().values():
        type = "{} -> {}".format(network[fr].type, network[to].type)
        if data.cor_score is not None:
            scores["correlation"][type].append(data.cor_score)
        if data.regr_score is not None:
            scores["regression"][type].append(data.regr_score)
    
    fig = go.Figure()

    for key, vals in scores["correlation"].items():
        fig.add_trace(go.Violin(
            y=vals,
            name=key,
            box_visible=True,
            meanline_visible=True
        ))

    return render_plot(fig, **kwargs)

def network_scores(
    network: rx.PyDiGraph,
    score_name: str,
    width: float = 800,
    height: float = 400,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
):
    """Plot average network edge scores by distance-to-TSS bin.

    Use this function to inspect how an edge score changes with genomic distance
    from transcription start sites.

    Anti-Patterns
    -------------
    - Do NOT pass a score name that is absent from edge data objects. Each edge
      must expose ``distance`` and the requested ``score_name`` attribute.
    - Do NOT use this function for node-level scores. It aggregates edge-level
      attributes only.

    Parameters
    ----------
    network : rustworkx.PyDiGraph
        Regulatory network whose edge data objects provide ``distance`` and the
        score attribute named by ``score_name``.
    score_name : str
        Name of the edge attribute to average within each distance bin.
    width : float
        Width of the rendered plot in pixels.
    height : float
        Height of the rendered plot in pixels.
    show : bool
        Whether to display the figure immediately.
    interactive : bool
        Whether to display an interactive Plotly figure when ``show=True``.
    out_file : str or None
        Output path for saving the plot. Supported suffixes include ``.svg``,
        ``.pdf``, ``.png``, and ``.html``.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns a Plotly figure when ``show=False`` and ``out_file=None``;
        otherwise renders or saves the plot and returns ``None``.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> import rustworkx as rx
    >>> import snapatac2 as snap
    >>> graph = rx.PyDiGraph()
    >>> peak = graph.add_node(SimpleNamespace(type="peak"))
    >>> gene = graph.add_node(SimpleNamespace(type="gene"))
    >>> graph.add_edge(peak, gene, SimpleNamespace(distance=1500, regr_score=0.4))
    >>> fig = snap.pl.network_scores(graph, score_name="regr_score", show=False)
    >>> fig.update_layout(title="Regression score by distance")
    """
    import plotly.express as px
    import pandas as pd
    import bisect

    def human_format(num):
        num = float('{:.3g}'.format(num))
        magnitude = 0
        while abs(num) >= 1000:
            magnitude += 1
            num /= 1000.0
        return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'), ['', 'K', 'M', 'B', 'T'][magnitude])

    break_points = [100, 500, 2000, 20000, 50000, 100000, 500000]
    intervals = []
    for i in range(len(break_points)):
        if i == 0:
            intervals.append("0 - " + human_format(break_points[i]))
        else:
            intervals.append(human_format(break_points[i - 1]) + " - " + human_format(break_points[i]))
    intervals.append("> 500k")
    values = [[] for _ in range(len(intervals))]
    for e in network.edges():
        i = bisect.bisect(break_points, e.distance)
        sc = getattr(e, score_name)
        if sc is not None:
            values[i].append(sc)

    intervals, values = zip(*filter(lambda x: len(x[1]) > 0, zip(intervals, values)))
    values = [np.nanmean(v) for v in values]

    df = pd.DataFrame({
        "Distance to TSS (bp)": intervals,
        "Average score": values,
    })
    fig = px.bar(
        df, x="Distance to TSS (bp)", y="Average score", title = score_name, 
    )
    return render_plot(fig, width, height, interactive, show, out_file)
