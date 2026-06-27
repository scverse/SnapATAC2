from __future__ import annotations

import numpy as np
import logging

import snapatac2
from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2.tools._misc import aggregate_X
from snapatac2._utils import find_elbow, is_anndata
from ._base import render_plot, heatmap, kde2d, scatter, scatter3d
from ._network import network_scores, network_edge_stat
import snapatac2._snapatac2 as internal

__all__ = [
    'tsse', 'frag_size_distr', 'umap', 'network_scores', 'spectral_eigenvalues',
    'regions', 'motif_enrichment', 'coverage'
]

def valid_cells(
    values,
    width: int = 500,
    height: int = 400,
    **kwargs,
):
    """Plot ranked barcode counts on log-log axes.

    Use this function to inspect the barcode rank curve before selecting a
    fragment-count cutoff for valid cells.

    Anti-Patterns
    -------------
    - Do NOT pass per-cell metadata tables. Pass a one-dimensional sequence of
      counts, such as fragment counts per barcode.
    - Do NOT use this function to filter cells. Use the plot to choose a cutoff,
      then apply filtering explicitly in preprocessing.

    Parameters
    ----------
    values : iterable of int or float
        Count values to rank in descending order.
    width : int
        Width of the rendered plot in pixels.
    height : int
        Height of the rendered plot in pixels.
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
    >>> import snapatac2 as snap
    >>> counts = [10000, 8500, 6200, 1200, 800, 120, 60, 20]
    >>> fig = snap.pl.valid_cells(counts, show=False)
    >>> fig.update_layout(title="Barcode rank curve")
    """
    import plotly.graph_objects as go

    values = sorted(values, reverse=True)
    result = {}
    for x, y in enumerate(values):
        x = x + 1
        if y in result:
            x_, n = result[y]
            result[y] = (x_ + x, n + 1)
        else:
            result[y] = (x, 1)
    for y, (x, n) in result.items():
        result[y] = x / n
    y, x = zip(*result.items())

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y))
    fig.update_xaxes(type="log")
    fig.update_yaxes(type="log")
    fig.update_layout(
        xaxis_title="Barcodes",
        yaxis_title="Counts",
    )

    return render_plot(fig, width, height, **kwargs)

def tsse(
    adata: AnnData,
    min_fragment: int = 500,
    width: int = 500,
    height: int = 400,
    **kwargs,
) -> 'plotly.graph_objects.Figure' | None:
    """Plot TSS enrichment against unique fragment counts.

    Use this function after computing TSS enrichment scores to assess cell
    quality and identify low-quality cells with low fragment counts or low TSS
    enrichment.

    Anti-Patterns
    -------------
    - Do NOT call this before running :func:`snapatac2.metrics.tsse`; the input
      must contain ``adata.obs["tsse"]``.
    - Do NOT interpret ``min_fragment`` as a filtering operation on ``adata``. It
      only excludes cells from this visualization.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing ``obs["tsse"]`` and
        ``obs["n_fragment"]``.
    min_fragment : int
        Minimum number of unique fragments required for a cell to be included in
        the plot.
    width : int
        Width of the rendered plot in pixels.
    height : int
        Height of the rendered plot in pixels.
    **kwargs
        Additional rendering options passed to :func:`snapatac2.pl.render_plot`,
        such as ``show``, ``interactive``, ``out_file``, and ``scale``.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns a Plotly figure when ``show=False`` and ``out_file=None``;
        otherwise renders or saves the plot and returns ``None``.

    See Also
    --------
    snapatac2.metrics.tsse : Compute TSS enrichment scores.
    render_plot : Render, show, or save Plotly figures.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.read(snap.datasets.pbmc5k(type="h5ad"))
    >>> snap.metrics.tsse(data, snap.genome.hg38)
    >>> fig = snap.pl.tsse(data, show=False)
    >>> fig.update_layout(title="TSS enrichment")
    """
    if "tsse" not in adata.obs:
        raise ValueError("TSS enrichment score is not computed, please run `metrics.tsse` first.")

    selected_cells = np.where(adata.obs["n_fragment"] >= min_fragment)[0]
    x = adata.obs["n_fragment"].to_numpy()[selected_cells]
    y = adata.obs["tsse"].to_numpy()[selected_cells]

    fig = kde2d(x, y, log_x=True, log_y=False)
    fig.update_layout(
        xaxis_title="Number of unique fragments",
        yaxis_title="TSS enrichment score",
    )

    return render_plot(fig, width, height, **kwargs)

def frag_size_distr(
    adata: AnnData | np.ndarray,
    use_rep: str = "frag_size_distr",
    max_recorded_size: int = 1000,
    **kwargs,
) -> 'plotly.graph_objects.Figure' | None:
    """Plot the fragment size distribution.

    Use this function to inspect nucleosome banding from either an AnnData
    object or a precomputed one-dimensional fragment-size count array.

    Anti-Patterns
    -------------
    - Do NOT pass raw fragment coordinates. Pass an AnnData object or a vector
      whose index is fragment size and whose value is the count for that size.
    - Do NOT expect this function to preserve an incomplete cached distribution;
      it recomputes ``adata.uns[use_rep]`` when the stored vector is too short.

    Parameters
    ----------
    adata : AnnData or numpy.ndarray
        Annotated data matrix with fragment-size information, or a precomputed
        fragment-size distribution vector.
    use_rep : str
        Key in ``adata.uns`` used to read or store the fragment-size
        distribution when ``adata`` is an AnnData object.
    max_recorded_size : int
        Maximum fragment size, in base pairs, to compute and display.
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
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> distribution = np.array([0, 5, 12, 18, 9, 3])
    >>> fig = snap.pl.frag_size_distr(distribution, show=False)
    >>> fig.update_layout(title="Fragment size distribution")
    """
    import plotly.graph_objects as go

    if is_anndata(adata):
        if use_rep not in adata.uns or len(adata.uns[use_rep]) <= max_recorded_size:
            logging.info("Computing fragment size distribution...")
            snapatac2.metrics.frag_size_distr(adata, add_key=use_rep, max_recorded_size=max_recorded_size)
        data = adata.uns[use_rep]
    else:
        data = adata
    data = data[:max_recorded_size+1]

    x, y = zip(*enumerate(data))
    # Make a line plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x[1:], y=y[1:], mode='lines'))
    fig.update_layout(
        xaxis_title="Fragment size",
        yaxis_title="Count",
    )
    return render_plot(fig, **kwargs)

def spectral_eigenvalues(
    adata: AnnData,
    width: int = 600,
    height: int = 400,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
) -> 'plotly.graph_objects.Figure' | None:
    """Plot spectral embedding eigenvalues and mark the elbow.

    Use this function after spectral decomposition to choose the number of
    eigenvectors retained for downstream analysis.

    Anti-Patterns
    -------------
    - Do NOT call this before computing spectral eigenvalues. The input must
      contain ``adata.uns["spectral_eigenvalue"]``.
    - Do NOT treat this as a pure plotting helper; it also writes the inferred
      elbow to ``adata.uns["num_eigen"]``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix containing ``uns["spectral_eigenvalue"]``.
    width : int
        Width of the rendered plot in pixels.
    height : int
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
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> adata = snap.AnnData(X=np.ones((3, 3)))
    >>> adata.uns["spectral_eigenvalue"] = np.array([4.0, 2.5, 1.2, 0.4])
    >>> fig = snap.pl.spectral_eigenvalues(adata, show=False)
    >>> fig.update_layout(title="Spectral eigenvalues")
    """
 
    import plotly.express as px
    import pandas as pd

    data = adata.uns["spectral_eigenvalue"]

    df = pd.DataFrame({"Component": map(str, range(1, data.shape[0] + 1)), "Eigenvalue": data})
    fig = px.scatter(df, x="Component", y="Eigenvalue", template="plotly_white")
    n = find_elbow(data)
    adata.uns["num_eigen"] = n
    fig.add_vline(x=n)

    return render_plot(fig, width, height, interactive, show, out_file)

def regions(
    adata: AnnData | AnnDataSet,
    groupby: str | list[str],
    peaks: dict[str, list[str]],
    width: float = 600,
    height: float = 400,
    show: bool = True,
    interactive: bool = True,
    out_file: str | None = None,
) -> 'plotly.graph_objects.Figure' | None:
    """Plot grouped accessibility over selected peak regions.

    Use this function to compare normalized accessibility across groups for a
    supplied peak set.

    Anti-Patterns
    -------------
    - Do NOT pass peaks that are absent from ``adata.var_names``; each peak name
      must map to a variable in the input matrix.
    - Do NOT use this function for very large peak sets when exact display is
      required. Inputs above 50,000 peaks are randomly downsampled for plotting.

    Parameters
    ----------
    adata : AnnData or AnnDataSet
        Annotated data matrix with peaks in ``var_names``.
    groupby : str or list of str
        Cell grouping definition. If a string, groups are read from
        ``adata.obs[groupby]``.
    peaks : dict of (str, list of str)
        Mapping from group names to peak names to include in the heatmap.
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
    >>> import snapatac2 as snap
    >>> adata = snap.read(snap.datasets.pbmc5k(type="h5ad"))
    >>> peaks = {"selected": list(adata.var_names[:20])}
    >>> fig = snap.pl.regions(adata, groupby="cell_type", peaks=peaks, show=False)
    >>> fig.update_layout(title="Grouped accessibility")
    """
    import polars as pl
    import plotly.graph_objects as go

    peaks = np.concatenate([[x for x in p] for p in peaks.values()])
    n = len(peaks)
    if n > 50000:
        logging.warning(f"Input contains {n} peaks, only 50000 peaks will be plotted.")
        np.random.seed(0)
        indices = np.random.choice(n, 50000, replace=False)
        peaks = peaks[sorted(indices)]

    count = aggregate_X(adata, groupby=groupby, normalize="RPKM")
    names = count.obs_names
    count = pl.DataFrame(count.X.T)
    count.columns = list(names)
    idx_map = {x: i for i, x in enumerate(adata.var_names)}
    idx = [idx_map[x] for x in peaks]
    mat = np.log2(1 + count.to_numpy()[idx, :])

    trace = go.Heatmap(
        x=count.columns,
        y=peaks,
        z=mat,
        type='heatmap',
        colorscale='Viridis',
        colorbar={ "title": "log2(1 + RPKM)" },
    )
    data = [trace]
    layout = {
        "yaxis": { "visible": False, "autorange": "reversed" },
        "xaxis": { "title": groupby },
    }
    fig = go.Figure(data=data, layout=layout)
    return render_plot(fig, width, height, interactive, show, out_file)

def umap(
    adata: AnnData | np.ndarray,
    color: str | np.ndarray | None = None,
    use_rep: str = "X_umap",
    marker_size: float = None,
    marker_opacity: float = 1,
    sample_size: int | None = None,
    **kwargs,
) -> 'plotly.graph_objects.Figure' | None:
    """Plot a two- or three-dimensional UMAP embedding.

    Use this function to visualize cells from ``adata.obsm[use_rep]`` or from a
    numeric embedding array.

    Anti-Patterns
    -------------
    - Do NOT pass ``color`` as a column name when ``adata`` is a raw NumPy array;
      provide an array of color values instead.
    - Do NOT use ``sample_size`` when every point must be displayed. Sampling is
      random and only affects the plotted points.

    Parameters
    ----------
    adata : AnnData or numpy.ndarray
        Annotated data matrix containing ``obsm[use_rep]``, or an embedding array
        with cells as rows and coordinates as columns.
    color : str, numpy.ndarray, or None
        Observation column name to color by when ``adata`` is AnnData, or a
        vector of color values aligned to the embedding rows.
    use_rep : str
        Key in ``adata.obsm`` containing the UMAP coordinates.
    marker_size : float or None
        Marker size. If ``None``, choose a size from the number of plotted cells.
    marker_opacity : float
        Marker opacity between 0 and 1.
    sample_size : int or None
        Maximum number of cells to plot. If the embedding has more rows,
        randomly sample this many rows without replacement.
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
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> embedding = np.array([[0.0, 0.1], [1.0, 1.1], [2.0, 0.9]])
    >>> labels = np.array(["A", "B", "A"])
    >>> fig = snap.pl.umap(embedding, color=labels, show=False)
    >>> fig.update_layout(title="UMAP")
    """
    from natsort import index_natsorted

    embedding = adata.obsm[use_rep] if is_anndata(adata) else adata
    if isinstance(color, str):
        groups = adata.obs[color].to_numpy()
    else:
        groups = color
        color = "color"
    
    if sample_size is not None and embedding.shape[0] > sample_size:
        idx = np.random.choice(embedding.shape[0], sample_size, replace=False)
        embedding = embedding[idx, :]
        if groups is not None: groups = groups[idx]

    if groups is not None:
        idx = index_natsorted(groups)
        embedding = embedding[idx, :]
        groups = [groups[i] for i in idx]

    if marker_size is None:
        num_points = embedding.shape[0]
        marker_size = (1000 / num_points)**(1/3) * 3

    if embedding.shape[1] >= 3:
        return scatter3d(embedding[:, 0], embedding[:, 1], embedding[:, 2], color=groups,
            x_label="UMAP-1", y_label="UMAP-2", z_label="UMAP-3", color_label=color,
            marker_size=marker_size, marker_opacity=marker_opacity, **kwargs)
    else:
        return scatter(embedding[:, 0], embedding[:, 1], color=groups,
            x_label="UMAP-1", y_label="UMAP-2", color_label=color,
            marker_size=marker_size, marker_opacity=marker_opacity, **kwargs)

def motif_enrichment(
    enrichment: list(str, 'pl.DataFrame'),
    min_log_fc: float = 1,
    max_fdr: float = 0.01,
    **kwargs,
) -> 'plotly.graph_objects.Figure' | None:
    """Plot motif enrichment scores across groups.

    Use this function to summarize motif enrichment tables returned for multiple
    groups as a clustered heatmap.

    Anti-Patterns
    -------------
    - Do NOT pass a single enrichment table. Pass a mapping from group names to
      Polars DataFrames with matching motif rows.
    - Do NOT rename required columns. Each table must contain ``id``,
      ``log2(fold change)``, ``adjusted p-value``, and ``p-value``.

    Parameters
    ----------
    enrichment : dict of (str, polars.DataFrame)
        Mapping from group names to motif enrichment result tables. Tables must
        have aligned rows and include required motif statistics columns.
    min_log_fc : float
        Keep motifs with at least one absolute log2 fold-change greater than or
        equal to this value.
    max_fdr : float
        Keep motifs with at least one adjusted p-value less than or equal to this
        value.
    **kwargs
        Additional rendering options passed to :func:`snapatac2.pl.heatmap` and
        :func:`snapatac2.pl.render_plot`, such as ``show``, ``interactive``,
        ``out_file``, and clustering options accepted by ``heatmap``.

    Returns
    -------
    plotly.graph_objects.Figure or None
        Returns a Plotly figure when ``show=False`` and ``out_file=None``;
        otherwise renders or saves the plot and returns ``None``.

    Examples
    --------
    >>> import polars as pl
    >>> import snapatac2 as snap
    >>> table = pl.DataFrame({
    ...     "id": ["MA0001.1", "MA0002.1"],
    ...     "log2(fold change)": [1.5, -0.8],
    ...     "adjusted p-value": [0.001, 0.2],
    ...     "p-value": [1e-5, 0.05],
    ... })
    >>> enrichment = {"cluster_1": table, "cluster_2": table}
    >>> fig = snap.pl.motif_enrichment(enrichment, show=False)
    >>> fig.update_layout(title="Motif enrichment")
    """
 
    import pandas as pd
    
    fc = np.vstack([df['log2(fold change)'] for df in enrichment.values()])
    filter1 = np.apply_along_axis(lambda x: np.any(np.abs(x) >= min_log_fc), 0, fc)
    
    fdr = np.vstack([df['adjusted p-value'] for df in enrichment.values()])
    filter2 = np.apply_along_axis(lambda x: np.any(x <= max_fdr), 0, fdr)

    passed = np.logical_and(filter1, filter2)
    
    sign = np.sign(fc[:, passed])
    pvals = np.vstack([df['p-value'].to_numpy()[passed] for df in enrichment.values()])
    minval = np.min(pvals[np.nonzero(pvals)])
    pvals = np.clip(pvals, minval, None)
    pvals = sign * np.log(-np.log10(pvals))

    df = pd.DataFrame(
        pvals.T,
        columns=list(enrichment.keys()),
        index=next(iter(enrichment.values()))['id'].to_numpy()[passed],
    )
      
    return heatmap(
        df.to_numpy(),
        row_names=df.index,
        column_names=df.columns,
        colorscale='RdBu_r',
        **kwargs,
    )

def coverage(
    adata: AnnData,
    region: str,
    groupby: str | list[str],
    out_file: str | None = None,
):
    """Plot coverage tracks for grouped cells across one genomic region.

    Use this function for quick local inspection of coverage patterns. Install
    ``matplotlib`` before calling it.

    This is a simple implementation for quick visualization. For more advanced
    visualization, export the data with :func:`snapatac2.ex.export_coverage` and
    inspect it in a genome browser.

    Anti-Patterns
    -------------
    - Do NOT use this function for publication-scale genome browser tracks;
      export coverage files instead.
    - Do NOT pass multiple genomic intervals. ``region`` must be a single string
      formatted as ``"chrom:start-end"``.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix with fragments available for coverage retrieval.
    region : str
        Genomic interval to plot, formatted as ``"chrom:start-end"``; for
        example, ``"chr1:100000-200000"``.
    groupby : str or list of str
        Cell grouping definition. If a string, groups are read from
        ``adata.obs[groupby]``.
    out_file : str or None
        Output path for saving the Matplotlib figure. If ``None``, display the
        plot with ``matplotlib.pyplot.show``.

    Returns
    -------
    None
        Displays or saves the Matplotlib figure.

    See Also
    --------
    snapatac2.ex.export_coverage : Export coverage tracks for genome browsers.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.read(snap.datasets.pbmc5k(type="h5ad"))
    >>> snap.pl.coverage(
    ...     adata,
    ...     region="chr1:100000-200000",
    ...     groupby="cell_type",
    ...     out_file="coverage.png",
    ... )
    """

    from matplotlib import pyplot as plt

    groupby = adata.obs[groupby] if isinstance(groupby, str) else groupby
    groupby = [x for x in groupby]
    signal_values = []
    track_names = []
    for k, v in sorted(list(internal.get_coverage(adata, region, groupby).items())):
        track_names.append(k)
        signal_values.append(v)
    signal_values = np.array(signal_values)

    start, end = region.split(":")[1].split("-")
    start = int(start)
    end = int(end)
    height_per_track = 1.2
    width = 6

    n_tracks, n_points = signal_values.shape
    fig, axes = plt.subplots(
        n_tracks, 1,
        figsize=[width, n_tracks * height_per_track],
        sharex=True,
        constrained_layout=True,
    )

    if n_tracks == 1:
        axes = [axes]

    # Compute global max for y-axis scaling
    global_max = signal_values.max()

    cmap = plt.get_cmap("tab10")
    for i, (ax, signal) in enumerate(zip(axes, signal_values)):
        color = cmap(i % 10)   # cycle through colors if >10 tracks
        ax.fill_between(range(n_points), 0, signal, color=color)
        ax.set_title(track_names[i], fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, global_max)

    axes[-1].set_xticks([0, n_points - 1])
    axes[-1].set_xticklabels([str(start), str(end)])
    axes[-1].set_xlabel(region)

    fig.supylabel("RPM")

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
