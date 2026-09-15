from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

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

def _parse_gtf_attributes(attr_str: str) -> dict[str, str]:
    """Parse a GTF ``key "value";`` attribute string."""
    attrs = {}
    for part in attr_str.rstrip(";").split(";"):
        part = part.strip()
        if not part:
            continue
        if '"' in part:
            key, val = part.split('"', 1)
            attrs[key.strip()] = val.strip('"').strip()
    return attrs


def _open_annotation(f: str | Path):
    import gzip
    path = Path(f)
    return gzip.open(path, "rt") if path.suffix == ".gz" else open(path, "r")


def _lookup_gene_region(annotation: str | Path, gene_name: str) -> tuple[str, int, int] | None:
    """Look up *(chrom, start, end)* of *gene_name* in a GTF file."""
    for seqid, feat_type, feat_start, feat_end, _strand, attrs in _iter_annotation(annotation):
        if feat_type != "gene":
            continue
        if attrs.get("gene_name") == gene_name:
            return (seqid, feat_start, feat_end)
    return None


def _gen_to_pixel(pos: int, region_start: int, region_end: int, n_points: int) -> float:
    """Map a genomic coordinate to the x-pixel range [0, n_points-1]."""
    return (pos - region_start) / (region_end - region_start) * (n_points - 1)


def _style_annotation_axis(ax, label: str):
    """Strip spines/ticks from *ax* and set its ylabel."""
    ax.set_yticks([])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False, labelleft=False)
    ax.set_xticks([])
    ax.set_ylabel(label, fontsize=7, labelpad=2)


def _greedy_row_layout(intervals: list[tuple[float, float]]) -> list[int]:
    """Assign non-overlapping rows to a list of *(x1, x2)* pixel intervals.
    Returns row indices (same order as *intervals*).
    """
    row_ends: list[float] = []
    placements: list[int] = []
    for x1, x2 in intervals:
        row = 0
        while row < len(row_ends) and x1 < row_ends[row]:
            row += 1
        if row == len(row_ends):
            row_ends.append(x2)
        else:
            row_ends[row] = max(row_ends[row], x2)
        placements.append(row)
    return placements


def _iter_annotation(
    annotation: str | Path,
    chrom: str | None = None,
    start: int | None = None,
    end: int | None = None,
):
    """Yield ``(seqid, feat_type, feat_start, feat_end, strand, attrs)``
    from a coordinate-sorted GTF file, optionally filtered by region."""
    found_chrom = False
    with _open_annotation(annotation) as fh:
        for line in fh:
            if line.startswith("#") or line.strip() == "":
                continue
            cols = line.strip().split("\t")
            if len(cols) < 9:
                continue
            seqid, _source, feat_type, feat_start, feat_end, _score, strand, _phase, attrs_str = cols
            if chrom is not None and seqid != chrom:
                if found_chrom:
                    break
                continue
            found_chrom = True
            feat_start = int(feat_start)
            feat_end = int(feat_end)
            if start is not None and feat_end < start:
                continue
            if end is not None and feat_start > end:
                break
            yield (seqid, feat_type, feat_start, feat_end, strand,
                   _parse_gtf_attributes(attrs_str))


def _normalize_gene_types(
    gene_type: str | list[str] | None,
) -> set[str] | None:
    """Normalize gene type selection for annotation track filtering."""
    if gene_type is None:
        return None
    if isinstance(gene_type, str):
        return {gene_type}
    return set(gene_type)


def _get_gene_models(
    annotation: str | Path,
    chrom: str,
    start: int,
    end: int,
    gene_types: set[str] | None = None,
) -> list[dict]:
    """Read gene/exon models overlapping *chrom:start-end* from a GTF file."""
    genes: dict[str, dict] = {}
    for seqid, feat_type, feat_start, feat_end, strand, attrs in _iter_annotation(
        annotation, chrom=chrom, start=start, end=end,
    ):
        gene_id = attrs.get("gene_id")
        if not gene_id:
            continue
        if gene_id not in genes:
            genes[gene_id] = {
                "name": attrs.get("gene_name", gene_id),
                "gene_type": attrs.get("gene_type", attrs.get("gene_biotype")),
                "strand": strand,
                "exons": [],
                "tx_start": feat_start,
                "tx_end": feat_end,
                "chrom": seqid,
            }
        gene = genes[gene_id]
        gene["tx_start"] = min(gene["tx_start"], feat_start)
        gene["tx_end"] = max(gene["tx_end"], feat_end)
        if feat_type == "exon":
            gene["exons"].append((feat_start, feat_end))

    result = []
    for gene in genes.values():
        if gene_types is not None and gene["gene_type"] not in gene_types:
            continue
        exons = _merge_intervals(gene["exons"])
        result.append({
            "name": gene["name"],
            "strand": gene["strand"],
            "tx_start": max(gene["tx_start"], start),
            "tx_end": min(gene["tx_end"], end),
            "exons": [(max(s, start), min(e, end)) for s, e in exons
                       if max(s, start) < min(e, end)],
        })
    return result


def _merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge duplicate and overlapping genomic intervals."""
    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(intervals)):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _draw_gene_track(
    ax,
    models: list[dict],
    region_start: int,
    region_end: int,
    n_points: int,
):
    """Draw gene models on *ax* spanning pixel range [0, n_points-1]."""
    import matplotlib.patches as mpatches

    if not models:
        _style_annotation_axis(ax, "Genes")
        return

    y_center = 0.5
    exon_height = 0.35

    intervals = [(_gen_to_pixel(g["tx_start"], region_start, region_end, n_points),
                  _gen_to_pixel(g["tx_end"], region_start, region_end, n_points))
                 for g in models]
    placements = _greedy_row_layout(intervals)

    n_rows = max(placements) + 1 if placements else 1
    ax.set_ylim(-0.3, n_rows + 0.3)

    for g, row, (gx1, gx2) in zip(models, placements, intervals):
        y_base = row + y_center
        ax.hlines(y_base, gx1, gx2, colors="0.2", linewidth=1.5, zorder=2)

        for ex_start, ex_end in g["exons"]:
            ex1 = _gen_to_pixel(ex_start, region_start, region_end, n_points)
            ex2 = _gen_to_pixel(ex_end, region_start, region_end, n_points)
            if ex2 - ex1 < 1:
                continue
            rect = mpatches.Rectangle(
                (ex1, y_base - exon_height / 2), ex2 - ex1, exon_height,
                facecolor="0.2", edgecolor="none", linewidth=0, zorder=3,
            )
            ax.add_patch(rect)

        if g["strand"] == "+":
            ax.annotate("▶", xy=(gx2, y_base), fontsize=5, ha="center",
                        va="center", color="0.2", zorder=4)
        elif g["strand"] == "-":
            ax.annotate("◀", xy=(gx1, y_base), fontsize=5, ha="center",
                        va="center", color="0.2", zorder=4)

        ax.text(gx1, y_base + exon_height / 2 + 0.06, g["name"],
                fontsize=5, ha="left", va="bottom", color="0.2", zorder=5)

    ax.set_xlim(0, n_points - 1)
    _style_annotation_axis(ax, "Genes")


def _add_highlights(
    axes: list,
    highlights: list[tuple[int, int, object]],
    region_start: int,
    region_end: int,
    n_points: int,
    alpha: float,
):
    """Overlay normalized highlighted regions across all *axes*."""
    if not 0 <= alpha <= 1:
        raise ValueError("highlight_alpha must be between 0 and 1.")
    for start, end, color in highlights:
        x1 = _gen_to_pixel(start, region_start, region_end, n_points)
        x2 = _gen_to_pixel(end, region_start, region_end, n_points)
        for ax in axes:
            ax.axvspan(x1, x2, color=color, alpha=alpha, zorder=0)


def _normalize_highlights(
    highlights: list,
    region_start: int,
    region_end: int,
) -> list[tuple[int, int, object]]:
    """Normalize and clip tuple- or dict-based highlight definitions."""
    result = []
    default_color = (1, 0.85, 0.6)
    for index, item in enumerate(highlights):
        if isinstance(item, dict):
            if "start" not in item or "end" not in item:
                raise ValueError(
                    f"highlight[{index}] must contain 'start' and 'end'."
                )
            start, end = item["start"], item["end"]
            color = item.get("color", default_color)
        elif isinstance(item, (list, tuple)) and len(item) in (2, 3):
            start, end = item[:2]
            color = item[2] if len(item) > 2 else default_color
        else:
            raise ValueError(
                f"highlight[{index}] must be a 2- or 3-item sequence or a dict."
            )

        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise ValueError(f"highlight[{index}] coordinates must be numeric.")
        if start >= end:
            raise ValueError(f"highlight[{index}] start must be smaller than end.")
        clipped_start = max(start, region_start)
        clipped_end = min(end, region_end)
        if clipped_start < clipped_end:
            result.append((clipped_start, clipped_end, color))
    return result


def _get_peaks_in_region(
    adata: AnnData,
    chrom: str,
    start: int,
    end: int,
    max_peaks: int = 2000,
) -> list[tuple[int, int, str]]:
    """Return peaks (var_names ``"chrN:start-end"``) overlapping *chrom:start-end*.

    Chromosome and coordinate order in ``adata.var_names`` is used to stop the
    scan as soon as it passes the requested region.
    """
    peaks: list[tuple[int, int, str]] = []
    prefix = f"{chrom}:"
    found_chrom = False
    for name in adata.var_names:
        if not name.startswith(prefix):
            if found_chrom:
                break
            continue
        found_chrom = True
        coord_part = name[len(prefix):]
        if "-" not in coord_part:
            continue
        try:
            ps, pe = coord_part.split("-", 1)
            ps = int(ps)
            pe = int(pe)
        except (ValueError, IndexError):
            continue
        if ps >= end:
            break
        if ps < end and pe > start:
            peaks.append((ps, pe, name))
            if len(peaks) >= max_peaks:
                break
    return peaks


def _draw_peak_track(
    ax,
    peaks: list[tuple[int, int, str]],
    region_start: int,
    region_end: int,
    n_points: int,
):
    """Draw overlapping peaks as horizontal bars in a compact track."""
    import matplotlib.patches as mpatches
    from matplotlib.colors import to_rgba

    if not peaks:
        _style_annotation_axis(ax, "Peaks")
        return

    bar_height = 0.25

    intervals = [(_gen_to_pixel(max(ps, region_start), region_start, region_end, n_points),
                  _gen_to_pixel(min(pe, region_end), region_start, region_end, n_points))
                 for ps, pe, _name in peaks]
    placements = _greedy_row_layout(intervals)

    n_rows = max(placements) + 1 if placements else 1
    ax.set_ylim(-0.3, n_rows + 0.3)
    ax.set_xlim(0, n_points - 1)

    for (ps, pe, _name), row, (px1, px2) in zip(peaks, placements, intervals):
        y_base = row + 0.5
        c = to_rgba("tab:red", alpha=0.7)
        rect = mpatches.Rectangle(
            (px1, y_base - bar_height / 2), max(px2 - px1, 1.0), bar_height,
            facecolor=c, edgecolor="none", linewidth=0, zorder=3,
        )
        ax.add_patch(rect)

    _style_annotation_axis(ax, "Peaks")


def _resolve_annotation_path(gene_annotation) -> Path | None:
    """Resolve a *gene_annotation* parameter to a Path, or ``None``."""
    if gene_annotation is None:
        return None
    if not isinstance(gene_annotation, (str, Path)):
        raise TypeError("gene_annotation must be a path to a GTF file.")
    return Path(gene_annotation)


def _parse_region(
    region: str,
    gene_annotation_path: Path | None,
) -> tuple[str, int, int, str]:
    """Parse *region* (``"chrom:start-end"`` or gene name) into
    ``(chrom, start, end, original_region_string)``.

    If *region* is a gene name and *gene_annotation_path* is provided,
    the gene coordinates are looked up from the annotation file.
    """
    if ":" in region and "-" in region:
        chrom, coord_part = region.split(":", 1)
        start_s, end_s = coord_part.split("-", 1)
        start, end = int(start_s), int(end_s)
        if start >= end:
            raise ValueError("Region start must be smaller than end.")
        return (chrom, start, end, region)
    if gene_annotation_path is not None:
        result = _lookup_gene_region(gene_annotation_path, region.strip())
        if result is not None:
            chrom, gs, ge = result
            return (chrom, gs, ge, region)
    raise ValueError(
        f"Could not parse region {region!r}. Format as 'chr:start-end' or "
        "provide gene_annotation for gene name lookup."
    )


def coverage(
    adata: AnnData,
    region: str,
    groupby: str | list[str],
    out_file: str | None = None,
    gene_annotation: str | Path | None = None,
    gene_type: str | list[str] | None = "protein_coding",
    highlight: list | None = None,
    highlight_alpha: float = 0.25,
    peak_track: bool = True,
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
        example, ``"chr1:100000-200000"``. If ``gene_annotation`` is provided
        and *region* does not match that format, it is treated as a gene name
        and the gene's coordinates are looked up from the annotation file.
    groupby : str or list of str
        Cell grouping definition. If a string, groups are read from
        ``adata.obs[groupby]``.
    out_file : str or None
        Output path for saving the Matplotlib figure. If ``None``, display the
        plot with ``matplotlib.pyplot.show``.
    gene_annotation : str or pathlib.Path, optional
        Path to a coordinate-sorted GTF annotation file. When provided, a gene
        annotation track is drawn below the coverage tracks showing gene
        introns, exons, and strand direction. Additionally, *region* can be
        given as a gene name (e.g. ``"GENE_A"``) to automatically resolve the
        coordinates.
    gene_type : str, list of str, or None
        Gene types shown in the annotation track. Defaults to
        ``"protein_coding"``. Pass multiple types as a list, or ``None`` to
        show all annotated gene types. This does not affect gene-name lookup.
    highlight : list, optional
        Regions to highlight across coverage tracks. Each element can be:

        * ``(start, end)`` — highlighted in pale orange.
        * ``(start, end, color)`` — highlighted in a custom color (any
          matplotlib-compatible color or RGBA tuple).
        * ``{"start": s, "end": e, "color": c}`` — dict form.

        Coordinates are genomic positions within the plotted interval.
    highlight_alpha : float
        Opacity of highlighted regions, between 0 (transparent) and 1 (opaque).
    peak_track : bool
        If ``True``, draw a peak annotation track at the top showing the
        overlapping peak regions from ``adata.var_names``.

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

    annotation_path = _resolve_annotation_path(gene_annotation)
    gene_types = _normalize_gene_types(gene_type)
    chrom, region_start, region_end, resolved_region = _parse_region(region, annotation_path)
    coverage_region_str = f"{chrom}:{region_start}-{region_end}"
    if not 0 <= highlight_alpha <= 1:
        raise ValueError("highlight_alpha must be between 0 and 1.")

    groupby = adata.obs[groupby] if isinstance(groupby, str) else groupby
    groupby = [x for x in groupby]
    signal_values = []
    track_names = []
    for k, v in sorted(list(internal.get_coverage(adata, coverage_region_str, groupby).items())):
        track_names.append(k)
        signal_values.append(v)
    signal_values = np.array(signal_values)
    height_per_track = 1.2
    width = 6

    n_tracks, n_points = signal_values.shape

    n_rows = n_tracks + int(peak_track) + int(annotation_path is not None)

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=[width, n_rows * height_per_track],
        sharex=True, constrained_layout=True,
    )
    if n_rows == 1:
        axes = [axes]
    elif not isinstance(axes, (list, tuple)):
        axes = list(axes)

    global_max = signal_values.max()

    coverage_start = int(peak_track)
    coverage_axes = axes[coverage_start:coverage_start + n_tracks]
    if peak_track:
        peaks = _get_peaks_in_region(adata, chrom, region_start, region_end)
        _draw_peak_track(axes[0], peaks, region_start, region_end, n_points)

    cmap = plt.get_cmap("tab10")
    for i, (ax, signal) in enumerate(zip(coverage_axes, signal_values)):
        color = cmap(i % 10)
        ax.fill_between(range(n_points), 0, signal, color=color)
        ax.set_title(track_names[i], fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.set_ylim(0, global_max)

    if annotation_path is not None:
        models = _get_gene_models(
            annotation_path, chrom, region_start, region_end, gene_types,
        )
        _draw_gene_track(axes[-1], models, region_start, region_end, n_points)

    for ax in axes[:-1]:
        ax.set_xticks([])
        ax.set_xlabel("")
    bottom_ax = axes[-1]
    bottom_ax.set_xticks([0, n_points - 1])
    bottom_ax.set_xticklabels([str(region_start), str(region_end)])
    bottom_ax.set_xlabel(resolved_region)

    if highlight is not None:
        highlights = _normalize_highlights(highlight, region_start, region_end)
        _add_highlights(
            coverage_axes, highlights, region_start, region_end, n_points,
            alpha=highlight_alpha,
        )

    fig.supylabel("RPM")

    if out_file is None:
        plt.show()
    else:
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
