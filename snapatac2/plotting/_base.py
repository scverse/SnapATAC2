from __future__ import annotations

import numpy as np
from scipy.stats import gaussian_kde

def scatter(
    X: list[float],
    Y: list[float],
    color: list[float] | None = None,
    x_label: str = "X",
    y_label: str = "Y",
    color_label: str = "Color",
    marker_size: float = 2,
    marker_opacity: float = 0.5,
    **kwargs,
) -> 'plotly.graph_objects.Figure' | None:
    """Plot a two-dimensional scatter plot.

    Use this helper to build a Plotly scatter plot from aligned coordinate
    vectors and optional categorical or continuous color values.

    Anti-Patterns
    -------------
    - Do NOT pass coordinate vectors with different lengths. ``X``, ``Y``, and
      ``color`` must describe the same points.
    - Do NOT call Plotly rendering methods directly when using SnapATAC2 plotting
      functions. Pass rendering options through ``**kwargs`` instead.

    Parameters
    ----------
    X : list of float or numpy.ndarray
        X coordinates of the points.
    Y : list of float or numpy.ndarray
        Y coordinates of the points.
    color : list, numpy.ndarray, or None
        Values used to color points. If ``None``, draw all points with one color.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    color_label : str
        Label for the legend or color scale when ``color`` is provided.
    marker_size : float
        Marker size in pixels.
    marker_opacity : float
        Marker opacity between 0 and 1.
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
    >>> fig = snap.pl.scatter(
    ...     [0.0, 1.0, 2.0],
    ...     [0.2, 0.8, 1.4],
    ...     color=["A", "B", "A"],
    ...     show=False,
    ... )
    >>> fig.update_layout(title="Scatter plot")
    """
    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame({
        x_label: X,
        y_label: Y,
    })
    if color is None:
        color_label = None
    else:
        df[color_label] = color

    fig = px.scatter(
        df, x=x_label, y=y_label, color=color_label,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_traces(
        marker_size=marker_size,
        marker={"opacity": marker_opacity},
    )
    fig.update_layout(
        template="simple_white",
        legend={'itemsizing': 'constant'},
    )
    return render_plot(fig, **kwargs)

def scatter3d(
    X: list[float],
    Y: list[float],
    Z: list[float],
    color: list[float] | None = None,
    x_label: str = "X",
    y_label: str = "Y",
    z_label: str = "Z",
    color_label: str = "Color",
    marker_size: float = 2,
    marker_opacity: float = 0.5,
    **kwargs,
) -> 'plotly.graph_objects.Figure' | None:
    """Plot a three-dimensional scatter plot.

    Use this helper to build a Plotly 3D scatter plot from aligned coordinate
    vectors and optional categorical or continuous color values.

    Anti-Patterns
    -------------
    - Do NOT pass coordinate vectors with different lengths. ``X``, ``Y``,
      ``Z``, and ``color`` must describe the same points.
    - Do NOT pass two-dimensional embeddings here. Use :func:`snapatac2.pl.scatter`
      for 2D coordinates.

    Parameters
    ----------
    X : list of float or numpy.ndarray
        X coordinates of the points.
    Y : list of float or numpy.ndarray
        Y coordinates of the points.
    Z : list of float or numpy.ndarray
        Z coordinates of the points.
    color : list, numpy.ndarray, or None
        Values used to color points. If ``None``, draw all points with one color.
    x_label : str
        X-axis label.
    y_label : str
        Y-axis label.
    z_label : str
        Z-axis label.
    color_label : str
        Label for the legend or color scale when ``color`` is provided.
    marker_size : float
        Marker size in pixels.
    marker_opacity : float
        Marker opacity between 0 and 1.
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
    >>> fig = snap.pl.scatter3d(
    ...     [0.0, 1.0, 2.0],
    ...     [0.2, 0.8, 1.4],
    ...     [1.0, 0.5, 0.0],
    ...     color=["A", "B", "A"],
    ...     show=False,
    ... )
    >>> fig.update_layout(title="3D scatter plot")
    """
    import plotly.express as px
    import pandas as pd

    df = pd.DataFrame({
        x_label: X,
        y_label: Y,
        z_label: Z,
    })
    if color is not None: df[color_label] = color

    fig = px.scatter_3d(
        df, x=x_label, y=y_label, z=z_label, color=color_label,
        color_discrete_sequence=px.colors.qualitative.Dark24,
    )
    fig.update_traces(
        marker_size=marker_size,
        marker={"opacity": marker_opacity},
    )
    fig.update_layout(
        template="simple_white",
        legend={'itemsizing': 'constant'},
    )
    return render_plot(fig, **kwargs)


def heatmap(
    data_array: np.ndarray,
    row_names: list[str] | None = None,
    column_names: list[str] | None = None,
    cluster_columns: bool = True,
    cluster_rows: bool = True,
    colorscale = "Blues",
    linkage: str = "ward",
    **kwargs,
):
    """Plot a heatmap with optional row and column dendrograms.

    Use this helper to visualize a numeric matrix and, by default, cluster both
    dimensions before rendering.

    Anti-Patterns
    -------------
    - Do NOT pass non-numeric arrays. The clustering and heatmap layers require
      numeric values.
    - Do NOT enable clustering when a dimension has too few observations for the
      selected linkage method; disable ``cluster_rows`` or ``cluster_columns`` in
      that case.

    Parameters
    ----------
    data_array : numpy.ndarray
        Two-dimensional numeric matrix to display.
    row_names : list of str or None
        Labels for matrix rows. If provided, length must match the number of
        rows in ``data_array``.
    column_names : list of str or None
        Labels for matrix columns. If provided, length must match the number of
        columns in ``data_array``.
    cluster_columns : bool
        Whether to hierarchically cluster columns before plotting.
    cluster_rows : bool
        Whether to hierarchically cluster rows before plotting.
    colorscale : str
        Plotly colorscale used for heatmap values.
    linkage : str
        Linkage method passed to :func:`scipy.cluster.hierarchy.linkage`.
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
    >>> matrix = np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 4.0], [3.0, 4.0, 1.0]])
    >>> fig = snap.pl.heatmap(
    ...     matrix,
    ...     row_names=["r1", "r2", "r3"],
    ...     column_names=["c1", "c2", "c3"],
    ...     show=False,
    ... )
    >>> fig.update_layout(title="Clustered heatmap")
    """
    import plotly.graph_objects as go
    import plotly.figure_factory as ff
    import scipy.cluster.hierarchy as sch
    
    link_func = lambda x: sch.linkage(x, linkage)
    fig = go.Figure()
    
    if cluster_columns:
        dendro_upper = ff.create_dendrogram(data_array.T, linkagefun=link_func, orientation='bottom')
        upper_leaves = list(map(int, dendro_upper['layout']['xaxis']['ticktext']))
        data_array = data_array[:, upper_leaves]
        if column_names is not None:
            dendro_upper['layout']['xaxis']['ticktext'] = np.array(column_names)[upper_leaves]
        for i in range(len(dendro_upper['data'])):
            dendro_upper['data'][i]['yaxis'] = 'y2'
            fig.add_trace(dendro_upper['data'][i])
        fig['layout'] = dendro_upper['layout']

    if cluster_rows:
        dendro_side = ff.create_dendrogram(data_array, linkagefun=link_func, orientation='right')
        side_leaves = list(map(int, dendro_side['layout']['yaxis']['ticktext']))
        data_array = data_array[side_leaves, :]
        if row_names is not None:
            dendro_side['layout']['yaxis']['ticktext'] = np.array(row_names)[side_leaves]
        for i in range(len(dendro_side['data'])):
            dendro_side['data'][i]['xaxis'] = 'x2'
            fig.add_trace(dendro_side['data'][i])
        fig['layout']['yaxis'] = dendro_side['layout']['yaxis']

    # Create Heatmap
    heatmap = [go.Heatmap(
        z=data_array,
        colorscale=colorscale,
        colorbar={'orientation': 'h', 'title': 'log(-log(P))'}
    )]
    if cluster_columns:
        heatmap[0]['x'] = dendro_upper['layout']['xaxis']['tickvals']
    if cluster_rows:
        heatmap[0]['y'] = dendro_side['layout']['yaxis']['tickvals']
    for data in heatmap:
        fig.add_trace(data)
        
    fig.update_layout({'showlegend':False, 'hovermode': 'closest'})
    
    if cluster_rows:
        fig.update_layout(
            xaxis={
                'domain': [.15, 1], 'mirror': False, 'showgrid': False,
                'showline': False, 'zeroline': False, 'ticks':"",
            },
            xaxis2={
                'domain': [0, .15], 'mirror': False, 'showgrid': False,
                'showline': False, 'zeroline': False, 'showticklabels': False,
                'ticks':"",
            },
        )
    if cluster_columns:
        fig.update_layout(
            yaxis={
                'domain': [0, .85], 'mirror': False, 'showgrid': False,
                'showline': False, 'zeroline': False, 'showticklabels': True,
                'side': 'right', 'ticks': ""
            },
            yaxis2={
                'domain':[.825, .975], 'mirror': False, 'showgrid': False,
                'showline': False, 'zeroline': False, 'showticklabels': False,
                'ticks':""
            },
        )
    return render_plot(fig, **kwargs)

def render_plot(
    fig: 'plotly.graph_objects.Figure',
    width: int = 600,
    height: int = 400,
    interactive: bool = True,
    show: bool = True,
    out_file: str | None = None,
    scale: float | None = None,
) -> 'plotly.graph_objects.Figure' | None:
    """Render, display, or save a Plotly figure.

    Use this function to apply SnapATAC2 plotting output rules to an existing
    Plotly figure. Most users should call higher-level plotting functions, which
    call this helper internally.

    Anti-Patterns
    -------------
    - Do NOT expect a figure object when ``show=True`` or ``out_file`` is set;
      the function returns a figure only for ``show=False`` and ``out_file=None``.
    - Do NOT save static image formats without the Plotly image export backend
      installed. Use ``.html`` for a dependency-light interactive file.

    Parameters
    ----------
    fig : plotly.graph_objects.Figure
        Plotly figure to render.
    width : int
        Width of the rendered plot in pixels.
    height : int
        Height of the rendered plot in pixels.
    interactive : bool
        Whether to display an interactive Plotly figure when ``show=True``. If
        ``False``, return an IPython PNG image object for display.
    show : bool
        Whether to display the figure immediately.
    out_file : str or None
        Output path for saving the plot. Supported suffixes include ``.svg``,
        ``.pdf``, ``.png``, and ``.html``.
    scale : float or None
        Scale factor for static image export. Use values greater than 1 to
        increase resolution.

    Returns
    -------
    plotly.graph_objects.Figure, IPython.display.Image, or None
        Returns the Plotly figure when ``show=False`` and ``out_file=None``;
        returns an IPython image when ``show=True`` and ``interactive=False``;
        otherwise renders or saves the plot and returns ``None``.

    Examples
    --------
    >>> import plotly.graph_objects as go
    >>> import snapatac2 as snap
    >>> fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[1, 3, 2]))
    >>> fig = snap.pl.render_plot(fig, width=500, height=300, show=False)
    >>> fig.update_layout(title="Rendered plot")
    """

    fig.update_layout({
        "width": width,
        "height": height,
    })

    # save figure to file
    if out_file is not None:
        if str(out_file).endswith(".html"):
            fig.write_html(out_file, include_plotlyjs="cdn")
        else:
            fig.write_image(out_file, scale=scale)

    # show figure
    if show:
        if interactive:
            fig.show()
        else:
            from IPython.display import Image
            return Image(fig.to_image(format="png"))

    # return plot object
    if not show and not out_file: return fig

def kde2d(
    x: np.ndarray,
    y: np.ndarray,
    log_x: bool = False,
    log_y: bool = False,
) -> 'plotly.graph_objects.Figure' | None:
    """Estimate and plot a two-dimensional kernel density surface.

    Use this helper to convert paired observations into a Plotly contour plot,
    optionally estimating density in log10-transformed coordinate space.

    Anti-Patterns
    -------------
    - Do NOT pass zeros or negative values when ``log_x=True`` or ``log_y=True``.
      Log-transformed axes require strictly positive coordinates.
    - Do NOT use this helper for pre-binned matrices. Pass raw paired
      observations instead.

    Parameters
    ----------
    x : numpy.ndarray
        X coordinates of the observations.
    y : numpy.ndarray
        Y coordinates of the observations. Must have the same length as ``x``.
    log_x : bool
        Whether to estimate density on ``log10(x)`` and display a log-scaled
        x-axis.
    log_y : bool
        Whether to estimate density on ``log10(y)`` and display a log-scaled
        y-axis.

    Returns
    -------
    plotly.graph_objects.Figure
        Plotly contour figure containing the estimated density levels.

    Examples
    --------
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> y = np.array([2.0, 2.5, 3.5, 3.8, 5.0])
    >>> fig = snap.pl.kde2d(x, y)
    >>> fig.update_layout(title="2D density")
    """
    import plotly.graph_objects as go
    import plotly.express as px

    if log_x:
        x = np.log10(x)
    if log_y:
        y = np.log10(y)

    '''
    (z, X, Y) = np.histogram2d(x,y, bins=(bins_x, bins_y), density=True)
    z = z.T
    quintiles = np.percentile(np.ravel(z[z>0]), [1,10,20,30,40,50,60,70,80,90])
    z = np.searchsorted(quintiles, np.ravel(z)).reshape(z.shape)
    '''

    estimator = KDE()
    z, (xx, yy) = estimator(x, y)

    levels = _quantile_to_level(z.ravel(), np.linspace(.05, 1, 10))
    z = np.searchsorted(levels, z) + 0.1

    #quintiles = np.percentile(np.ravel(z[z>0]), [5,10,20,30,40,50,60,70,80,90])
    #z = np.searchsorted(quintiles, np.ravel(z)).reshape(z.shape)

    if log_x:
        xx = 10**xx
    if log_y:
        yy = 10**yy
    contour = go.Contour(
        z=z,
        x=xx,
        y=yy,
        line_smoothing=0.85,
        colorscale="Blues",
        colorbar=dict(
            title="Density",
            #thicknessmode="pixels", thickness=50,
            #lenmode="pixels", len=200,
            yanchor="top", y=1,
            tickvals=list(range(1, len(levels))),
            ticktext=["{:.3f}".format(x) for x in levels],
            #ticks="outside", ticksuffix=" bills",
            #dtick=5
        ),
    )
    fig = go.Figure(data=contour)


    #zmax = z.max()
    #breaks = [0.0] + [x / zmax for x in np.quantile(np.ravel(z[z>0]), np.linspace(0.0, 1.0, num=len(colors)-1))]
    #print(breaks)
    #fig.data[0].colorscale = [(a, b[1]) for a, b in zip(breaks, colors)]

    if log_x:
        fig.update_xaxes(type="log")
    if log_y:
        fig.update_yaxes(type="log")
    fig.update_layout(template="simple_white")

    return fig

def _quantile_to_level(data, quantile):
    """Return data levels corresponding to quantile cuts of mass."""
    isoprop = np.asarray(quantile)
    values = np.ravel(data)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")
    return levels

# Adapted from seaborn
class KDE:
    """Univariate and bivariate kernel density estimator."""
    def __init__(
        self, *,
        bw_method=None,
        bw_adjust=1,
        gridsize=100,
        cut=3,
        clip=None,
        cumulative=False,
    ):
        """Initialize the estimator with its parameters.
        Parameters
        ----------
        bw_method : string, scalar, or callable, optional
            Method for determining the smoothing bandwidth to use; passed to
            :class:`scipy.stats.gaussian_kde`.
        bw_adjust : number, optional
            Factor that multiplicatively scales the value chosen using
            ``bw_method``. Increasing will make the curve smoother. See Notes.
        gridsize : int, optional
            Number of points on each dimension of the evaluation grid.
        cut : number, optional
            Factor, multiplied by the smoothing bandwidth, that determines how
            far the evaluation grid extends past the extreme datapoints. When
            set to 0, truncate the curve at the data limits.
        clip : pair of numbers or None, or a pair of such pairs
            Do not evaluate the density outside of these limits.
        cumulative : bool, optional
            If True, estimate a cumulative distribution function. Requires scipy.
        """
        if clip is None:
            clip = None, None

        self.bw_method = bw_method
        self.bw_adjust = bw_adjust
        self.gridsize = gridsize
        self.cut = cut
        self.clip = clip
        self.cumulative = cumulative
        self.support = None

    def _define_support_grid(self, x, bw, cut, clip, gridsize):
        """Create the grid of evaluation points depending for vector x."""
        clip_lo = -np.inf if clip[0] is None else clip[0]
        clip_hi = +np.inf if clip[1] is None else clip[1]
        gridmin = max(x.min() - bw * cut, clip_lo)
        gridmax = min(x.max() + bw * cut, clip_hi)
        return np.linspace(gridmin, gridmax, gridsize)

    def _define_support_univariate(self, x, weights):
        """Create a 1D grid of evaluation points."""
        kde = self._fit(x, weights)
        bw = np.sqrt(kde.covariance.squeeze())
        grid = self._define_support_grid(
            x, bw, self.cut, self.clip, self.gridsize
        )
        return grid

    def _define_support_bivariate(self, x1, x2, weights):
        """Create a 2D grid of evaluation points."""
        clip = self.clip
        if clip[0] is None or np.isscalar(clip[0]):
            clip = (clip, clip)

        kde = self._fit([x1, x2], weights)
        bw = np.sqrt(np.diag(kde.covariance).squeeze())

        grid1 = self._define_support_grid(
            x1, bw[0], self.cut, clip[0], self.gridsize
        )
        grid2 = self._define_support_grid(
            x2, bw[1], self.cut, clip[1], self.gridsize
        )

        return grid1, grid2

    def define_support(self, x1, x2=None, weights=None, cache=True):
        """Create the evaluation grid for a given data set."""
        if x2 is None:
            support = self._define_support_univariate(x1, weights)
        else:
            support = self._define_support_bivariate(x1, x2, weights)

        if cache:
            self.support = support

        return support

    def _fit(self, fit_data, weights=None):
        """Fit the scipy kde while adding bw_adjust logic and version check."""
        fit_kws = {"bw_method": self.bw_method}
        if weights is not None:
            fit_kws["weights"] = weights

        kde = gaussian_kde(fit_data, **fit_kws)
        kde.set_bandwidth(kde.factor * self.bw_adjust)

        return kde

    def _eval_univariate(self, x, weights=None):
        """Fit and evaluate a univariate on univariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x, cache=False)

        kde = self._fit(x, weights)

        if self.cumulative:
            s_0 = support[0]
            density = np.array([
                kde.integrate_box_1d(s_0, s_i) for s_i in support
            ])
        else:
            density = kde(support)

        return density, support

    def _eval_bivariate(self, x1, x2, weights=None):
        """Fit and evaluate a univariate on bivariate data."""
        support = self.support
        if support is None:
            support = self.define_support(x1, x2, cache=False)

        kde = self._fit([x1, x2], weights)

        if self.cumulative:

            grid1, grid2 = support
            density = np.zeros((grid1.size, grid2.size))
            p0 = grid1.min(), grid2.min()
            for i, xi in enumerate(grid1):
                for j, xj in enumerate(grid2):
                    density[i, j] = kde.integrate_box(p0, (xi, xj))

        else:

            xx1, xx2 = np.meshgrid(*support)
            density = kde([xx1.ravel(), xx2.ravel()]).reshape(xx1.shape)

        return density, support

    def __call__(self, x1, x2=None, weights=None):
        """Fit and evaluate on univariate or bivariate data."""
        if x2 is None:
            return self._eval_univariate(x1, weights)
        else:
            return self._eval_bivariate(x1, x2, weights)
