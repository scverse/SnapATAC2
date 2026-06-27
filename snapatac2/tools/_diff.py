from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.stats import chi2, norm, zscore
import logging

from snapatac2._snapatac2 import AnnData, AnnDataSet
from snapatac2.tools._misc import aggregate_X

def marker_regions(
    data: AnnData | AnnDataSet,
    groupby: str | list[str],
    pvalue: float = 0.01,
) -> dict[str, list[str]]:
    """
    Select marker regions for each group by z-score enrichment.

    Use this lightweight screen to obtain candidate group-specific regions from
    aggregated accessibility before running more formal differential tests.

    Anti-Patterns
    -------------
    - Do NOT treat these markers as regression-adjusted differential results;
      use :func:`diff_test` for hypothesis testing between two cell groups.
    - Do NOT pass a grouping key that is absent from `data.obs`.

    Parameters
    ----------
    data : AnnData | AnnDataSet
        Annotated data object with regions in `.var_names` and counts in `.X`.
    groupby : str | list[str]
        Grouping key in `data.obs`, or one group label per cell.
    pvalue : float
        One-sided normal survival-function threshold applied to z-scores.

    Returns
    -------
    dict[str, list[str]]
        Mapping from group name to marker region names.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> markers = snap.tl.marker_regions(adata, groupby="cell_type", pvalue=0.01)
    >>> isinstance(markers, dict)
    True
    """
    count = aggregate_X(data, groupby, normalize="RPKM")
    z = zscore(np.log2(1 + count.X), axis = 0)
    peaks = {}
    for i in range(z.shape[0]):
        select = norm.sf(z[i, :]) < pvalue
        if np.where(select)[0].size >= 1:
            peaks[count.obs_names[i]] = count.var_names[select]
    return peaks

def mad(data, axis=None):
    """ Compute Median Absolute Deviation """
    return np.median(np.absolute(data - np.median(data, axis)), axis)

def modified_zscore(matrix, axis=0):
    """ Compute Modified Z-score for a matrix along specified axis """
    median = np.median(matrix, axis=axis)
    median_absolute_deviation = mad(matrix, axis=axis)
    min_non_zero = np.min(median_absolute_deviation[median_absolute_deviation > 0])
    median_absolute_deviation[median_absolute_deviation == 0] = min_non_zero

    if axis == 0:
        modified_z_scores = 0.6745 * (matrix - median) / median_absolute_deviation
    elif axis == 1:
        modified_z_scores = 0.6745 * (matrix.T - median).T / median_absolute_deviation
    else:
        raise ValueError("Invalid axis, it should be 0 or 1")

    return modified_z_scores

def diff_test(
    data: AnnData | AnnDataSet,
    cell_group1: list[int] | list[str],
    cell_group2: list[int] | list[str],
    features : list[str] | list[int] | None = None,
    covariates: list[str] | None = None,
    direction: Literal["positive", "negative", "both"] = "both",
    min_log_fc: float = 0.25,
    min_pct: float = 0.05,
    solver: str = "lbfgs",
) -> 'polars.DataFrame':
    """
    Test regions for differential accessibility between two cell groups.

    Use this function to compare two explicit cell sets with logistic-regression
    likelihood-ratio tests after filtering by detection fraction and fold change.

    Anti-Patterns
    -------------
    - Do NOT pass group names directly; pass cell indices, cell barcodes, or a
      Boolean mask for each group.
    - Do NOT use `covariates`; the parameter is currently not implemented.
    - Do NOT expect features failing `min_pct` or `min_log_fc` to appear in the
      output table.

    Parameters
    ----------
    data : AnnData | AnnDataSet
        Annotated data object with cells in observations and regions in
        variables.
    cell_group1 : list[int] | list[str]
        Cells in group 1 as indices, barcodes, or a Boolean mask.
    cell_group2 : list[int] | list[str]
        Cells in group 2 as indices, barcodes, or a Boolean mask.
    features : list[str] | list[int] | None
        Region names or indices to test. If None, test all regions that pass
        filtering.
    covariates : list[str] | None
        Reserved for covariates; currently raises `NameError` when provided.
    direction : {"positive", "negative", "both"}
        Direction of enrichment to retain. `"positive"` keeps regions enriched
        in group 1, `"negative"` keeps regions enriched in group 2, and
        `"both"` keeps either direction.
    min_log_fc : float
        Minimum absolute or directional log2 fold change required before testing.
    min_pct : float
        Minimum fraction of cells with nonzero accessibility in either group.
    solver : str
        Solver passed to `sklearn.linear_model.LogisticRegression`.

    Returns
    -------
    pl.DataFrame
        Differential accessibility table sorted by adjusted p-value, with
        columns `"feature name"`, `"log2(fold_change)"`, `"p-value"`, and
        `"adjusted p-value"`.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> group1 = list(range(50))
    >>> group2 = list(range(50, 100))
    >>> result = snap.tl.diff_test(adata, group1, group2, features=list(range(20)))
    >>> set(result.columns) <= {"feature name", "log2(fold_change)", "p-value", "adjusted p-value"}
    True
    """
    import polars as pl

    def to_indices(xs, type):
        xs = [_convert_to_bool_if_np_bool(x) for x in xs]
        if all(isinstance(x, bool) for x in xs):
            return [i for i, value in enumerate(xs) if value]
        elif all([isinstance(item, str) for item in xs]):
            if type == "obs":
                if data.isbacked:
                    return data.obs_ix(xs)
                else:
                    return [data.obs_names.get_loc(x) for x in xs]
            else:
                if data.isbacked:
                    return data.var_ix(xs)
                else:
                    return [data.var_names.get_loc(x) for x in xs]
        else:
            return xs

    cell_group1 = to_indices(cell_group1, "obs")
    n_group1 = len(cell_group1)
    cell_group2 = to_indices(cell_group2, "obs")
    n_group2 = len(cell_group2)

    cell_by_peak = data.X[cell_group1 + cell_group2, :].tocsc()
    test_var = np.array([0] * n_group1 + [1] * n_group2)
    if covariates is not None:
        raise NameError("covariates is not implemented")

    features = range(data.n_vars) if features is None else to_indices(features, "var")
    logging.info("Input contains {} features, now perform filtering with 'min_log_fc = {}' and 'min_pct = {}' ...".format(len(features), min_log_fc, min_pct))
    filtered = _filter_features(
        cell_by_peak[:n_group1, :],
        cell_by_peak[n_group1:, :],
        features,
        direction,
        min_pct,
        min_log_fc,
    )

    if len(filtered) == 0:
        logging.warning("Zero feature left after filtering, perhaps 'min_log_fc' or 'min_pct' is too large")
        return pl.DataFrame()
    else:
        features, log_fc = zip(*filtered)
        logging.info("Testing {} features ...".format(len(features)))
        pvals = _diff_test_helper(cell_by_peak, test_var, features, covariates, solver=solver)
        var_names = data.var_names
        return pl.DataFrame({
            "feature name": [var_names[i] for i in features],
            "log2(fold_change)": np.array(log_fc),
            "p-value": np.array(pvals),
            "adjusted p-value": _p_adjust_bh(pvals),
        }).sort("adjusted p-value")

def _p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    p = np.asarray(p, dtype=np.float64)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]

def _filter_features(mat1, mat2, peak_indices, direction,
    min_pct, min_log_fc, pseudo_count = 1,
):
    def rpm(m):
        x = np.ravel(np.sum(m, axis = 0)) + pseudo_count
        s = x.sum()
        return x / (s / 1000000)

    def pass_min_pct(i):
        cond1 = mat1[:, i].count_nonzero() / mat1.shape[0] >= min_pct 
        cond2 = mat2[:, i].count_nonzero() / mat2.shape[0] >= min_pct 
        return cond1 or cond2

    def adjust_sign(fc):
        if direction == "both":
            return abs(fc)
        elif direction == "positive":
            return fc
        elif direction == "negative":
            return -fc
        else:
            raise NameError("direction must be 'positive', 'negative' or 'both'")

    log_fc = np.log2(rpm(mat1) / rpm(mat2))
    peak_indices = [i for i in peak_indices if pass_min_pct(i)]
    return [(i, log_fc[i])  for i in peak_indices if adjust_sign(log_fc[i]) >= min_log_fc]

def _diff_test_helper(mat, z, peaks=None, covariate=None, solver: str = "lbfgs") -> list[float]:
    """
    Parameters
    ----------
    mat
        cell by peak matrix.
    z
        variables to test
    peaks
        peak indices
    covariate 
        additional variables to regress out.
    """

    if len(z.shape) == 1:
        z = z.reshape((-1, 1))
    
    if covariate is None:
        X = np.log1p(np.sum(mat, axis=1))
    else:
        X = covariate

    mat = mat.tocsc()
    if peaks is not None:
        mat = mat[:, peaks]

    return _likelihood_ratio_test_many(np.asarray(X), np.asarray(z), mat, solver=solver)


def _likelihood_ratio_test_many(X, z, Y, solver: str = "lbfgs") -> list[float]:
    """
    Parameters
    ----------
    X
        (n_sample, n_feature).
    z
        (n_sample, 1), the additional variable.
    Y
        (n_sample, k), labels
    
    Returns
    -------
    P-values of whether adding z to the models improves the prediction.
    """
    from tqdm import tqdm
 
    X0 = X
    X1 = np.concatenate((X, z), axis=1)

    _, n = Y.shape
    Y.data = np.ones(Y.data.shape)

    result = []
    for i in tqdm(range(n)):
        result.append(
            _likelihood_ratio_test(X0, X1, np.asarray(np.ravel(Y[:, i].todense())), solver=solver)
        )
    return result

def _likelihood_ratio_test(
    X0: np.ndarray,
    X1: np.ndarray,
    y: np.ndarray,
    solver: str = "lbfgs",
) -> float:
    """
    Comparing null model with alternative model using the likehood ratio test.

    Parameters
    ----------
    X0
        (n_sample, n_feature), variables used in null model.
    X1
        (n_sample, n_feature2), variables used in alternative model.
        Note X1 contains X0.
    Y
        (n_sample, ), labels.

    Returns
    -------
    The P-value.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import log_loss

    model = LogisticRegression(penalty=None, random_state=0, n_jobs=1,
        solver=solver, warm_start=False,
        max_iter = 1000,
        ).fit(X0, y)
    reduced = -log_loss(y, model.predict_proba(X0), normalize=False)

    model = LogisticRegression(penalty=None, random_state=0, n_jobs=1,
        solver=solver, warm_start=False,
        max_iter = 1000,
        ).fit(X1, y)
    full = -log_loss(y, model.predict_proba(X1), normalize=False)
    chi = -2 * (reduced - full)
    return chi2.sf(chi, X1.shape[1] - X0.shape[1])

def _convert_to_bool_if_np_bool(value):
    if isinstance(value, np.bool_):
        return bool(value)
    return value
