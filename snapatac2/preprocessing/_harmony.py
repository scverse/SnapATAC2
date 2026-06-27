"""
Use harmony to integrate cells from different experiments.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial

import snapatac2._snapatac2 as internal
from snapatac2._utils import is_anndata


def harmony(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    *,
    batch: str | list[str],
    use_rep: str = "X_spectral",
    use_dims: int | list[int] | None = None,
    groupby: str | list[str] | None = None,
    key_added: str | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
    **kwargs,
) -> np.ndarray | None:
    """
    Correct batch effects in an embedding with Harmony.

    Use this function after dimensionality reduction and before neighbor-graph
    construction to align cells across experiments, samples, or other batch
    covariates. The function reads the input embedding from `adata.obsm[use_rep]`
    for AnnData-like input, or directly uses a NumPy array when `adata` is an
    array. Additional keyword arguments are passed to `harmonypy.run_harmony`.

    Anti-Patterns
    -------------
    - Do NOT run Harmony on raw count matrices; provide a reduced embedding such
      as `X_spectral`.
    - Do NOT pass `batch` as a column name when `adata` is a NumPy array; provide
      a pandas DataFrame or Series of batch labels instead.

    Parameters
    ----------
    adata
        AnnData-like object with `use_rep` in `.obsm`, AnnDataSet-like object,
        or a NumPy array of shape `n_obs` x `n_components`.
    batch
        Column name or list of column names in `.obs` that identify batches, or
        a tabular batch-label object accepted by Harmony when using array input.
    use_rep
        Key in `.obsm` containing the input embedding.
    use_dims
        Dimensions of `use_rep` or the input array to use. If an integer, use the
        first `use_dims` columns. If a list, use those column indices.
    groupby
        Column name or labels used to split cells and run Harmony independently
        within each group.
    key_added
        Key used to store the corrected embedding. If `None`, store it in
        `.obsm[use_rep + "_harmony"]`.
    inplace
        If `True` and `adata` is AnnData-like, store the corrected embedding in
        `.obsm`. Ignored for NumPy input.
    n_jobs
        Number of worker processes used when `groupby` is specified.
    kwargs
        Additional arguments passed to `harmonypy.run_harmony()`.

    Returns
    -------
    np.ndarray | None
        Corrected embedding of shape `n_obs` x `n_selected_components` when
        `inplace=False` or when `adata` is a NumPy array. Returns `None` when
        `inplace=True` and stores the result in `.obsm`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import snapatac2 as snap
    >>> X = np.array([[0.0, 0.1], [0.2, 0.0], [3.0, 3.1], [3.2, 3.0]])
    >>> batch = pd.DataFrame({"sample": ["a", "a", "b", "b"]})
    >>> corrected = snap.pp.harmony(X, batch=batch, inplace=False, max_iter_harmony=1)
    >>> corrected.shape
    (4, 2)
    """
    # Check if the data is in an AnnData object
    if is_anndata(adata):
        mat = adata.obsm[use_rep]
    else:
        mat = adata
        inplace = False

    # Use only the specified dimensions
    if isinstance(use_dims, int):
        use_dims = range(use_dims)
    mat = mat if use_dims is None else mat[:, use_dims]

    # Create a pandas dataframe with the batch information
    if isinstance(batch, str) or isinstance(batch, list):
        batch = pd.DataFrame(adata.obs[batch])

    if groupby is None:
        mat = _harmony(mat, batch, **kwargs)
    else:
        if isinstance(groupby, str):
            groupby = adata.obs[groupby]
        groups = list(set(groupby))
        group_idxs = [
            [i for i, x in enumerate(groupby) if x == group] for group in groups
        ]
        margs = [
            (mat[group_idx, :], batch.iloc[group_idx, :].copy())
            for group_idx in group_idxs
        ]

        with mp.Pool(processes=min(n_jobs, len(groups))) as pool:
            mats = pool.starmap(partial(_harmony, **kwargs), margs)
        for i, group_idx in enumerate(group_idxs):
            mat[group_idx, :] = mats[i]

    if inplace:
        if key_added is None:
            adata.obsm[use_rep + "_harmony"] = mat
        else:
            adata.obsm[key_added] = mat
    else:
        return mat


def _harmony(data_matrix, batch_labels, **kwargs):
    try:
        import harmonypy
    except ImportError:
        raise ImportError("\nplease install harmonypy:\n\n\tpip install harmonypy")
    if data_matrix.shape[0] == 1:
        return data_matrix
    if len(batch_labels.shape) == 1:
        batch_labels = pd.DataFrame(batch_labels)
    # check if batch has >1 unique values
    for b in batch_labels.columns:
        if len(batch_labels[b].unique()) == 1:
            batch_labels = batch_labels.drop(b, axis=1)
    if batch_labels.shape[1] == 0:
        return data_matrix
    harmony_out = harmonypy.run_harmony(
        data_matrix,
        pd.DataFrame(batch_labels.values, columns=batch_labels.columns),
        batch_labels.columns.values,
        **kwargs,
    )
    return harmony_out.Z_corr.T
