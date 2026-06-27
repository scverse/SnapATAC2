from __future__ import annotations

import numpy as np
import itertools

import snapatac2._snapatac2 as internal
from snapatac2._utils import is_anndata 

def scanorama_integrate(
    adata: internal.AnnData | internal.AnnDataSet | np.adarray,
    *,
    batch: str | list[str],
    n_neighbors: int = 20,
    use_rep: str = 'X_spectral',
    use_dims: int | list[int] | None = None,
    groupby: str | list[str] | None = None,
    key_added: str | None = None,
    sigma: float = 15,
    approx: bool = True,
    alpha: float = 0.10,
    batch_size: int = 5000,
    inplace: bool = True,
    **kwargs,
):
    """
    Integrate batch-specific embeddings with Scanorama.

    Use this function after `snap.tl.spectral` and before `snap.pp.knn` to align
    cells from multiple batches. The function reads the input embedding from
    `adata.obsm[use_rep]` for AnnData-like input, or directly uses a NumPy array
    when `adata` is an array. It uses the Scanorama implementation from
    https://github.com/brianhie/scanorama.

    Anti-Patterns
    -------------
    - Do NOT run Scanorama on raw count matrices; provide a reduced embedding
      such as `X_spectral`.
    - Do NOT pass `batch` as a column name when `adata` is a NumPy array; provide
      one label per observation instead.

    Parameters
    ----------
    adata
        AnnData-like object with `use_rep` in `.obsm`, AnnDataSet-like object,
        or a NumPy array of shape `n_obs` x `n_components`.
    batch
        Column name in `.obs` that identifies batches, or a list of labels with
        one entry per observation.
    n_neighbors
        Number of mutual nearest neighbors used by Scanorama.
    use_rep
        Key in `.obsm` containing the input embedding.
    use_dims
        Dimensions of `use_rep` or the input array to use. If an integer, use the
        first `use_dims` columns. If a list, use those column indices.
    groupby
        Column name or labels used to split cells and run Scanorama
        independently within each group.
    key_added
        Key used to store the corrected embedding. If `None`, store it in
        `.obsm[use_rep + "_scanorama"]`.
    sigma
        Gaussian kernel width passed to Scanorama.
    approx
        Whether Scanorama uses approximate nearest-neighbor search.
    alpha
        Alignment score cutoff passed to Scanorama.
    batch_size
        Batch size passed to Scanorama for nearest-neighbor search.
    inplace
        If `True` and `adata` is AnnData-like, store the corrected embedding in
        `.obsm`. Ignored for NumPy input.
    kwargs
        Additional arguments passed to `scanorama.assemble()`.

    Returns
    -------
    np.ndarray | None
        Corrected embedding of shape `n_obs` x `n_selected_components` when
        `inplace=False` or when `adata` is a NumPy array. Returns `None` when
        `inplace=True` and stores the result in `.obsm`.
    
    See Also
    --------
    :func:`~snapatac2.tl.spectral`: compute spectral embedding of the data matrix.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.read(snap.datasets.pbmc5k(type='h5ad'), backed=None)
    >>> snap.pp.select_features(adata)
    >>> snap.tl.spectral(adata)
    >>> midpoint = adata.n_obs // 2
    >>> adata.obs['batch'] = ['a'] * midpoint + ['b'] * (adata.n_obs - midpoint)
    >>> snap.pp.scanorama_integrate(adata, batch='batch')
    >>> 'X_spectral_scanorama' in adata.obsm
    True
    """
    if is_anndata(adata):
        mat = adata.obsm[use_rep]
    else:
        mat = adata
        inplace = False

    # Use only the specified dimensions
    if isinstance(use_dims, int): use_dims = range(use_dims) 
    mat = mat if use_dims is None else mat[:, use_dims]

    if isinstance(batch, str):
        batch = adata.obs[batch]

    if groupby is None:
        mat = _scanorama(mat, batch, n_neighbors, sigma, approx, alpha, batch_size, **kwargs)
    else:
        if isinstance(groupby, str): groupby = adata.obs[groupby]
        groups = list(set(groupby))
        for group in groups:
            group_idx = [i for i, x in enumerate(groupby) if x == group]
            mat[group_idx, :] = _scanorama(
                mat[group_idx, :], batch[group_idx], n_neighbors, sigma, approx, alpha, batch_size, **kwargs)

    if inplace:
        if key_added is None:
            adata.obsm[use_rep + "_scanorama"] = mat
        else:
            adata.obsm[key_added] = mat
    else:
        return mat

def _scanorama(data_matrix, batch_labels, knn, sigma, approx, alpha, batch_size, **kwargs):
    try:
        import scanorama
    except ImportError:
        raise ImportError("\nplease install Scanorama:\n\n\tpip install scanorama")
    import pandas as pd

    label_uniq = list(set(batch_labels))

    if len(label_uniq) > 1:
        batch_idx = []
        data_by_batch = []
        for label in label_uniq:
            idx = [i for i, x in enumerate(batch_labels) if x == label]
            batch_idx.append(idx)
            data_by_batch.append(data_matrix[idx,:])
        new_matrix = np.concatenate(scanorama.assemble(
            data_by_batch,
            knn=knn,
            sigma=sigma,
            approx=approx,
            alpha=alpha,
            ds_names=label_uniq,
            batch_size=batch_size,
            verbose=0,
            **kwargs,
        ))
        idx = list(itertools.chain.from_iterable(batch_idx))
        idx = np.argsort(idx)
        data_matrix = new_matrix[idx, :]
    return data_matrix
