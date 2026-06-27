from __future__ import annotations

from scipy.sparse import vstack
import numpy as np
import scipy.sparse as ss

import snapatac2._snapatac2 as internal

def smooth(
    adata: internal.AnnData | internal.AnnDataSet | ss.spmatrix,
    distances: str | ss.spmatrix | None = None,
    inplace: bool = True,
) -> np.ndarray | None:
    """
    Smooth cell-by-feature values over a nearest-neighbor graph.

    Use this function to diffuse `.X` values through the cell graph stored in
    `adata.obsp`, or through a sparse distance matrix passed directly.

    Anti-Patterns
    -------------
    - Do NOT run this before constructing a neighbor graph when `distances` is a
      string key.
    - Do NOT expect embeddings to be written; smoothing updates `.X` when
      `inplace=True`.

    Parameters
    ----------
    adata : AnnData | AnnDataSet | scipy.sparse.spmatrix
        Annotated data object whose `.X` matrix is smoothed, or a sparse matrix
        to smooth when `inplace=False`.
    distances : str | scipy.sparse.spmatrix | None
        Neighbor distance graph. If None, use `adata.obsp["distances"]`. If a
        string, use `adata.obsp[distances]`.
    inplace : bool
        If True, store the smoothed matrix in `adata.X`; if False, return it.

    Returns
    -------
    np.ndarray | scipy.sparse.spmatrix | None
        If `inplace=True`, updates `adata.X` and returns None. If
        `inplace=False`, returns the smoothed matrix.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> snap.pp.knn(adata, use_rep="X_spectral")
    >>> snap.tl.smooth(adata)
    >>> adata.X.shape == (adata.n_obs, adata.n_vars)
    True
    """
 
    if distances is None: distances = "distances"
    if isinstance(str, distances): distances = adata.obsp[distances] 
    data = adata.X[:] if isinstance(internal.AnnData, adata) or isinstance(internal.AnnDataSet, adata) else adata
    data = data * make_diffuse_operator(distances)
    if inplace:
        adata.X = data
    else:
        return data

def make_diffuse_operator(knn_d, t = 3):
    """Create a diffusion operator from a nearest-neighbor distance matrix.

    Use this helper to convert a KNN distance graph into a Markov matrix and
    raise it to `t` diffusion steps before smoothing feature values.
    """
    return make_markov_matrix(knn_d)**t

def make_markov_matrix(knn_d):
    """
    Turn a (knn) distance matrix into a markov matrix.
    """
    rows = []
    for i in range(knn_d.shape[0]):
        row = knn_d.getrow(i)
        # local nearest neighbor for estimating local density
        ka = int(row.nnz / 3)
        # set sigma for each cell i to the distance to its kath nearest neighbor
        # FIXME: corner case where sigma == 0
        sigma = np.sort(row.data)[ka - 1]
        # apply guassian kernel to get the affinity matrix
        row.data = np.exp(-np.square(row.data / sigma))
        rows.append(row)
    affinity = vstack(rows, format="csr")

    # symmetrize the affinity matrix
    affinity += affinity.T
    # make stochastic matrix
    s = np.ravel(affinity.sum(axis=1))
    for i in range(affinity.shape[0]):
        affinity.data[affinity.indptr[i] : affinity.indptr[i+1]] /= s[i]
    return affinity
