from __future__ import annotations

from typing import Literal
import numpy as np
from scipy.sparse import csr_matrix

from snapatac2._utils import is_anndata
import snapatac2._snapatac2 as internal

def knn(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    n_neighbors: int = 50,
    use_dims: int | list[int] | None = None,
    use_rep: str = 'X_spectral',
    method: Literal['kdtree', 'hora', 'pynndescent'] = "kdtree",
    inplace: bool = True,
    random_state: int = 0,
) -> csr_matrix | None:
    """
    Build a Euclidean k-nearest-neighbor graph for observations.

    Use this function after dimensionality reduction to construct the graph used
    by downstream clustering, embedding, or graph-based analysis. When `adata` is
    an AnnData-like object, the input matrix is read from `adata.obsm[use_rep]`.
    When `adata` is a NumPy array, the array itself is used and the result is
    always returned.

    Anti-Patterns
    -------------
    - Do NOT pass a raw count matrix unless Euclidean distances on counts are the
      intended analysis; use a reduced representation such as `X_spectral`.
    - Do NOT expect `random_state` to make `method="hora"` deterministic; the
      HNSW backend currently ignores this value.

    Parameters
    ----------
    adata
        AnnData-like object with `use_rep` in `.obsm`, AnnDataSet-like object,
        or a NumPy array of shape `n_obs` x `n_features`.
    n_neighbors
        Number of nearest neighbors to store for each observation.
    use_dims
        Dimensions of `use_rep` or the input array to use. If an integer, use the
        first `use_dims` columns. If a list, use those column indices.
    use_rep
        Key in `.obsm` containing the representation to search.
    method
        Neighbor-search backend. Use `"kdtree"` for exact search, `"hora"` for
        approximate HNSW search, or `"pynndescent"` for approximate NNDescent.
    inplace
        If `True` and `adata` is AnnData-like, store the graph in
        `.obsp["distances"]`. Ignored for NumPy input.
    random_state
        Random seed used only by `method="pynndescent"`.

    Returns
    -------
    csr_matrix | None
        Sparse distance matrix of shape `n_obs` x `n_obs` when `inplace=False`
        or when `adata` is a NumPy array. Returns `None` when `inplace=True` and
        stores the matrix in `.obsp["distances"]`.

    Examples
    --------
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> X = np.array([[0.0, 0.0], [0.1, 0.0], [2.0, 2.0], [2.1, 2.0]])
    >>> graph = snap.pp.knn(X, n_neighbors=2, method="kdtree")
    >>> graph.shape
    (4, 4)
    """
    if is_anndata(adata):
        data = adata.obsm[use_rep]
    else:
        inplace = False
        data = adata
    if data.size == 0:
        raise ValueError("matrix is empty")

    if use_dims is not None:
        if isinstance(use_dims, int):
            data = data[:, :use_dims]
        else:
            data = data[:, use_dims]

    n = data.shape[0]
    if method == 'hora':
        adj = internal.approximate_nearest_neighbour_graph(
            data.astype(np.float32), n_neighbors)
    elif method == 'pynndescent':
        import pynndescent
        index = pynndescent.NNDescent(data, n_neighbors=max(50, n_neighbors), random_state=random_state)
        adj, distances = index.neighbor_graph
        indices = np.ravel(adj[:, :n_neighbors])
        distances = np.ravel(distances[:, :n_neighbors]) 
        indptr = np.arange(0, distances.size + 1, n_neighbors)
        adj = csr_matrix((distances, indices, indptr), shape=(n, n))
        adj.sort_indices()
    elif method == 'kdtree':
        adj = internal.nearest_neighbour_graph(data, n_neighbors)
    else:
        raise ValueError("method must be one of 'hora', 'pynndescent', 'kdtree'")
    
    if inplace:
        adata.obsp['distances'] = adj
    else:
        return adj
