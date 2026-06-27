from __future__ import annotations

import numpy as np
import itertools
from scipy.special import logsumexp

import snapatac2._snapatac2 as internal
from snapatac2._utils import is_anndata 

def mnc_correct(
    adata: internal.AnnData | internal.AnnDataSet | np.adarray,
    *,
    batch: str | list[str],
    n_neighbors: int = 5,
    n_clusters: int = 40,
    n_iter: int = 1,
    use_rep: str = "X_spectral",
    use_dims: int | list[int] | None = None,
    groupby: str | list[str] | None = None,
    key_added: str | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | None:
    """
    Correct batch effects with centroid-based mutual nearest neighbors.

    Use this function after dimensionality reduction and before neighbor-graph
    construction to align cells across batches. The method clusters each batch,
    identifies mutual nearest cluster centroids, and projects cells along the
    resulting correction vectors.

    Anti-Patterns
    -------------
    - Do NOT run this function on raw count matrices unless distances between raw
      counts are the intended analysis; use a reduced representation such as
      `X_spectral`.
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
        Number of nearest centroids to inspect when finding mutual nearest
        neighbors.
    n_clusters
        Maximum number of clusters to form in each batch.
    n_iter
        Number of correction iterations.
    use_rep
        Key in `.obsm` containing the input embedding.
    use_dims
        Dimensions of `use_rep` or the input array to use. If an integer, use the
        first `use_dims` columns. If a list, use those column indices.
    groupby
        Column name or labels used to split cells and run correction
        independently within each group.
    key_added
        Key used to store the corrected embedding. If `None`, store it in
        `.obsm[use_rep + "_mnn"]`.
    inplace
        If `True` and `adata` is AnnData-like, store the corrected embedding in
        `.obsm`. Ignored for NumPy input.
    n_jobs
        Number of worker processes used when `groupby` is specified.

    Returns
    -------
    np.ndarray | None
        Corrected embedding of shape `n_obs` x `n_selected_components` when
        `inplace=False` or when `adata` is a NumPy array. Returns `None` when
        `inplace=True` and stores the result in `.obsm`.

    Examples
    --------
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> X = np.array([[0.0, 0.1], [0.2, 0.0], [3.0, 3.1], [3.2, 3.0]])
    >>> batch = ["a", "a", "b", "b"]
    >>> corrected = snap.pp.mnc_correct(X, batch=batch, n_clusters=2, inplace=False)
    >>> corrected.shape
    (4, 2)
    """
    if is_anndata(adata):
        mat = adata.obsm[use_rep]
    else:
        mat = adata
        inplace = False

    if isinstance(use_dims, int): use_dims = range(use_dims) 
    mat = mat if use_dims is None else mat[:, use_dims]
    mat = np.asarray(mat)

    if isinstance(batch, str):
        batch = adata.obs[batch]
    elif isinstance(batch, list):
        assert len(batch) == mat.shape[0], "When `batch` is a list of strings,  \
            it is interpreted as the batch labels of cells and it must \
            have the same length as the number of cells."

    if groupby is None:
        mat = _mnc_correct_main(mat, batch, n_iter, n_neighbors, n_clusters)
    else:
        from multiprocess import Pool

        if isinstance(groupby, str): groupby = adata.obs[groupby]

        group_indices = {}
        for i, group in enumerate(groupby):
            if group in group_indices:
                group_indices[group].append(i)
            else:
                group_indices[group] = [i]
        group_indices = [x for x in group_indices.values()]

        inputs = [(mat[group_idx, :], batch[group_idx]) for group_idx in group_indices]
        with Pool(n_jobs) as p:
            results = p.map(lambda x: _mnc_correct_main(x[0], x[1], n_iter, n_neighbors, n_clusters), inputs)
        for idx, result in zip(group_indices, results):
            mat[idx, :] = result

    if inplace:
        if key_added is None:
            adata.obsm[use_rep + "_mnn"] = mat
        else:
            adata.obsm[key_added] = mat
    else:
        return mat

def _mnc_correct_main(
    data_matrix,
    batch_labels,
    n_iter,
    n_neighbors,
    n_clusters,
    random_state=0
):
    label_uniq = list(set(batch_labels))

    if len(label_uniq) > 1:
        for _ in range(n_iter):
            batch_idx = []
            data_by_batch = []
            for label in label_uniq:
                idx = [i for i, x in enumerate(batch_labels) if x == label]
                batch_idx.append(idx)
                data_by_batch.append(data_matrix[idx,:])
            new_matrix = _mnc_correct_multi(data_by_batch, n_neighbors, n_clusters, random_state)

            idx = list(itertools.chain.from_iterable(batch_idx))
            idx = np.argsort(idx)
            data_matrix = new_matrix[idx, :]
    return data_matrix

def _mnc_correct_multi(datas, n_neighbors, n_clusters, random_state):
    data0 = datas[0]
    for i in range(1, len(datas)):
        data1 = datas[i]
        pdata0, pdata1 = _mnc_correct_pair(data0, data1, n_clusters, n_neighbors, random_state)
        ratio = data0.shape[0] / (data0.shape[0] + data1.shape[0])
        data0_ = pdata0.project(data0, weight = 1 - ratio)
        data1_ = pdata1.project(data1, weight = ratio)
        data0 = np.concatenate((data0_, data1_), axis=0)
    return data0

def _mnc_correct_pair(X, Y, n_clusters, n_neighbors, random_state):
    from sklearn.neighbors import KDTree
    from sklearn.cluster import KMeans

    n_X = X.shape[0]
    n_Y = Y.shape[0]
    c_X = KMeans(
        n_clusters=min(n_clusters, n_X), n_init=10, random_state=random_state
    ).fit(X).cluster_centers_
    c_Y = KMeans(
        n_clusters=min(n_clusters, n_Y), n_init=10, random_state=random_state
    ).fit(Y).cluster_centers_

    tree_X = KDTree(c_X)
    tree_Y = KDTree(c_Y)

    # X by Y matrix
    m_X = tree_Y.query(c_X, k=min(n_neighbors, n_Y), return_distance=False)

    # Y by X matrix
    m_Y_ = tree_X.query(c_Y, k=min(n_neighbors, n_X), return_distance=False)
    m_Y = []
    for i in range(m_Y_.shape[0]):
        m_Y.append(set(m_Y_[i,:]))

    i_X = []
    i_Y = []
    for i in range(m_X.shape[0]):
        for j in m_X[i]:
            if i in m_Y[j]:
                i_X.append(i)
                i_Y.append(j)
    a = c_X[i_X,:]
    b = c_Y[i_Y,:]
    return (Projector(a, b), Projector(b, a))

class Projector(object):
    def __init__(self, X, Y):
        self.reference = X
        self.vector = Y - X

    def project(self, X, weight=0.5):
        def project(x):
            P = self.reference
            U = self.vector
            d = np.sqrt(np.sum((P - x)**2, axis=1))
            w = _normalize(-(d/0.005))
            #w = 1/d
            return (x + weight * np.average(U, axis=0, weights=w))
        return np.apply_along_axis(project, 1, X)

# exp transform the weights and then normalize them to sum to 1.
def _normalize(ws):
    s = logsumexp(ws)
    return np.exp(ws - s)
