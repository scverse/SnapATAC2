from __future__ import annotations

from typing import Literal
import scipy as sp
import numpy as np
import gc
import logging
import math

from snapatac2._utils import is_anndata 
import snapatac2._snapatac2 as internal

__all__ = ['umap', 'spectral', 'multi_spectral']

def umap(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    n_comps: int = 2,
    use_dims: int | list[int] | None = None,
    use_rep: str = "X_spectral",
    key_added: str = 'umap',
    random_state: int | None = 0,
    inplace: bool = True,
    **kwargs
) -> np.ndarray | None:
    """
    Compute a UMAP embedding from an existing representation.

    Use this function after computing a low-dimensional representation such as
    `X_spectral`. Pass a NumPy array directly when you want the embedding returned
    instead of written to an AnnData object.

    Anti-Patterns
    -------------
    - Do NOT set `inplace=True` when passing a raw NumPy array; arrays cannot store
      `.obsm` results and the embedding is returned instead.
    - Do NOT pass cluster labels through `key_added`; this key names
      `adata.obsm["X_" + key_added]`, not `adata.obs`.

    Parameters
    ----------
    adata : AnnData | AnnDataSet | np.ndarray
        Annotated data object containing `adata.obsm[use_rep]`, or a numeric
        matrix with shape `(n_cells, n_features)`.
    n_comps : int
        Number of UMAP dimensions to compute.
    use_dims : int | list[int] | None
        Dimensions from `use_rep` to use. If an integer, use the first
        `use_dims` columns; if a list, use those column indices; if None, use
        all columns.
    use_rep : str
        Key in `adata.obsm` containing the input representation.
    key_added : str
        Suffix for the output key `adata.obsm["X_" + key_added]`.
    random_state : int | None
        Random seed passed to `umap.UMAP`.
    inplace : bool
        If True, store the embedding in `adata.obsm`; if False, return it.
    **kwargs
        Additional keyword arguments passed to `umap.UMAP`.

    Returns
    -------
    np.ndarray | None
        If `inplace=True` and `adata` is an AnnData object, stores the embedding
        in `adata.obsm["X_" + key_added]` and returns None. Otherwise, returns
        the embedding with shape `(n_cells, n_comps)`.

    Examples
    --------
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> X = np.random.default_rng(0).normal(size=(20, 5))
    >>> embedding = snap.tl.umap(X, n_comps=2, inplace=False, n_neighbors=5)
    >>> embedding.shape
    (20, 2)
    """
    from umap import UMAP

    if is_anndata(adata):
        data = adata.obsm[use_rep]
    else:
        data = adata
        inplace = False

    if use_dims is not None:
        data = data[:, :use_dims] if isinstance(use_dims, int) else data[:, use_dims]

    umap = UMAP(random_state=random_state,
                n_components=n_comps,
                **kwargs).fit_transform(data)
    if inplace:
        adata.obsm["X_" + key_added] = umap
    else:
        return umap

def idf(data, features=None):
    n, m = data.shape
    count = np.zeros(m)
    for batch, _, _ in data.chunked_X(2000):
        batch.data = np.ones(batch.indices.shape, dtype=np.float64)
        count += np.ravel(batch.sum(axis = 0))
    if features is not None:
        count = count[features]
    return np.log(n / (1 + count))

def spectral(
    adata: internal.AnnData | internal.AnnDataSet,
    n_comps: int = 30,
    features: str | np.ndarray | None = "selected",
    random_state: int = 0,
    sample_size: int | float | None = None,
    sample_method: Literal["random", "degree"] = "random",
    chunk_size: int = 5000,
    distance_metric: Literal["jaccard", "cosine"] = "cosine",
    weighted_by_sd: bool = True,
    feature_weights: list[float] | None = None,
    inplace: bool = True,
    num_threads: int = 32,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Compute a spectral embedding with Laplacian Eigenmaps.

    Use this function to convert a cell-by-feature matrix into a lower-dimensional
    representation before neighbor graph construction, clustering, and visualization.
    With `distance_metric="cosine"`, the matrix-free implementation scales linearly
    with the number of cells. Other distance metrics materialize pairwise similarity
    matrices and scale quadratically with cell count.

    Anti-Patterns
    -------------
    - Do NOT use `distance_metric="jaccard"` on very large datasets without
      `sample_size`; the full pairwise computation can require quadratic memory.
    - Do NOT assume exactly `n_comps` components are returned when
      `weighted_by_sd=True`; small negative eigenvalues may be removed.
    - Do NOT leave `features="selected"` unless `adata.var["selected"]` exists.
    
    Note
    ----
    - Determining the appropriate number of components is crucial when performing
      downstream analyses to ensure optimal clustering outcomes. Utilizing components
      that are either uninformative or irrelevant can compromise the quality of the results.
      By default, this function adopts a strategy where all eigenvectors are weighted
      according to the square root of their corresponding eigenvalues, rather than
      implementing a strict cutoff threshold. This method generally provides satisfactory
      results, circumventing the necessity for manual specification of component numbers.
      However, it's important to note that there might be exceptional cases with
      certain datasets where deviating from this default setting could yield better
      outcomes. In such scenarios, you can disable the automatic weighting by
      setting `weighted_by_sd=False`. Subsequently, you will need to manually determine
      and select the number of components to use for your specific analysis.
    - This funciton may not always return the exact number of eigenvectors requested.
      This function computes lower-dimensional embeddings by performing the
      eigen-decomposition of the normalized graph Laplacian matrix, where all
      eigenvalues should be non-negative. However, the method used to calculate
      eigenvectors, specifically `scipy.sparse.linalg.eigsh`, may not perform
      optimally for small eigenvalues. This occasionally leads to the function
      outputting negative eigenvalues at the lower spectrum. To address this issue,
      a post-processing step is introduced to eliminate these erroneous eigenvalues
      when `weighted_by_sd=True` (which is the default setting). This step
      typically has minimal impact, as the affected eigenvalues are generally very small.

    Parameters
    ----------
    adata : AnnData | AnnDataSet
        Annotated data object with a cell-by-feature count matrix in `.X`.
    n_comps : int
        Maximum number of spectral dimensions to compute.
    features : str | np.ndarray | None
        Feature selector. If a string, use `adata.var[features]` as a Boolean
        mask. If an array, use it directly. If None, use all features.
    random_state : int
        Seed for random sampling and eigensolver initialization.
    sample_size : int | float | None
        Number or fraction of cells to sample for the Nystrom approximation. If
        None, use all cells.
    sample_method : {"random", "degree"}
        Sampling method for the matrix-free Nystrom approximation.
    chunk_size : int
        Number of cells per chunk in Nystrom extension. The effective work batch
        is approximately `chunk_size * num_threads`.
    distance_metric : {"jaccard", "cosine"}
        Similarity metric used to construct the cell graph.
    weighted_by_sd : bool
        If True, multiply eigenvectors by the square root of their eigenvalues.
    feature_weights : list[float] | None
        Per-feature weights for similarity computation. If None, use inverse
        document frequency weights where required.
    inplace : bool
        If True, write results to `adata`; if False, return them.
    num_threads : int
        Number of threads used by the Nystrom implementation.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        If `inplace=True`, stores eigenvectors in `adata.obsm["X_spectral"]`
        and eigenvalues in `adata.uns["spectral_eigenvalue"]`, then returns
        None. If `inplace=False`, returns `(eigenvalues, eigenvectors)`.

    See Also
    --------
    multi_spectral

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> snap.tl.spectral(adata, features=None, n_comps=30)
    >>> adata.obsm["X_spectral"].shape[0] == adata.n_obs
    True
    """
    np.random.seed(random_state)

    if isinstance(features, str):
        if features in adata.var:
            features = adata.var[features].to_numpy()
        else:
            raise NameError("Please call `select_features` first or explicitly set `features = None`")

    n_comps = min(adata.n_vars - 1, adata.n_obs - 1, n_comps)

    n_sample, _ = adata.shape
    if sample_size is None:
        sample_size = n_sample
    elif isinstance(sample_size, int):
        if sample_size <= 1:
            raise ValueError("when sample_size is an integer, it should be > 1")
        if sample_size > n_sample:
            sample_size = n_sample
    else:
        if sample_size <= 0.0 or sample_size > 1.0:
            raise ValueError("when sample_size is a float, it should be > 0 and <= 1")
        else:
            sample_size = int(sample_size * n_sample)

    if sample_size >= n_sample:
        if distance_metric == "cosine":
            evals, evecs = internal.spectral_embedding(adata, features, n_comps, random_state, feature_weights)
        else:
            if feature_weights is None:
                feature_weights = idf(adata, features)
            model = Spectral(n_comps, distance_metric, feature_weights)
            X = adata.X[...] if features is None else adata.X[:, features]
            model.fit(X)
            evals, evecs = model.transform()
    else:
        logging.info("Perform spectral embedding using the Nystrom algorithm...")
        if distance_metric == "cosine":
            if sample_method == "random":
                weighted_by_degree = False
            else:
                weighted_by_degree = True
            v, u = internal.spectral_embedding_nystrom(adata, features, n_comps, sample_size, weighted_by_degree, chunk_size, None, num_threads)
            evals, evecs = orthogonalize(v, u)
        else:
            if feature_weights is None:
                feature_weights = idf(adata, features)
            model = Spectral(n_comps, distance_metric, feature_weights)
            if adata.isbacked:
                S = adata.X.chunk(sample_size, replace=False)
            else:
                S = sp.sparse.csr_matrix(adata.chunk_X(sample_size, replace=False))
            if features is not None: S = S[:, features]

            model.fit(S)

            from tqdm import tqdm
            for batch, _, _ in tqdm(adata.chunked_X(chunk_size), total=math.ceil(adata.n_obs/chunk_size)):
                if distance_metric == "jaccard":
                    batch.data = np.ones(batch.indices.shape, dtype=np.float64)
                if features is not None: batch = batch[:, features]
                model.extend(batch)
            evals, evecs = model.transform()

    if weighted_by_sd:
        idx = [i for i in range(evals.shape[0]) if evals[i] > 0]
        evals = evals[idx]
        evecs = evecs[:, idx] * np.sqrt(evals)

    if inplace:
        adata.uns['spectral_eigenvalue'] = evals
        adata.obsm['X_spectral'] = evecs
    else:
        return (evals, evecs)


class Spectral:
    def __init__(
        self,
        out_dim: int = 30,
        distance: Literal["jaccard", "cosine"] = "jaccard",
        feature_weights = None,
    ):
        self.out_dim = out_dim
        self.distance = distance
        if (self.distance == "jaccard"):
            self.compute_similarity = lambda x, y=None: internal.jaccard_similarity(x, y, feature_weights)
        elif (self.distance == "cosine"):
            self.compute_similarity = lambda x, y=None: internal.cosine_similarity(x, y, feature_weights)
        elif (self.distance == "rbf"):
            from sklearn.metrics.pairwise import rbf_kernel
            self.compute_similarity = lambda x, y=None: rbf_kernel(x, y)
        else:
            raise ValueError("Invalid distance")

    def fit(self, mat, verbose: int = 1):
        """
        mat
            Sparse matrix, note that if `distance == jaccard`, the matrix will be
            interpreted as a binary matrix.
        """
        self.sample = mat
        self.in_dim = mat.shape[1]
        if verbose > 0:
            logging.info("Compute similarity matrix")
        A = self.compute_similarity(mat)

        if (self.distance == "jaccard"):
            if verbose > 0:
                logging.info("Normalization")
            self.coverage = mat.sum(axis=1) / self.in_dim
            self.normalizer = JaccardNormalizer(A, self.coverage)
            self.normalizer.normalize(A, self.coverage, self.coverage)
            np.fill_diagonal(A, 0)
            # Remove outlier
            self.normalizer.outlier = np.quantile(A, 0.999)
            np.clip(A, a_min=0, a_max=self.normalizer.outlier, out=A)
        else:
            np.fill_diagonal(A, 0)

        # M <- D^-1/2 * A * D^-1/2
        D = np.sqrt(A.sum(axis=1)).reshape((-1, 1))
        np.divide(A, D, out=A)
        np.divide(A, D.T, out=A)

        if verbose > 0:
            logging.info("Perform decomposition")
        evals, evecs = sp.sparse.linalg.eigsh(A, self.out_dim, which='LM')
        ix = evals.argsort()[::-1]
        self.evals = np.real(evals[ix])
        self.evecs = np.real(evecs[:, ix])

        B = np.divide(self.evecs, D)
        np.divide(B, self.evals.reshape((1, -1)), out=B)

        self.B = B
        self.Q = []

        return self

    def extend(self, data):
        A = self.compute_similarity(self.sample, data)
        if (self.distance == "jaccard"):
            self.normalizer.normalize(
                A, self.coverage, data.sum(axis=1) / self.in_dim,
                clip_min=0, clip_max=self.normalizer.outlier
            )
        self.Q.append(A.T @ self.B)

    def transform(self, orthogonalize = True):
        if len(self.Q) > 0:
            Q = np.concatenate(self.Q, axis=0)
            D_ = np.sqrt(np.multiply(Q, self.evals.reshape(1, -1)) @ Q.sum(axis=0).T)
            np.divide(Q, D_.reshape((-1, 1)), out=Q)

            if orthogonalize:
                # orthogonalization
                sigma, V = np.linalg.eig(Q.T @ Q)
                sigma = np.sqrt(sigma)
                B = np.multiply(V.T, self.evals.reshape((1,-1))) @ V
                np.multiply(B, sigma.reshape((-1, 1)), out=B)
                np.multiply(B, sigma.reshape((1, -1)), out=B)
                evals_new, evecs_new = np.linalg.eig(B)

                # reorder
                ix = evals_new.argsort()[::-1]
                self.evals = evals_new[ix]
                evecs_new = evecs_new[:, ix]

                np.divide(evecs_new, sigma.reshape((-1, 1)), out=evecs_new)
                self.evecs = Q @ V @ evecs_new
            else:
                self.evecs = Q
        return (self.evals, self.evecs)

def orthogonalize(evals, evecs):
    _, sigma, Vt = np.linalg.svd(evecs, full_matrices=False)
    V = Vt.T

    B = np.multiply(V.T, evals.reshape((1,-1))) @ V
    np.multiply(B, sigma.reshape((-1, 1)), out=B)
    np.multiply(B, sigma.reshape((1, -1)), out=B)
    evals_new, evecs_new = np.linalg.eig(B)

    # reorder
    ix = evals_new.argsort()[::-1]
    evals_new = evals_new[ix]
    evecs_new = evecs_new[:, ix]

    np.divide(evecs_new, sigma.reshape((-1, 1)), out=evecs_new)
    evecs_new = evecs @ V @ evecs_new
    return (evals_new, evecs_new)

class JaccardNormalizer:
    def __init__(self, jm, c):
        (slope, intersect) = internal.jm_regress(jm, c)
        self.slope = slope
        self.intersect = intersect
        self.outlier = None

    def normalize(self, jm, c1, c2, clip_min=None, clip_max=None):
        # jm / (self.slope / (1 / c1 + 1 / c2.T - 1) + self.intersect)
        temp = 1 / c1 + 1 / c2.T
        temp -= 1
        np.reciprocal(temp, out=temp)
        np.multiply(temp, self.slope, out=temp)
        temp += self.intersect
        jm /= temp
        if clip_min is not None or clip_max is not None:
            np.clip(jm, a_min=clip_min, a_max=clip_max, out=jm)
        gc.collect()

def multi_spectral(
    adatas: list[internal.AnnData] | list[internal.AnnDataSet], 
    n_comps: int = 30,
    features: str | list[str] | list[np.ndarray] | None = "selected",
    weights: list[float] | None = None,
    random_state: int = 0,
    weighted_by_sd: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one spectral embedding from multiple modalities.

    Use this function to integrate multiple AnnData or AnnDataSet objects that
    describe the same cells with different feature spaces. The returned embedding
    combines modality-specific graph structure before downstream clustering.

    Anti-Patterns
    -------------
    - Do NOT pass modalities with different cell order; rows are assumed to refer
      to the same cells in the same order.
    - Do NOT pass `features="selected"` unless each modality has
      `adata.var["selected"]`.

    Parameters
    ----------
    adatas : list[AnnData] | list[AnnDataSet]
        Modalities to embed. Each object must have the same observations in the
        same order.
    n_comps : int
        Maximum number of spectral dimensions to compute.
    features : str | list[str] | list[np.ndarray] | None
        Feature selectors for each modality. A single string or None is reused
        for every modality.
    weights : list[float] | None
        Modality weights. If None, weight all modalities equally.
    random_state : int
        Seed for random initialization.
    weighted_by_sd : bool
        If True, multiply eigenvectors by the square root of their eigenvalues.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Eigenvalues and integrated eigenvectors.

    See Also
    --------
    spectral

    Examples
    --------
    >>> import snapatac2 as snap
    >>> atac = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> gene = snap.pp.make_gene_matrix(atac, snap.genome.hg38)
    >>> evals, evecs = snap.tl.multi_spectral([atac, gene], features=None)
    >>> evecs.shape[0] == atac.n_obs
    True
    """
    np.random.seed(random_state)

    if features is None or isinstance(features, str):
        features = [features] * len(adatas)
    if all(isinstance(f, str) for f in features):
        features = [adata.var[feature].to_numpy() for adata, feature in zip(adatas, features)]

    if weights is None:
        weights = [1.0 for _ in adatas]

    evals, evecs = internal.multi_spectral_embedding(adatas, features, weights, n_comps, random_state)

    if weighted_by_sd:
        idx = [i for i in range(evals.shape[0]) if evals[i] > 0]
        evals = evals[idx]
        evecs = evecs[:, idx] * np.sqrt(evals)

    return (evals, evecs)
