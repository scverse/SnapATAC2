from __future__ import annotations

import numpy as np
import logging

from snapatac2._snapatac2 import AnnData, AnnDataSet

def transfer_labels(
    adata: AnnData | AnnDataSet,
    use_rep: str | np.ndarray,
    labels: str | list[str],
    n_neighbors: int = 15,
    metric: str = "cosine",
    inplace: bool = True,
):
    """
    Predict missing cell labels from labeled neighbors.

    Use this function when `labels` contains known labels for some cells and
    `None` for cells that should be annotated from nearest neighbors in an
    embedding.

    Anti-Patterns
    -------------
    - Do NOT call this when all cells already have labels; the function returns
      None and logs a warning.
    - Do NOT use a `labels` string with `inplace=True` unless you intend to
      overwrite that `adata.obs` column.

    Parameters
    ----------
    adata : AnnData | AnnDataSet
        Annotated data object containing the embedding and label metadata.
    use_rep : str | np.ndarray
        Key in `adata.obsm` containing the embedding, or an embedding matrix with
        cells as rows.
    labels : str | list[str]
        Label key in `adata.obs`, or one label per cell. Entries equal to None
        are predicted.
    n_neighbors : int
        Number of neighbors used by `sklearn.neighbors.KNeighborsClassifier`.
    metric : str
        Distance metric passed to the classifier.
    inplace : bool
        If True and `labels` is a string, overwrite `adata.obs[labels]`; otherwise
        return the completed labels.

    Returns
    -------
    np.ndarray | None
        If labels are written in place or no prediction is needed, returns None.
        Otherwise, returns an array with missing labels filled.

    Examples
    --------
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> labels = np.array(["known"] * 10 + [None] * (adata.n_obs - 10), dtype=object)
    >>> predicted = snap.tl.transfer_labels(adata, "X_spectral", labels, inplace=False)
    >>> (predicted != None).all()
    True
    """
    from sklearn.neighbors import KNeighborsClassifier
    
    labs = adata.obs[labels].to_numpy(copy=True) if isinstance(labels, str) else np.array(labels, copy=True)
    if (labs != None).all():
        logging.warning("Nothing to do, because every cell has a label.")
        return None
    
    embedding = adata.obsm[use_rep] if isinstance(use_rep, str) else use_rep

    model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
    X = embedding[labs != None, :]
    y = labs[labs != None]
    model.fit(X, y)
    
    labs[np.where(labs == None)] = model.predict(embedding[labs == None, :])
    
    if inplace and isinstance(labels, str):
        adata.obs[labels] = labs
    else:
        return labs
