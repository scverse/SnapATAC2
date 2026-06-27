from __future__ import annotations

from typing import Literal
import logging
from pathlib import Path
import numpy as np

import snapatac2._snapatac2 as internal
from snapatac2._utils import is_anndata 
from snapatac2.tools import leiden
from snapatac2.preprocessing import knn

__all__ = ['aggregate_X', 'aggregate_cells']

def aggregate_X(
    adata: internal.AnnData | internal.AnnDataSet,
    groupby: str | list[str] | None = None,
    normalize: Literal["RPM", "RPKM"] | None = None,
    file: Path | None = None,
) -> internal.AnnData:
    """
    Aggregate `.X` values across cells or cell groups.

    Use this function to create pseudobulk count profiles, optionally normalized
    as RPM or RPKM, from all cells or from groups defined by `groupby`.

    Anti-Patterns
    -------------
    - Do NOT use `normalize="RPKM"` unless `adata.var_names` are genomic regions
      in `chrom:start-end` format.
    - Do NOT expect a raw NumPy array return; the function always returns an
      AnnData object containing the aggregated matrix.

    Parameters
    ----------
    adata : AnnData | AnnDataSet
        Annotated data object with cells in observations and features in
        variables.
    groupby : str | list[str] | None
        Grouping key in `adata.obs`, one group label per cell, or None to
        aggregate all cells together.
    normalize : {"RPM", "RPKM"} | None
        Optional normalization applied to each aggregated profile.
    file : pathlib.Path | None
        Output h5ad path for backed results. If None, return an in-memory AnnData.

    Returns
    -------
    AnnData
        AnnData object with aggregated profiles in `.X`, original feature names
        in `.var_names`, and group names in `.obs_names` when `groupby` is set.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> pseudobulk = snap.tl.aggregate_X(adata, groupby="cell_type", normalize="RPM")
    >>> pseudobulk.n_vars == adata.n_vars
    True
    """
    from anndata import AnnData

    def _normalize(X, size_factor = None):
        for i in range(X.shape[0]):
            s = X[i, :].sum()
            if s > 0:
                X[i, :] /= s / 1000000.0
                if size_factor is not None:
                    X[i, :] /= size_factor

    def norm(x):
        if normalize is None:
            return x
        elif normalize == "RPKM":
            size_factor = _get_sizes(adata.var_names) / 1000.0
            return _normalize(x, size_factor)
        elif normalize == "RPM":
            return _normalize(x)
        else:
            raise NameError("Normalization method must be 'RPKM' or 'RPM'")

    if groupby is None:
        groups = None
    else:
        groups = adata.obs[groupby] if isinstance(groupby, str) else groupby
        groups = [x for x in groups]

    names, result = internal.aggregate_x(adata, groups)
    norm(result)

    if file is None:
        out_adata = AnnData(X=result)
    else:
        out_adata = internal.AnnData(filename=file, X=result)

    if groups is not None:
        out_adata.obs_names = names
    out_adata.var_names = adata.var_names
    return out_adata

def aggregate_cells(
    adata: internal.AnnData | internal.AnnDataSet | np.ndarray,
    use_rep: str = 'X_spectral',
    target_num_cells: int | None = None,
    min_cluster_size: int = 50,
    random_state: int = 0,
    key_added: str = 'pseudo_cell',
    inplace: bool = True,
) -> np.ndarray | None:
    """Assign cells to pseudo-cell groups by iterative clustering.

    Use this function to coarsen a cell embedding into pseudo-cell labels while
    preserving local graph structure through repeated Leiden clustering.

    Anti-Patterns
    -------------
    - Do NOT pass `use_rep` as an `.obs` key; it must name an embedding in
      `adata.obsm`.
    - Do NOT expect exactly `target_num_cells` groups; iterative splitting stops
      when clusters cannot be split reliably.

    Parameters
    ----------
    adata : AnnData | AnnDataSet | np.ndarray
        Annotated data object containing `adata.obsm[use_rep]`, or a numeric
        matrix with cells as rows.
    use_rep : str
        Key in `adata.obsm` containing the input embedding.
    target_num_cells : int | None
        Target number of pseudo-cell groups. If None, use
        `adata.n_obs // min_cluster_size`.
    min_cluster_size : int
        Minimum cluster size used during iterative splitting.
    random_state : int
        Seed passed to Leiden clustering.
    key_added : str
        Key in `adata.obs` used to store pseudo-cell labels.
    inplace : bool
        If True, store labels in `adata.obs[key_added]`; if False, return them.

    Returns
    -------
    np.ndarray | None
        If `inplace=True`, stores categorical labels in `adata.obs[key_added]`
        and returns None. If `inplace=False`, returns the labels.

    Examples
    --------
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> X = np.random.default_rng(0).normal(size=(100, 5))
    >>> labels = snap.tl.aggregate_cells(X, min_cluster_size=10, inplace=False)
    >>> labels.shape
    (100,)
    """
    def clustering(data):
        return leiden(knn(data), resolution=1, objective_function='modularity',
            min_cluster_size=min_cluster_size, random_state=random_state)

    if is_anndata(adata):
        X = adata.obsm[use_rep]
    else:
        inplace = False
        X = adata

    if target_num_cells is None:
        target_num_cells = X.shape[0] // min_cluster_size

    logging.info("Perform initial clustering ...")
    membership = clustering(X).astype('object')
    cluster_ids = [x for x in np.unique(membership) if x != "-1"]
    ids_next = cluster_ids
    n_clusters = len(cluster_ids)
    depth = 0
    while n_clusters < target_num_cells and len(ids_next) > 0:
        depth += 1
        logging.info("Iterative clustering: {}, number of clusters: {}".format(depth, n_clusters))
        ids = set()
        for cid in ids_next:
            mask = membership == cid
            sub_clusters = clustering(X[mask, :])
            n_sub_clusters = np.count_nonzero(np.unique(sub_clusters) != "-1")
            if n_sub_clusters > 1 and np.count_nonzero(sub_clusters != "-1") / sub_clusters.shape[0] > 0.9:
                n_clusters += n_sub_clusters - 1
                for i, i_ in enumerate(np.where(mask)[0]):
                    lab = sub_clusters[i]
                    if lab == "-1":
                        membership[i_] = lab
                    else:
                        new_lab = membership[i_] + "." + lab
                        membership[i_] = new_lab
                        ids.add(new_lab)
            if n_clusters >= target_num_cells:
                break
        ids_next = ids
    logging.info("Asked for {} pseudo-cells; Got: {}.".format(target_num_cells, n_clusters))

    if inplace:
        import polars
        adata.obs[key_added] = polars.Series(
            [str(x) for x in membership],
            dtype=polars.datatypes.Categorical,
        )
    else:
        return membership
 
def marker_enrichment(
    gene_matrix: internal.AnnData,
    groupby: str | list[str],
    markers: dict[str, list[str]],
    min_num_markers: int = 1,
    hierarchical: bool = True,
):
    """
    Parameters
    ----------
    gene_matrix
        The cell by gene activity matrix.
    groupby
        Group the cells into different groups. If a `str`, groups are obtained from
        `.obs[groupby]`.
    """
    from scipy.stats import zscore
    import polars as pl

    gene_names = dict((x.upper(), i) for i, x in enumerate(gene_matrix.var_names))
    retained = []
    removed = []
    for key in markers.keys():
        genes = []
        for name in markers[key]:
            name = name.upper()
            if name in gene_names:
                genes.append(gene_names[name])
        if len(genes) >= min_num_markers:
            retained.append((key, genes))
        else:
            removed.append(key)
    if len(removed) > 0:
        logging.warn("The following cell types are not annotated because they have less than {} marker genes: {}", min_num_markers, removed)

    aggr_counts = aggregate_X(gene_matrix, groupby=groupby, normalize="RPM")
    zscores = zscore(
        np.log2(np.vstack(list(aggr_counts.values())) + 1),
        axis = 0,
    )

    if hierarchical:
        return _hierarchical_enrichment(dict(retained), zscores)
    else:
        df = pl.DataFrame(
            np.vstack([zscores[:, genes].mean(axis = 1) for _, genes in retained]),
            columns = list(aggr_counts.keys()),
        )
        df.insert_at_idx(0, pl.Series("Cell type", [cell_type for cell_type, _ in retained]))
        return df

def _hierarchical_enrichment(
    marker_genes,
    zscores,
):
    from scipy.cluster.hierarchy import linkage, to_tree
    from collections import Counter
    
    def jaccard_distances(x):
        def jaccard(a, b):
            a = set(a)
            b = set(b)
            return 1 - len(a.intersection(b)) / len(a.union(b))

        result = []
        n = len(x)
        for i in range(n):
            for j in range(i+1, n):
                result.append(jaccard(x[i], x[j]))
        return result

    def make_tree(Z, genes, labels):
        def get_genes_weighted(node, node2 = None):
            leaves = node.pre_order(lambda x: x.id)
            if node2 is not None:
                leaves = leaves + node2.pre_order(lambda x: x.id)
            n = len(leaves)
            count = Counter(g for i in leaves for g in genes[i])
            for key in count.keys():
                count[key] /= n
            return count
        
        def normalize_weights(a, b):
            a_ = []
            for k, v in a.items():
                if k in b:
                    v = v - b[k]
                if v > 0:
                    a_.append((k, v))
            return a_
        
        def process(pid, x, score):
            scores.append(score)
            parents.append(pid)
            ids.append(x.id)
            if x.id < len(labels):
                labels_.append(labels[x.id])
            else:
                labels_.append("")
            go(x)     

        def norm(b, x):
            return np.sqrt(np.exp(b) * np.exp(x))

        def go(tr):
            def sc_fn(gene_w):
                if len(gene_w) > 0:
                    idx, ws = zip(*gene_w)
                    return np.average(zscores[:, list(idx)], axis = 1, weights=list(ws))
                else:
                    return np.zeros(zscores.shape[0])

            left = tr.left
            right = tr.right
            if left is not None and right is not None:
                genes_left = get_genes_weighted(left)
                genes_right = get_genes_weighted(right)
                base = sc_fn(list(get_genes_weighted(left, right).items()))
                sc_left = sc_fn(normalize_weights(genes_left, genes_right))
                sc_right = sc_fn(normalize_weights(genes_right, genes_left))
                process(tr.id, left, norm(base, sc_left))
                process(tr.id, right, norm(base, sc_right))
                
        root = to_tree(Z)
        ids = [root.id]
        parents = [""]
        labels_ = [""]
        scores = [np.zeros(zscores.shape[0])]
        go(root)
        return (ids, parents, labels_, np.vstack(scores).T)

    jm = jaccard_distances([v for v in marker_genes.values()])
    Z = linkage(jm, method='average')
    return make_tree(
        Z, list(marker_genes.values()), list(marker_genes.keys()),
    )

def _get_sizes(regions):
    def size(x):
        x = x.split(':')[1].split("-")
        return int(x[1]) - int(x[0])
    return np.array(list(size(x) for x in regions), dtype=np.float64)
