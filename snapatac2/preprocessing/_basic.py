from __future__ import annotations

from typing import Literal
from pathlib import Path
import numpy as np
from anndata import AnnData
import logging

import snapatac2
import snapatac2._snapatac2 as internal
from snapatac2.genome import Genome
from snapatac2.preprocessing._cell_calling import filter_cellular_barcodes_ordmag

__all__ = [ 'add_tile_matrix', 'make_peak_matrix', 'make_gene_matrix',
           'call_cells', 'filter_cells', 'select_features',
]

def add_tile_matrix(
    adata: internal.AnnData | list[internal.AnnData],
    *,
    bin_size: int = 500,
    inplace: bool = True,
    chunk_size: int = 500,
    exclude_chroms: list[str] | str | None = ["chrM", "chrY", "M", "Y"],
    min_frag_size: int | None = None,
    max_frag_size: int | None = None,
    counting_strategy: Literal['fragment', 'insertion', 'paired-insertion'] = 'paired-insertion',
    value_type: Literal['target', 'total', 'fraction'] = 'target',
    summary_type: Literal['sum', 'mean'] = 'sum',
    file: Path | None = None,
    backend: Literal['hdf5'] = 'hdf5',
    n_jobs: int = 8,
) -> internal.AnnData | None:
    """Generate cell by bin count matrix.

    This function is used to generate and add a cell by bin count matrix to the AnnData
    object.

    :func:`~snapatac2.pp.import_fragments` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` could also be a list of AnnData objects when `inplace=True`.
        In this case, the function will be applied to each AnnData object in parallel.
    bin_size
        The size of consecutive genomic regions used to record the counts.
    inplace
        Whether to add the tile matrix to the AnnData object or return a new AnnData object.
    chunk_size
        Increasing the chunk_size speeds up I/O but uses more memory.
    exclude_chroms
        A list of chromosomes to exclude.
    min_frag_size
        Minimum fragment size to include.
    max_frag_size
        Maximum fragment size to include.
    counting_strategy
        The strategy to compute feature counts. It must be one of the following:
        "fragment", "insertion", or "paired-insertion". "fragment" means the
        feature counts are assigned based on the number of fragments that overlap
        with a region of interest. "insertion" means the feature counts are assigned
        based on the number of insertions that overlap with a region of interest.
        "paired-insertion" is similar to "insertion", but it only counts the insertions
        once if the pair of insertions of a fragment are both within the same region
        of interest [Miao24]_.
        Note that this parameter has no effect if input are single-end reads.
    value_type
        The type of value to use from `.obsm['_values']`, only available when 
        data is imported using :func:`~snapatac2.pp.import_values`. It must be one of the following:
        "target", "total", or "fraction". "target" means the value is the number
        of recrods that are with postive measurements, e.g., number of methylated bases.
        "total" means the value is the total number of measurements, e.g., methylated bases plus
        unmethylated bases. "fraction" means the value is the fraction of the
        records that are positive, e.g., the fraction of methylated bases.
    summary_type
        The type of summary to use when multiple values are found in a bin. This parameter
        is only used when `.obsm['_values']` exists, which is created by :func:`~snapatac2.pp.import_values`. 
        It must be one of the following: "sum" or "mean".
    file
        File name of the output file used to store the result. If provided, result will
        be saved to a backed AnnData, otherwise an in-memory AnnData is used.
        This has no effect when `inplace=True`.
    backend
        The backend to use for storing the result. If `None`, the default backend will be used.
    n_jobs
        Number of jobs to run in parallel when `adata` is a list.
        If `n_jobs=-1`, all CPUs will be used.
    
    Returns
    -------
    AnnData | ad.AnnData | None
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to bins. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.

    See Also
    --------
    make_peak_matrix
    make_gene_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> snap.pp.add_tile_matrix(data, bin_size=500)
    >>> print(data)
    AnnData object with n_obs × n_vars = 585 × 6062095
        obs: 'n_fragment', 'frac_dup', 'frac_mito'
        uns: 'reference_sequences'
        obsm: 'fragment_paired'
    """
    def fun(data, out):
        internal.mk_tile_matrix(data, bin_size, chunk_size, counting_strategy, value_type, summary_type, exclude_chroms, min_frag_size, max_frag_size, out)

    if isinstance(exclude_chroms, str):
        exclude_chroms = [exclude_chroms]

    if inplace:
        if isinstance(adata, list):
            snapatac2._utils.anndata_par(
                adata,
                lambda x: fun(x, None),
                n_jobs=n_jobs,
            )
        else:
            fun(adata, None)
    else:
        if file is None:
            if adata.isbacked:
                out = AnnData(obs=adata.obs[:].to_pandas())
            else:
                out = AnnData(obs=adata.obs[:])
        else:
            out = internal.AnnData(filename=file, backend=backend, obs=adata.obs[:])
        fun(adata, out)
        return out

def make_peak_matrix(
    adata: internal.AnnData | internal.AnnDataSet,
    *,
    use_rep: str | list[str] | None = None,
    inplace: bool = False,
    file: Path | None = None,
    backend: Literal['hdf5'] = 'hdf5',
    peak_file: Path | None = None,
    chunk_size: int = 500,
    use_x: bool = False,
    min_frag_size: int | None = None,
    max_frag_size: int | None = None,
    counting_strategy: Literal['fragment', 'insertion', 'paired-insertion'] = 'paired-insertion',
    value_type: Literal['target', 'total', 'fraction'] = 'target',
    summary_type: Literal['sum', 'mean'] = 'sum',
) -> internal.AnnData:
    """Generate cell by peak count matrix.

    This function will generate a cell by peak count matrix and store it in a 
    new .h5ad file.

    :func:`~snapatac2.pp.import_fragments` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    use_rep
        This is used to read peak information from `.uns[use_rep]`.
        The peaks can also be provided by a list of strings:
        ["chr1:1-100", "chr2:2-200"].
    inplace
        Whether to add the tile matrix to the AnnData object or return a new AnnData object.
    file
        File name of the output h5ad file used to store the result. If provided,
        result will be saved to a backed AnnData, otherwise an in-memory AnnData
        is used. This has no effect when `inplace=True`.
    backend
        The backend to use for storing the result. If `None`, the default backend will be used.
    peak_file
        Bed file containing the peaks. If provided, peak information will be read
        from this file.
    chunk_size
        Chunk size
    use_x
        If True, use the matrix stored in `.X` as raw counts.
        Otherwise the `.obsm['insertion']` is used.
    min_frag_size
        Minimum fragment size to include.
    max_frag_size
        Maximum fragment size to include.
    counting_strategy
        The strategy to compute feature counts. It must be one of the following:
        "fragment", "insertion", or "paired-insertion". "fragment" means the
        feature counts are assigned based on the number of fragments that overlap
        with a region of interest. "insertion" means the feature counts are assigned
        based on the number of insertions that overlap with a region of interest.
        "paired-insertion" is similar to "insertion", but it only counts the insertions
        once if the pair of insertions of a fragment are both within the same region
        of interest [Miao24]_.
        Note that this parameter has no effect if input are single-end reads.
    value_type
        The type of value to use from `.obsm['_values']`, only available when 
        data is imported using :func:`~snapatac2.pp.import_values`. It must be one of the following:
        "target", "total", or "fraction". "target" means the value is the number
        of recrods that are with postive measurements, e.g., number of methylated bases.
        "total" means the value is the total number of measurements, e.g., methylated bases plus
        unmethylated bases. "fraction" means the value is the fraction of the
        records that are positive, e.g., the fraction of methylated bases.
    summary_type
        The type of summary to use when multiple values are found in a bin. This parameter
        is only used when `.obsm['_values']` exists, which is created by :func:`~snapatac2.pp.import_values`. 
        It must be one of the following: "sum" or "mean".

    Returns
    -------
    AnnData | ad.AnnData | None
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to peaks. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.

    See Also
    --------
    add_tile_matrix
    make_gene_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> peak_mat = snap.pp.make_peak_matrix(data, peak_file=snap.datasets.cre_HEA())
    >>> print(peak_mat)
    AnnData object with n_obs × n_vars = 585 × 1154611
        obs: 'n_fragment', 'frac_dup', 'frac_mito'
    """
    import gzip

    if peak_file is not None and use_rep is not None:
        raise RuntimeError("'peak_file' and 'use_rep' cannot be both set") 

    if use_rep is None and peak_file is None:
        use_rep = "peaks"

    if isinstance(use_rep, str):
        df = adata.uns[use_rep]
        peaks = df[df.columns[0]]
    else:
        peaks = use_rep

    if peak_file is not None:
        if Path(peak_file).suffix == ".gz":
            with gzip.open(peak_file, 'rt') as f:
                peaks = [line.strip() for line in f]
        else:
            with open(peak_file, 'r') as f:
                peaks = [line.strip() for line in f]

    if inplace:
        out = None
    elif file is None:
        if adata.isbacked:
            out = AnnData(obs=adata.obs[:].to_pandas())
        else:
            out = AnnData(obs=adata.obs[:])
    else:
        out = internal.AnnData(filename=file, backend=backend, obs=adata.obs[:])
    internal.mk_peak_matrix(adata, peaks, chunk_size, use_x, counting_strategy, value_type, summary_type, min_frag_size, max_frag_size, out)
    return out

def make_gene_matrix(
    adata: internal.AnnData | internal.AnnDataSet,
    gene_anno: Genome | Path,
    *,
    inplace: bool = False,
    file: Path | None = None,
    backend: Literal['hdf5'] | None = 'hdf5',
    chunk_size: int = 500,
    use_x: bool = False,
    id_type: Literal['gene', 'transcript'] = "gene",
    upstream: int = 2000,
    downstream: int = 0,
    include_gene_body: bool = True,
    transcript_name_key: str = "transcript_name",
    transcript_id_key: str = "transcript_id",
    gene_name_key: str = "gene_name",
    gene_id_key: str = "gene_id",
    min_frag_size: int | None = None,
    max_frag_size: int | None = None,
    counting_strategy: Literal['fragment', 'insertion', 'paired-insertion'] = 'paired-insertion',
) -> internal.AnnData:
    """Generate cell by gene activity matrix.

    Generate cell by gene activity matrix by counting the TN5 insertions in each gene's
    regulatory domain. The regulatory domain is initially defined as the TSS or the
    whole gene body (if `include_gene_body=True`). We then extends this domain
    by `upstream` and `downstream` base pairs on both sides.
      
    The result will be stored in a new file and a new AnnData object
    will be created.
    :func:`~snapatac2.pp.import_fragments` must be ran first in order to use this function.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
    gene_anno
        Either a Genome object or the path of a gene annotation file in GFF or GTF format.
    inplace
        Whether to add the gene matrix to the AnnData object or return a new AnnData object.
    file
        File name of the h5ad file used to store the result. This has no effect when `inplace=True`.
    backend
        The backend to use for storing the result. If `None`, the default backend will be used.
    chunk_size
        Chunk size
    use_x
        If True, use the matrix stored in `.X` to compute the gene activity.
        Otherwise the `.obsm['insertion']` is used.
    id_type
        "gene" or "transcript".
    upstream
        The number of base pairs upstream of the regulatory domain.
    downstream
        The number of base pairs downstream of the regulatory domain.
    include_gene_body
        Whether to include the gene body in the regulatory domain. If False, the
        TSS is used as the regulatory domain.
    transcript_name_key
        The key of the transcript name in the gene annotation file.
    transcript_id_key
        The key of the transcript id in the gene annotation file.
    gene_name_key
        The key of the gene name in the gene annotation file.
    gene_id_key
        The key of the gene id in the gene annotation file.
    min_frag_size
        Minimum fragment size to include.
    max_frag_size
        Maximum fragment size to include.
    counting_strategy
        The strategy to compute feature counts. It must be one of the following:
        "fragment", "insertion", or "paired-insertion". "fragment" means the
        feature counts are assigned based on the number of fragments that overlap
        with a region of interest. "insertion" means the feature counts are assigned
        based on the number of insertions that overlap with a region of interest.
        "paired-insertion" is similar to "insertion", but it only counts the insertions
        once if the pair of insertions of a fragment are both within the same region
        of interest [Miao24]_.
        Note that this parameter has no effect if input are single-end reads.

    Returns
    -------
    AnnData
        An annotated data matrix of shape `n_obs` x `n_vars`. Rows correspond to
        cells and columns to genes. If `file=None`, an in-memory AnnData will be
        returned, otherwise a backed AnnData is returned.

    See Also
    --------
    add_tile_matrix
    make_peak_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> gene_mat = snap.pp.make_gene_matrix(data, gene_anno=snap.genome.hg38)
    >>> print(gene_mat)
    AnnData object with n_obs × n_vars = 585 × 60606
        obs: 'n_fragment', 'frac_dup', 'frac_mito'
    >>> gene_mat = snap.pp.make_gene_matrix(data, gene_anno=snap.genome.hg38, upstream=1000, downstream=1000, include_gene_body=False)
    """
    if isinstance(gene_anno, Genome):
        gene_anno = gene_anno.annotation

    if inplace:
        out = None
    elif file is None:
        if adata.isbacked:
            out = AnnData(obs=adata.obs[:].to_pandas())
        else:
            out = AnnData(obs=adata.obs[:])
    else:
        out = internal.AnnData(filename=file, backend=backend, obs=adata.obs[:])
    internal.mk_gene_matrix(adata, gene_anno, chunk_size, use_x, id_type,
        upstream, downstream, include_gene_body,
        transcript_name_key, transcript_id_key, gene_name_key, gene_id_key,
        counting_strategy, min_frag_size, max_frag_size, out)
    return out

def call_cells(
    data: internal.AnnData | list[internal.AnnData],
    use_rep: str | np.ndarray[float],
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | None:
    """
    Calling cells based on the number of feature counts.

    This implements Cell Ranger's
    [cell calling algorithm](https://www.10xgenomics.com/support/software/cell-ranger/latest/algorithms-overview/cr-gex-algorithm),
    which is based on two primary algorithms: Order of magnitude (OrdMag) and EmptyDrops.
    
    Currently only OrdMag is implemented.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `data` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    use_rep
        The representation to use for filtering. This can be a string or a numpy array.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `data` is a list.

    Returns
    -------
    np.ndarray | None:
        If `inplace = True`, directly subsets the data matrix. Otherwise return 
        indices of cells that pass the filtering.
    """
    if isinstance(data, list):
        result = snapatac2._utils.anndata_par(
            data,
            lambda x: call_cells(x, inplace=inplace),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    counts = data.obs[use_rep].to_numpy() if isinstance(use_rep, str) else use_rep
    selected_cells = filter_cellular_barcodes_ordmag(counts, None)[0]
    if inplace:
        if data.isbacked:
            data.subset(selected_cells)
        else:
            data._inplace_subset_obs(selected_cells)
    else:
        return selected_cells

def filter_cells(
    data: internal.AnnData | list[internal.AnnData],
    min_counts: int | None = 1000,
    min_tsse: float | None = 5.0,
    max_counts: int | None = None,
    max_tsse: float | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | None:
    """
    Filter cell outliers based on counts and numbers of genes expressed.
    For instance, only keep cells with at least `min_counts` counts or
    `min_tsse` TSS enrichment scores. This is to filter measurement outliers,
    i.e. "unreliable" observations.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `data` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    min_counts
        Minimum number of counts required for a cell to pass filtering.
    min_tsse
        Minimum TSS enrichemnt score required for a cell to pass filtering.
    max_counts
        Maximum number of counts required for a cell to pass filtering.
    max_tsse
        Maximum TSS enrichment score expressed required for a cell to pass filtering.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `data` is a list.

    Returns
    -------
    np.ndarray | None:
        If `inplace = True`, directly subsets the data matrix. Otherwise return 
        indices of cells that pass the filtering.
    """
    if isinstance(data, list):
        result = snapatac2._utils.anndata_par(
            data,
            lambda x: filter_cells(x, min_counts, min_tsse, max_counts, max_tsse, inplace=inplace),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    selected_cells = True
    if min_counts: selected_cells &= data.obs["n_fragment"] >= min_counts
    if max_counts: selected_cells &= data.obs["n_fragment"] <= max_counts
    if min_tsse: selected_cells &= data.obs["tsse"] >= min_tsse
    if max_tsse: selected_cells &= data.obs["tsse"] <= max_tsse

    selected_cells = np.flatnonzero(selected_cells)
    if inplace:
        if data.isbacked:
            data.subset(selected_cells)
        else:
            data._inplace_subset_obs(selected_cells)
    else:
        return selected_cells

def _find_most_accessible_features(
    feature_count,
    filter_lower_quantile,
    filter_upper_quantile,
    total_features,
) -> np.ndarray:
    idx = np.argsort(feature_count)
    for i in range(idx.size):
        if feature_count[idx[i]] > 0:
            break
    idx = idx[i:]
    n = idx.size
    n_lower = int(filter_lower_quantile * n)
    n_upper = int(filter_upper_quantile * n)
    idx = idx[n_lower:n-n_upper]
    return idx[::-1][:total_features]
 
def select_features(
    adata: internal.AnnData | internal.AnnDataSet | list[internal.AnnData],
    n_features: int = 500000,
    filter_lower_quantile: float = 0.005,
    filter_upper_quantile: float = 0.005,
    whitelist: Path | None = None,
    blacklist: Path | None = None,
    max_iter: int = 1,
    inplace: bool = True,
    n_jobs: int = 8,
    verbose: bool = True,
) -> np.ndarray | list[np.ndarray] | None:
    """
    Perform feature selection by selecting the most accessibile features across
    all cells unless `max_iter` > 1.

    Note
    ----
    This function does not perform the actual subsetting. The feature mask is used by
    various functions to generate submatrices on the fly.
    Features that are zero in all cells will be always removed regardless of the
    filtering criteria.
    For more discussion about feature selection, see: https://github.com/scverse/SnapATAC2/discussions/116.

    Parameters
    ----------
    adata
        The (annotated) data matrix of shape `n_obs` x `n_vars`.
        Rows correspond to cells and columns to regions.
        `adata` can also be a list of AnnData objects.
        In this case, the function will be applied to each AnnData object in parallel.
    n_features
        Number of features to keep. Note that the final number of features
        may be smaller than this number if there is not enough features that pass
        the filtering criteria.
    filter_lower_quantile
        Lower quantile of the feature count distribution to filter out.
        For example, 0.005 means the bottom 0.5% features with the lowest counts will be removed.
    filter_upper_quantile
        Upper quantile of the feature count distribution to filter out.
        For example, 0.005 means the top 0.5% features with the highest counts will be removed.
        Be aware that when the number of feature is very large, the default value of 0.005 may
        risk removing too many features.
    whitelist
        A user provided bed file containing genome-wide whitelist regions.
        None-zero features listed here will be kept regardless of the other
        filtering criteria.
        If a feature is present in both whitelist and blacklist, it will be kept.
    blacklist 
        A user provided bed file containing genome-wide blacklist regions.
        Features that are overlapped with these regions will be removed.
    max_iter
        If greater than 1, this function will perform iterative clustering and feature selection
        based on variable features found using previous clustering results.
        This is similar to the procedure implemented in ArchR, but we do not recommend it,
        see https://github.com/scverse/SnapATAC2/issues/111.
        Default value is 1, which means no iterative clustering is performed.
    inplace
        Perform computation inplace or return result.
    n_jobs
        Number of parallel jobs to use when `adata` is a list.
    verbose
        Whether to print progress messages.
    
    Returns
    -------
    np.ndarray | None:
        If `inplace = False`, return a boolean index mask that does filtering,
        where `True` means that the feature is kept, `False` means the feature is removed.
        Otherwise, store this index mask directly to `.var['selected']`.
    """
    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: select_features(x, n_features, filter_lower_quantile,
                                      filter_upper_quantile, whitelist,
                                      blacklist, max_iter, inplace, verbose=False),
            n_jobs=n_jobs,
        )
        if inplace:
            return None
        else:
            return result

    count = np.zeros(adata.shape[1])
    for batch, _, _ in adata.chunked_X(2000):
        count += np.ravel(batch.sum(axis = 0))
    if inplace:
        adata.var['count'] = count

    selected_features = _find_most_accessible_features(
        count, filter_lower_quantile, filter_upper_quantile, n_features)

    if blacklist is not None:
        blacklist = np.array(internal.intersect_bed(adata.var_names, str(blacklist)))
        selected_features = selected_features[np.logical_not(blacklist[selected_features])]

    # Iteratively select features
    iter = 1
    while iter < max_iter:
        embedding = snapatac2.tl.spectral(adata, features=selected_features, inplace=False)[1]
        clusters = snapatac2.tl.leiden(snapatac2.pp.knn(embedding, inplace=False))
        rpm = snapatac2.tl.aggregate_X(adata, groupby=clusters).X
        var = np.var(np.log(rpm + 1), axis=0)
        selected_features = np.argsort(var)[::-1][:n_features]

        # Apply blacklist to the result
        if blacklist is not None:
            selected_features = selected_features[np.logical_not(blacklist[selected_features])]
        iter += 1

    result = np.zeros(adata.shape[1], dtype=bool)
    result[selected_features] = True

    # Finally, apply whitelist to the result
    if whitelist is not None:
        whitelist = np.array(internal.intersect_bed(adata.var_names, str(whitelist)))
        whitelist &= count != 0
        result |= whitelist
    
    if verbose:
        logging.info(f"Selected {result.sum()} features.")

    if inplace:
        adata.var["selected"] = result
    else:
        return result
