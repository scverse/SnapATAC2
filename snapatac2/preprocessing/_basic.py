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
    """Generate a cell-by-genomic-bin count matrix.

    Use this function after :func:`~snapatac2.pp.import_fragments` or
    :func:`~snapatac2.pp.import_values` to summarize per-cell fragments or values
    into fixed-width genomic bins. Execute this step before feature selection,
    dimensionality reduction, clustering, or other workflows that require a count
    matrix in `.X`.

    Anti-Patterns
    -------------
    - Do NOT call this function on an AnnData object that was not created by
      :func:`~snapatac2.pp.import_fragments` or :func:`~snapatac2.pp.import_values`.
      The input must contain the internal fragment or value storage used by
      SnapATAC2.
    - Do NOT use `inplace=False` with a list of AnnData objects. Lists are only
      supported when `inplace=True`, where each object is updated in parallel.
    - Do NOT expect `file` or `backend` to change output storage when
      `inplace=True`; these arguments are only used when `inplace=False`.
    - Do NOT pass `exclude_chroms=None` unless mitochondrial, sex, and other
      special chromosomes should be retained in the tile matrix.

    Parameters
    ----------
    adata
        The imported AnnData object, or a list of imported AnnData objects when
        `inplace=True`. Each object must contain fragment data from
        :func:`~snapatac2.pp.import_fragments` or value data from
        :func:`~snapatac2.pp.import_values`.
    bin_size
        The width, in base pairs, of each consecutive genomic bin.
    inplace
        If `True`, store the tile matrix in `adata.X` and return `None`. If
        `False`, return a new AnnData object containing the tile matrix.
    chunk_size
        Number of bins processed per chunk. Increase this value to improve I/O
        throughput when memory is sufficient; decrease it to reduce peak memory
        use.
    exclude_chroms
        Chromosome names to exclude before binning. By default, mitochondrial
        and Y chromosomes are excluded (`"chrM"`, `"chrY"`, `"M"`, `"Y"`).
    min_frag_size
        Minimum fragment size to include. Fragments shorter than this threshold
        are ignored. Use `None` to disable the lower bound.
    max_frag_size
        Maximum fragment size to include. Fragments longer than this threshold
        are ignored. Use `None` to disable the upper bound.
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
        The value to summarize from `.obsm['_values']` when data was imported
        with :func:`~snapatac2.pp.import_values`. It must be one of "target",
        "total", or "fraction". "target" means the number of records with
        positive measurements, e.g. methylated bases. "total" means the total
        number of measurements, e.g. methylated plus unmethylated bases.
        "fraction" means the fraction of records with positive measurements.
    summary_type
        The aggregation to use when multiple values are found in a bin. This
        parameter is only used when `.obsm['_values']` exists, which is created
        by :func:`~snapatac2.pp.import_values`. It must be "sum" or "mean".
    file
        Output file for the returned AnnData object when `inplace=False`. If
        provided, the result is stored as backed AnnData. If `None`, the result
        is returned in memory. This argument has no effect when `inplace=True`.
    backend
        Backend used for backed output when `file` is provided.
    n_jobs
        Number of parallel jobs to use when `adata` is a list. If `n_jobs=-1`,
        all CPUs are used.
    
    Returns
    -------
    AnnData | ad.AnnData | None
        If `inplace=False`, returns an annotated data matrix whose rows are
        cells and columns are genomic bins. If `file=None`, returns an in-memory
        AnnData object; otherwise returns a backed AnnData object. If
        `inplace=True`, returns `None` and updates `adata.X` in place.

    See Also
    --------
    make_peak_matrix
    make_gene_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> fragments = snap.datasets.pbmc500(downsample=True)
    >>> data = snap.pp.import_fragments(
    ...     fragments,
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> snap.pp.add_tile_matrix(
    ...     data,
    ...     bin_size=500,
    ...     exclude_chroms=["chrM", "chrY"],
    ... )
    >>> print(data.shape)
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
    """Generate a cell-by-peak count matrix.

    Use this function after :func:`~snapatac2.pp.import_fragments`,
    :func:`~snapatac2.pp.import_values`, or peak calling to aggregate fragments
    or values over peak intervals. Provide peak intervals with exactly one of
    `peak_file` or `use_rep`; if both are omitted, the function reads peaks from
    `adata.uns["peaks"]`.

    Anti-Patterns
    -------------
    - Do NOT pass both `peak_file` and `use_rep`; the function raises an error
      because the peak source would be ambiguous.
    - Do NOT call this function before importing fragments or values. The input
      must contain SnapATAC2's internal fragment or value storage.
    - Do NOT expect `file` or `backend` to affect output storage when
      `inplace=True`; these arguments are only used when `inplace=False`.
    - Do NOT set `use_x=True` unless `.X` already contains the feature-by-cell
      counts that should be reused as raw counts.

    Parameters
    ----------
    adata
        The imported AnnData object, or AnnDataSet, containing per-cell fragment
        or value storage.
    use_rep
        Peak source stored in `adata.uns[use_rep]`, or a list of peak strings
        such as `["chr1:1-100", "chr2:2-200"]`. If `None` and `peak_file` is
        also `None`, `"peaks"` is used.
    inplace
        If `True`, store the peak matrix in `adata.X` and return `None`. If
        `False`, return a new AnnData object containing the peak matrix.
    file
        Output file for the returned AnnData object when `inplace=False`. If
        provided, the result is stored as backed AnnData. If `None`, the result
        is returned in memory. This argument has no effect when `inplace=True`.
    backend
        Backend used for backed output when `file` is provided.
    peak_file
        BED file containing peak intervals. Plain text and `.gz` files are
        supported. Do not set this together with `use_rep`.
    chunk_size
        Number of peaks processed per chunk. Increase this value to improve I/O
        throughput when memory is sufficient; decrease it to reduce peak memory
        use.
    use_x
        If `True`, use the matrix stored in `.X` as raw counts. If `False`, use
        the imported fragment or insertion storage.
    min_frag_size
        Minimum fragment size to include. Fragments shorter than this threshold
        are ignored. Use `None` to disable the lower bound.
    max_frag_size
        Maximum fragment size to include. Fragments longer than this threshold
        are ignored. Use `None` to disable the upper bound.
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
        The value to summarize from `.obsm['_values']` when data was imported
        with :func:`~snapatac2.pp.import_values`. It must be one of "target",
        "total", or "fraction". "target" means the number of records with
        positive measurements, e.g. methylated bases. "total" means the total
        number of measurements, e.g. methylated plus unmethylated bases.
        "fraction" means the fraction of records with positive measurements.
    summary_type
        The aggregation to use when multiple values are found in a peak. This
        parameter is only used when `.obsm['_values']` exists, which is created
        by :func:`~snapatac2.pp.import_values`. It must be "sum" or "mean".

    Returns
    -------
    AnnData | ad.AnnData | None
        If `inplace=False`, returns an annotated data matrix whose rows are
        cells and columns are peaks. If `file=None`, returns an in-memory AnnData
        object; otherwise returns a backed AnnData object. If `inplace=True`,
        returns `None` and updates `adata.X` in place.

    See Also
    --------
    add_tile_matrix
    make_gene_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> fragments = snap.datasets.pbmc500(downsample=True)
    >>> data = snap.pp.import_fragments(
    ...     fragments,
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> peak_mat = snap.pp.make_peak_matrix(
    ...     data,
    ...     peak_file=snap.datasets.cre_HEA(),
    ... )
    >>> print(peak_mat.shape)
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
    """Generate a cell-by-gene activity matrix.

    Use this function after :func:`~snapatac2.pp.import_fragments` to summarize
    chromatin accessibility over each gene's regulatory domain. The regulatory
    domain is defined from the TSS, or from the full gene body when
    `include_gene_body=True`, then extended by `upstream` and `downstream` base
    pairs.

    Anti-Patterns
    -------------
    - Do NOT call this function on an AnnData object that was not created by
      :func:`~snapatac2.pp.import_fragments`; the input must contain fragment or
      insertion storage.
    - Do NOT set `use_x=True` unless `.X` already contains the binned or peak
      counts that should be reused for gene aggregation.
    - Do NOT expect `file` or `backend` to affect output storage when
      `inplace=True`; these arguments are only used when `inplace=False`.
    - Do NOT change annotation key names unless the GFF/GTF file uses different
      attribute names than the defaults.

    Parameters
    ----------
    adata
        The imported AnnData object, or AnnDataSet, containing per-cell fragment
        or insertion storage.
    gene_anno
        Genome object or path to a gene annotation file in GFF or GTF format.
    inplace
        If `True`, store the gene activity matrix in `adata.X` and return
        `None`. If `False`, return a new AnnData object containing the gene
        activity matrix.
    file
        Output file for the returned AnnData object when `inplace=False`. If
        provided, the result is stored as backed AnnData. If `None`, the result
        is returned in memory. This argument has no effect when `inplace=True`.
    backend
        Backend used for backed output when `file` is provided.
    chunk_size
        Number of genes processed per chunk. Increase this value to improve I/O
        throughput when memory is sufficient; decrease it to reduce peak memory
        use.
    use_x
        If `True`, use the matrix stored in `.X` to compute gene activity. If
        `False`, use the imported fragment or insertion storage.
    id_type
        Feature identifier to aggregate by. Use "gene" to aggregate transcripts
        into genes, or "transcript" to keep transcript-level entries.
    upstream
        Number of base pairs to extend upstream of the regulatory domain.
    downstream
        Number of base pairs to extend downstream of the regulatory domain.
    include_gene_body
        If `True`, include the full gene body before extension. If `False`, use
        the TSS as the regulatory domain before extension.
    transcript_name_key
        Attribute key for transcript names in the gene annotation file.
    transcript_id_key
        Attribute key for transcript IDs in the gene annotation file.
    gene_name_key
        Attribute key for gene names in the gene annotation file.
    gene_id_key
        Attribute key for gene IDs in the gene annotation file.
    min_frag_size
        Minimum fragment size to include. Fragments shorter than this threshold
        are ignored. Use `None` to disable the lower bound.
    max_frag_size
        Maximum fragment size to include. Fragments longer than this threshold
        are ignored. Use `None` to disable the upper bound.
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
        If `inplace=False`, returns an annotated data matrix whose rows are
        cells and columns are genes or transcripts. If `file=None`, returns an
        in-memory AnnData object; otherwise returns a backed AnnData object. If
        `inplace=True`, returns `None` and updates `adata.X` in place.

    See Also
    --------
    add_tile_matrix
    make_peak_matrix

    Examples
    --------
    >>> import snapatac2 as snap
    >>> fragments = snap.datasets.pbmc500(downsample=True)
    >>> data = snap.pp.import_fragments(
    ...     fragments,
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> gene_mat = snap.pp.make_gene_matrix(data, gene_anno=snap.genome.hg38)
    >>> print(gene_mat.shape)
    >>> promoter_mat = snap.pp.make_gene_matrix(
    ...     data,
    ...     gene_anno=snap.genome.hg38,
    ...     upstream=1000,
    ...     downstream=1000,
    ...     include_gene_body=False,
    ... )
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
    """Call valid cells from feature counts using the OrdMag algorithm.

    Use this function to remove empty or low-signal barcodes after importing
    fragments and computing a per-barcode count metric. The implementation uses
    the order-of-magnitude (OrdMag) strategy from Cell Ranger's cell-calling
    workflow; EmptyDrops is not implemented.

    Anti-Patterns
    -------------
    - Do NOT pass a representation that is missing from `data.obs` when
      `use_rep` is a string.
    - Do NOT use this function as a replacement for QC thresholding by TSS
      enrichment or fragment count; use :func:`filter_cells` when explicit QC
      thresholds are required.
    - Do NOT expect a return value when `inplace=True`; the object is subset in
      place and the function returns `None`.

    Parameters
    ----------
    data
        AnnData object, or list of AnnData objects, to subset to called cells.
    use_rep
        Count representation used for cell calling. If a string, read counts
        from `data.obs[use_rep]`. If an array, use it directly as one count per
        barcode.
    inplace
        If `True`, subset `data` to called cells and return `None`. If `False`,
        return integer indices of called cells without modifying `data`.
    n_jobs
        Number of parallel jobs to use when `data` is a list.

    Returns
    -------
    np.ndarray | list[np.ndarray] | None:
        If `inplace=False`, returns integer indices of barcodes called as cells.
        If `data` is a list, returns one index array per object. If
        `inplace=True`, returns `None` and subsets `data` in place.

    See Also
    --------
    filter_cells : Apply explicit QC thresholds to cells.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> fragments = snap.datasets.pbmc500(downsample=True)
    >>> data = snap.pp.import_fragments(
    ...     fragments,
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> selected = snap.pp.call_cells(data, use_rep="n_fragment", inplace=False)
    >>> data = data[selected, :]
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
    """Filter cells by fragment-count and TSS-enrichment QC thresholds.

    Use this function after computing per-cell QC metrics to remove unreliable
    observations. By default, cells must have at least 1000 fragments and a TSS
    enrichment score of at least 5.0.

    Anti-Patterns
    -------------
    - Do NOT call this function before `data.obs["n_fragment"]` and, when TSS
      filtering is enabled, `data.obs["tsse"]` are available.
    - Do NOT leave `min_tsse` enabled when TSS enrichment was not computed; pass
      `min_tsse=None` to filter only by fragment counts.
    - Do NOT expect a return value when `inplace=True`; the object is subset in
      place and the function returns `None`.

    Parameters
    ----------
    data
        AnnData object, or list of AnnData objects, to filter.
    min_counts
        Minimum `data.obs["n_fragment"]` value required for a cell to pass
        filtering. Use `None` to disable the lower fragment-count bound.
    min_tsse
        Minimum `data.obs["tsse"]` value required for a cell to pass filtering.
        Use `None` to disable the lower TSS-enrichment bound.
    max_counts
        Maximum `data.obs["n_fragment"]` value allowed for a cell to pass
        filtering. Use `None` to disable the upper fragment-count bound.
    max_tsse
        Maximum `data.obs["tsse"]` value allowed for a cell to pass filtering.
        Use `None` to disable the upper TSS-enrichment bound.
    inplace
        If `True`, subset `data` in place and return `None`. If `False`, return
        integer indices of cells passing all enabled thresholds.
    n_jobs
        Number of parallel jobs to use when `data` is a list.

    Returns
    -------
    np.ndarray | list[np.ndarray] | None:
        If `inplace=False`, returns integer indices of cells that pass all
        enabled thresholds. If `data` is a list, returns one index array per
        object. If `inplace=True`, returns `None` and subsets `data` in place.

    See Also
    --------
    call_cells : Call cell-containing barcodes from count distributions.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> fragments = snap.datasets.pbmc500(downsample=True)
    >>> data = snap.pp.import_fragments(
    ...     fragments,
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> snap.metrics.tsse(data, snap.genome.hg38)
    >>> selected = snap.pp.filter_cells(
    ...     data,
    ...     min_counts=1000,
    ...     min_tsse=5.0,
    ...     inplace=False,
    ... )
    >>> data = data[selected, :]
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
    """Select informative genomic features for downstream analysis.

    Use this function after generating a tile, peak, or other count matrix to
    mark features that should be used for dimensionality reduction and graph
    construction. With the default `max_iter=1`, features are selected by total
    accessibility across all cells after lower- and upper-quantile filtering.

    Notes
    -----
    - This function does not subset the matrix. It stores a boolean mask in
      `.var["selected"]` when `inplace=True`, or returns the mask when
      `inplace=False`. Downstream functions use this mask to generate submatrices
      on the fly. Features that are zero in all cells are always removed. For more
      discussion, see https://github.com/scverse/SnapATAC2/discussions/116.
    - How to set n_features: This value depends on the number of features in the input matrix.
      It is generally recommended to set n_features to a large value (10% to 50% of the total features)
      to retain enough features for downstream analysis.

    Anti-Patterns
    -------------
    - Do NOT expect this function to reduce `adata.shape`; it only creates or
      returns a feature mask.
    - Do NOT set `max_iter > 1` unless iterative clustering-based feature
      selection is explicitly required; this mode is slower and is not generally
      recommended.
    - Do NOT use very large `filter_upper_quantile` values on datasets with many
      features unless highly accessible features should be removed aggressively.
    - Do NOT assume blacklist overrides whitelist. If a feature appears in both,
      the whitelist keeps it.
    - Do NOT set n_features too small; the spectral embedding used in this package
      usually benefits from a large number of features.

    Parameters
    ----------
    adata
        AnnData, AnnDataSet, or list of AnnData objects containing a count matrix
        in `.X`. If a list is provided, feature selection is applied to each
        object in parallel.
    n_features
        Maximum number of features to keep. The final number can be smaller if
        too few features pass filtering or have nonzero counts.
    filter_lower_quantile
        Lower quantile of the feature-count distribution to remove. For example,
        `0.005` removes the bottom 0.5% features by total count.
    filter_upper_quantile
        Upper quantile of the feature-count distribution to remove. For example,
        `0.005` removes the top 0.5% features by total count. When the number of
        features is very large, this value can remove many features.
    whitelist
        BED file containing regions to keep. Nonzero features overlapping these
        regions are kept regardless of other filtering criteria. If a feature is
        present in both `whitelist` and `blacklist`, it is kept.
    blacklist
        BED file containing regions to remove. Features overlapping these
        regions are removed unless they are also kept by `whitelist`.
    max_iter
        Number of feature-selection iterations. Use `1` for count-based feature
        selection. Values greater than `1` perform iterative clustering and
        feature selection based on variable features found from previous
        clustering results. This is similar to ArchR but is not generally
        recommended; see https://github.com/scverse/SnapATAC2/issues/111.
    inplace
        If `True`, store the boolean mask in `adata.var["selected"]` and return
        `None`. If `False`, return the mask without modifying `adata`.
    n_jobs
        Number of parallel jobs to use when `adata` is a list.
    verbose
        Whether to print progress messages.
    
    Returns
    -------
    np.ndarray | list[np.ndarray] | None:
        If `inplace=False`, returns a boolean feature mask where `True` means the
        feature is kept and `False` means the feature is removed. If `adata` is
        a list, returns one mask per object. If `inplace=True`, returns `None`
        and stores the mask in `.var["selected"]`.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> fragments = snap.datasets.pbmc500(downsample=True)
    >>> data = snap.pp.import_fragments(
    ...     fragments,
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> snap.pp.add_tile_matrix(data, bin_size=500)
    >>> snap.pp.select_features(data, n_features=250000)
    >>> print(data.var["selected"].sum())
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
