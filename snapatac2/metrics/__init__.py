from __future__ import annotations

from pathlib import Path
from typing import Literal
import numpy as np

import snapatac2
import snapatac2._snapatac2 as internal
from snapatac2.genome import Genome

def tsse(
    adata: internal.AnnData | list[internal.AnnData],
    gene_anno: Genome | Path,
    *,
    exclude_chroms: list[str] | str | None = ["chrM", "M"],
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """Compute transcription start site enrichment for each cell.

    Run this metric after :func:`~snapatac2.pp.import_fragments` has attached
    fragment metadata to the AnnData object. With `inplace=True`, the function
    writes cell-level scores to `adata.obs["tsse"]` and library-level summaries
    to `adata.uns`.

    Anti-Patterns
    -------------
    - Do NOT call this function on an AnnData object that lacks imported
      fragments.
    - Do NOT pass a genome object without an annotation file; `gene_anno` must
      resolve to a GTF/GFF annotation.

    Parameters
    ----------
    adata : snapatac2._snapatac2.AnnData or list[snapatac2._snapatac2.AnnData]
        AnnData object, or a list of AnnData objects, with imported fragments.
        When a list is provided, compute TSSe for each object in parallel.
    gene_anno : snapatac2.genome.Genome or pathlib.Path
        Genome object with an `annotation` path, or a GTF/GFF annotation file
        path used to define transcription start sites.
    exclude_chroms : list[str], str, or None, default: ["chrM", "M"]
        Chromosome names to exclude when computing the TSS profile. Use None to
        include all chromosomes.
    inplace : bool, default: True
        If True, store results in `adata.obs` and `adata.uns`. If False, return
        the result dictionary instead.
    n_jobs : int, default: 8
        Number of jobs to run when `adata` is a list. If `n_jobs=-1`, use all
        available CPUs.

    Returns
    -------
    dict[str, object] or list[dict[str, object]] or None
        If `inplace=True`, returns None after storing `tsse` in `adata.obs` and
        `library_tsse`, `frac_overlap_TSS`, and `TSS_profile` in `adata.uns`. If
        `inplace=False`, returns the same values in a dictionary, or a list of
        dictionaries when `adata` is a list.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> snap.metrics.tsse(data, snap.genome.hg38)
    >>> print(data.obs['tsse'].head())
    AAACTGCAGACTCGGA-1    32.129514
    AAAGATGCACCTATTT-1    22.052786
    AAAGATGCAGATACAA-1    27.109808
    AAAGGGCTCGCTCTAC-1    24.990329
    AAATGAGAGTCCCGCA-1    33.264463
    Name: tsse, dtype: float64
    """
    gene_anno = gene_anno.annotation if isinstance(gene_anno, Genome) else gene_anno
 
    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: tsse(x, gene_anno, exclude_chroms=exclude_chroms, inplace=inplace),
            n_jobs=n_jobs,
        )
    else:
        result = internal.tss_enrichment(adata, gene_anno, exclude_chroms)
        result['tsse'] = np.array(result['tsse'])
        result['TSS_profile'] = np.array(result['TSS_profile'])
        if inplace:
            adata.obs["tsse"] = result['tsse']
            adata.uns['library_tsse'] = result['library_tsse']
            adata.uns['frac_overlap_TSS'] = result['frac_overlap_TSS']
            adata.uns['TSS_profile'] = result['TSS_profile']
    if inplace:
        return None
    else:
        return result

def frip(
    adata: internal.AnnData | list[internal.AnnData],
    regions: dict[str, Path | list[str]],
    *,
    normalized: bool = True,
    count_as_insertion: bool = False,
    inplace: bool = True,
    n_jobs: int = 8,
) -> dict[str, list[float]] | list[dict[str, list[float]]] | None:
    """Compute fraction of reads or insertions in selected regions.

    Run this metric after :func:`~snapatac2.pp.import_fragments` has attached
    fragment metadata to the AnnData object. Use the keys of `regions` as output
    column names; with `inplace=True`, each metric is written to `adata.obs`.

    Anti-Patterns
    -------------
    - Do NOT call this function on an AnnData object that lacks imported
      fragments.
    - Do NOT reuse the same `regions` dictionary across calls if you need to
      preserve original path values; this function converts path values to
      region lists in place.

    Parameters
    ----------
    adata : snapatac2._snapatac2.AnnData or list[snapatac2._snapatac2.AnnData]
        AnnData object, or a list of AnnData objects, with imported fragments.
        When a list is provided, compute FRiP for each object in parallel.
    regions : dict[str, pathlib.Path or list[str]]
        Mapping from output metric name to a BED file path or a list of genomic
        intervals such as `"chr1:100-200"`.
    normalized : bool, default: True
        If True, return fractions normalized by the total number of fragments or
        insertions. If False, return raw counts overlapping each region set.
    count_as_insertion : bool, default: False
        If True, count transposition insertions at fragment ends instead of whole
        fragments.
    inplace : bool, default: True
        If True, store each result vector in `adata.obs` using the corresponding
        `regions` key. If False, return the result dictionary.
    n_jobs : int, default: 8
        Number of jobs to run when `adata` is a list. If `n_jobs=-1`, use all
        available CPUs.

    Returns
    -------
    dict[str, list[float]] | list[dict[str, list[float]]] | None
        If `inplace = True`, directly adds the results to `adata.obs`.
        Otherwise return a dictionary containing the results.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(snap.datasets.pbmc500(downsample=True), chrom_sizes=snap.genome.hg38, sorted_by_barcode=False)
    >>> snap.metrics.frip(data, {"peaks_frac": snap.datasets.cre_HEA()})
    >>> print(data.obs['peaks_frac'].head())
    AAACTGCAGACTCGGA-1    0.715930
    AAAGATGCACCTATTT-1    0.697364
    AAAGATGCAGATACAA-1    0.713615
    AAAGGGCTCGCTCTAC-1    0.678428
    AAATGAGAGTCCCGCA-1    0.724910
    Name: peaks_frac, dtype: float64
    """

    for k in regions.keys():
        if isinstance(regions[k], str) or isinstance(regions[k], Path):
            regions[k] = internal.read_regions(Path(regions[k]))
        elif not isinstance(regions[k], list):
            regions[k] = list(iter(regions[k]))

    if isinstance(adata, list):
        result = snapatac2._utils.anndata_par(
            adata,
            lambda x: frip(x, regions, inplace=inplace),
            n_jobs=n_jobs,
        )
    else:
        result = internal.add_frip(adata, regions, normalized, count_as_insertion)
        if inplace:
            for k, v in result.items():
                adata.obs[k] = v
    if inplace:
        return None
    else:
        return result

def frag_size_distr(
    adata: internal.AnnData | list[internal.AnnData],
    *,
    max_recorded_size: int = 1000,
    add_key: str = "frag_size_distr",
    inplace: bool = True,
    n_jobs: int = 8,
) -> np.ndarray | list[np.ndarray] | None:
    """Compute the dataset-level fragment size distribution.

    Run this metric after :func:`~snapatac2.pp.import_fragments` has attached
    fragment metadata to the AnnData object. The result is a vector where index
    `i` counts fragments of length `i`, except index 0 counts fragments longer
    than `max_recorded_size`. This metric summarizes the whole dataset rather
    than individual cells.

    Anti-Patterns
    -------------
    - Do NOT interpret the returned vector as cell-level values; it is one
      distribution per AnnData object.
    - Do NOT call this function on an AnnData object that lacks imported
      fragments.

    Parameters
    ----------
    adata : snapatac2._snapatac2.AnnData or list[snapatac2._snapatac2.AnnData]
        AnnData object, or a list of AnnData objects, with imported fragments.
        When a list is provided, compute one distribution for each object in
        parallel.
    max_recorded_size : int, default: 1000
        Largest fragment length with its own output bin. Fragments longer than
        this value are counted at index 0.
    add_key : str, default: "frag_size_distr"
        Key used to store the distribution in `adata.uns` when `inplace=True`.
    inplace : bool, default: True
        If True, store the distribution in `adata.uns[add_key]`. If False,
        return the distribution.
    n_jobs : int, default: 8
        Number of jobs to run when `adata` is a list. If `n_jobs=-1`, use all
        available CPUs.

    Returns
    -------
    np.ndarray | list[np.ndarray] | None
        If `inplace=True`, returns None after storing the distribution in
        `adata.uns[add_key]`. If `inplace=False`, returns the distribution, or a
        list of distributions when `adata` is a list.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(
    ...     snap.datasets.pbmc500(downsample=True),
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> snap.metrics.frag_size_distr(data)
    >>> data.uns["frag_size_distr"].shape[0]
    1001
    """
    if isinstance(adata, list):
        return snapatac2._utils.anndata_par(
            adata,
            lambda x: frag_size_distr(x, add_key=add_key, max_recorded_size=max_recorded_size, inplace=inplace),
            n_jobs=n_jobs,
        )
    else:
        result = np.array(internal.fragment_size_distribution(adata, max_recorded_size))
        if inplace:
            adata.uns[add_key] = result
        else:
            return result

def summary_by_chrom(
    adata: internal.AnnData | list[internal.AnnData],
    *,
    mode: Literal['sum', 'mean', 'count'] = 'count',
    n_jobs: int = 8,
) -> dict[str, np.ndarray]:
    """Compute per-cell summary statistics for each chromosome.

    Run this metric after :func:`~snapatac2.pp.import_fragments` has attached
    fragment metadata to the AnnData object. The returned dictionary contains one
    vector per chromosome, with one value per cell.

    Anti-Patterns
    -------------
    - Do NOT call this function on an AnnData object that lacks imported
      fragments.
    - Do NOT pass this result directly as a matrix without aligning chromosome
      keys; dictionary order is not a biological ordering guarantee.

    Parameters
    ----------
    adata : snapatac2._snapatac2.AnnData or list[snapatac2._snapatac2.AnnData]
        AnnData object, or a list of AnnData objects, with imported fragments.
        When a list is provided, compute chromosome summaries for each object in
        parallel.
    mode : {"sum", "mean", "count"}, default: "count"
        Statistic to compute per chromosome and per cell. Use "sum" for summed
        values, "mean" for mean values, or "count" for counts.
    n_jobs : int, default: 8
        Number of jobs to run when `adata` is a list. If `n_jobs=-1`, use all
        available CPUs.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping from chromosome name to a one-dimensional array of per-cell
        summary values. When `adata` is a list, returns a list of such mappings.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(
    ...     snap.datasets.pbmc500(downsample=True),
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> chrom_counts = snap.metrics.summary_by_chrom(data, mode="count")
    >>> chrom_counts["chr1"].shape[0] == data.n_obs
    True
    """
    if isinstance(adata, list):
        return snapatac2._utils.anndata_par(
            adata,
            lambda x: summary_by_chrom(x, mode=mode),
            n_jobs=n_jobs,
        )
    else:
        return internal.summary_by_chrom(adata, mode)
