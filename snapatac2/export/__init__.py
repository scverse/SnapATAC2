from __future__ import annotations

from typing import Literal
from pathlib import Path

import snapatac2._snapatac2 as internal
from snapatac2._utils import get_file_format

def export_fragments(
    adata: internal.AnnData | internal.AnnDataSet,
    groupby: str | list[str],
    selections: list[str] | None = None,
    ids: str | list[str] | None = None,
    min_frag_length: int | None = None,
    max_frag_length: int | None = None,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bed.zst",
    compression: Literal["gzip", "zstandard"] | None = None,
    compression_level: int | None = None,
) -> dict[str, str]:
    """Export grouped fragments to BED-like files.

    Use this function after importing fragments to write one fragment file per
    cell group. Provide group labels directly or pass the name of an `.obs`
    column. The output filenames are constructed as
    `{prefix}{group_name}{suffix}` inside `out_dir`.

    Anti-Patterns
    -------------
    - Do NOT pass an AnnData object without fragment metadata created by
      :func:`~snapatac2.pp.import_fragments`.
    - Do NOT pass `groupby` or `ids` lists with lengths different from
      `adata.n_obs`.

    Parameters
    ----------
    adata : snapatac2._snapatac2.AnnData or snapatac2._snapatac2.AnnDataSet
        Annotated data object with `n_obs` cells and fragment metadata.
    groupby : str or list[str]
        Group assignment for each cell. If a string, values are read from
        `adata.obs[groupby]`. If a list, it must contain one group label per
        cell in observation order.
    selections : list[str] or None, default: None
        Group names to export. If None, export every group found in `groupby`.
    ids : str, list[str], or None, default: None
        Cell IDs to write into BED records. If a string, values are read from
        `adata.obs[ids]`. If a list, it must contain one ID per cell. If None,
        `adata.obs_names` are used.
    min_frag_length : int or None, default: None
        Minimum fragment length to export. If None, do not apply a minimum.
    max_frag_length : int or None, default: None
        Maximum fragment length to export. If None, do not apply a maximum.
    out_dir : pathlib.Path, default: "./"
        Directory where output files are written.
    prefix : str, default: ""
        Text prepended to each output filename.
    suffix : str, default: ".bed.zst"
        Text appended to each output filename. Used to infer compression when
        `compression=None`.
    compression : {"gzip", "zstandard"} or None, default: None
        Compression codec. If None, infer it from `suffix`.
    compression_level : int or None, default: None
        Compression level. Use 1-9 for gzip or 1-22 for zstandard. If None, use
        the backend default: 6 for gzip or 3 for zstandard.

    Returns
    -------
    dict[str, str]
        Mapping from group name to output filename.

    See Also
    --------
    export_coverage

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.pp.import_fragments(
    ...     snap.datasets.pbmc500(downsample=True),
    ...     chrom_sizes=snap.genome.hg38,
    ...     sorted_by_barcode=False,
    ... )
    >>> snap.ex.export_fragments(
    ...     data,
    ...     groupby=["pbmc500"] * data.n_obs,
    ...     out_dir="fragments_by_group",
    ... )
    {'pbmc500': 'fragments_by_group/pbmc500.bed.zst'}
    """
    if isinstance(groupby, str):
        groupby = adata.obs[groupby]
    if selections is not None:
        selections = set(selections)
    
    if ids is None:
        ids = adata.obs_names
    elif isinstance(ids, str):
        ids = adata.obs[ids]

    if compression is None:
        _, compression = get_file_format(suffix)

    return internal.export_fragments(
        adata, list(ids), list(groupby), out_dir, prefix, suffix, selections, 
        min_frag_length, max_frag_length, compression, compression_level,
    )

def export_coverage(
    adata: internal.AnnData | internal.AnnDataSet,
    groupby: str | list[str],
    selections: list[str] | None = None,
    bin_size: int = 10,
    blacklist: Path | None = None,
    normalization: Literal["RPKM", "CPM", "BPM"] | None = "RPKM",
    include_for_norm: list[str] | Path = None,
    exclude_for_norm: list[str] | Path = None,
    min_frag_length: int | None = None,
    max_frag_length: int | None = 2000,
    counting_strategy: Literal['fragment', 'insertion'] = 'fragment',
    smooth_base: int | None = None,
    out_dir: Path = "./",
    prefix: str = "",
    suffix: str = ".bw",
    output_format: Literal["bedgraph", "bigwig"] | None = None,
    compression: Literal["gzip", "zstandard"] | None = None,
    compression_level: int | None = None,
    tempdir: Path | None = None,
    n_jobs: int = 8,
) -> dict[str, str]:
    """Export grouped genome-wide coverage tracks.

    Use this function after importing fragments to write one bedGraph or bigWig
    coverage track per cell group. Coverage is counted in fixed-width genomic
    bins, optionally filtered by fragment length, smoothed, and normalized.
    Disable normalization with `normalization=None`.

    Anti-Patterns
    -------------
    - Do NOT pass an AnnData object without fragment metadata created by
      :func:`~snapatac2.pp.import_fragments`.
    - Do NOT use `include_for_norm` and `exclude_for_norm` as peak-calling
      filters; they only define which fragments contribute to normalization.
    - Do NOT rely on suffix inference for custom extensions; pass
      `output_format` and `compression` explicitly.

    .. image:: /_static/images/func+export_coverage.svg
        :align: center

    Parameters
    ----------
    adata : snapatac2._snapatac2.AnnData or snapatac2._snapatac2.AnnDataSet
        Annotated data object with `n_obs` cells and fragment metadata.
    groupby : str or list[str]
        Group assignment for each cell. If a string, values are read from
        `adata.obs[groupby]`. If a list, it must contain one group label per
        cell in observation order.
    selections : list[str] or None, default: None
        Group names to export. If None, export every group found in `groupby`.
    bin_size : int, default: 10
        Width, in bases, of each coverage bin.
    blacklist : pathlib.Path or None, default: None
        BED file of regions to exclude from coverage output.
    normalization : {"RPKM", "CPM", "BPM"} or None, default: "RPKM"
        Coverage normalization method. Use None to export raw counts. RPKM
        divides each bin by mapped reads in millions and bin length in kilobases;
        CPM divides by mapped reads in millions; BPM divides by the sum of all
        binned reads in millions.
    include_for_norm : list[str] or pathlib.Path, default: None
        Genomic intervals or BED file of intervals to include when computing the
        normalization denominator. If None, include all non-excluded fragments.
    exclude_for_norm : list[str] or pathlib.Path, default: None
        Genomic intervals or BED file of intervals to exclude when computing the
        normalization denominator. If a fragment overlaps both included and
        excluded intervals, it is excluded.
    min_frag_length : int or None, default: None
        Minimum fragment length to count. If None, do not apply a minimum.
    max_frag_length : int or None, default: 2000
        Maximum fragment length to count. If None, do not apply a maximum.
    counting_strategy : {"fragment", "insertion"}, default: "fragment"
        Counting mode. Use "fragment" to count overlapping fragments or
        "insertion" to count transposition insertion sites.
    smooth_base : int or None, default: None
        Width, in bases, of the smoothing window. If None, do not smooth.
    out_dir : pathlib.Path, default: "./"
        Directory where output files are written.
    prefix : str, default: ""
        Text prepended to each output filename.
    suffix : str, default: ".bw"
        Text appended to each output filename. Used to infer output format and
        compression when the corresponding arguments are None.
    output_format : {"bedgraph", "bigwig"} or None, default: None
        Coverage-track format. If None, infer it from `suffix`.
    compression : {"gzip", "zstandard"} or None, default: None
        Compression codec for compressed bedGraph output. If None, infer it from
        `suffix`.
    compression_level : int or None, default: None
        Compression level. Use 1-9 for gzip or 1-22 for zstandard. If None, use
        the backend default: 6 for gzip or 3 for zstandard.
    tempdir : pathlib.Path or None, default: None
        Directory for temporary files created during export. If None, use the
        system temporary directory.
    n_jobs : int, default: 8
        Number of worker threads. If `n_jobs <= 0`, use all available threads.

    Returns
    -------
    dict[str, str]
        Mapping from group name to output filename.

    See Also
    --------
    export_fragments

    Examples
    --------
    >>> import snapatac2 as snap
    >>> data = snap.read(snap.datasets.pbmc5k(type="annotated_h5ad"), backed='r')
    >>> snap.ex.export_coverage(
    ...     data,
    ...     groupby='cell_type',
    ...     selections=['Naive B'],
    ...     suffix='.bedgraph.zst',
    ... )
    {'Naive B': './Naive B.bedgraph.zst'}
    """
    if isinstance(groupby, str):
        groupby = adata.obs[groupby]
    if selections is not None:
        selections = set(selections)
    
    if output_format is None:
        output_format, inferred_compression = get_file_format(suffix)
        if output_format is None:
            raise ValueError("Output format cannot be inferred from suffix.")
        if compression is None:
            compression = inferred_compression

    n_jobs = None if n_jobs <= 0 else n_jobs
    return internal.export_coverage(
        adata, list(groupby), bin_size, out_dir, prefix, suffix, output_format, counting_strategy,
        selections, blacklist, normalization, include_for_norm, exclude_for_norm, min_frag_length,
        max_frag_length, smooth_base, compression, compression_level, tempdir, n_jobs,
    )
