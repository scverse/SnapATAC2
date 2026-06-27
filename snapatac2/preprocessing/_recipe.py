from __future__ import annotations

from pathlib import Path
import numpy as np

import snapatac2
import snapatac2._snapatac2

def filter_kwargs(func, kwargs_dict):
    """
    Keep only keyword arguments accepted by a target function.

    Use this helper inside recipes to forward a shared keyword dictionary to
    multiple functions without passing names that the target function cannot
    accept.

    Anti-Patterns
    -------------
    - Do NOT use this helper to validate required arguments; it only removes
      unsupported keyword names.

    Parameters
    ----------
    func
        Callable whose signature defines the accepted keyword names.
    kwargs_dict
        Dictionary of candidate keyword arguments.

    Returns
    -------
    dict
        Dictionary containing only entries from `kwargs_dict` whose keys appear
        in `func`'s signature.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> def target(a, b=1):
    ...     return a + b
    >>> snap.pp.filter_kwargs(target, {"a": 1, "b": 2, "c": 3})
    {'a': 1, 'b': 2}
    """
    import inspect
    signature = inspect.signature(func)
    accepted_params = set(signature.parameters.keys())
    filtered_kwargs = {k: v for k, v in kwargs_dict.items() if k in accepted_params}
    return filtered_kwargs

def recipe_10x_metrics(
    bam_file: Path,
    output_fragment_file: Path,
    output_h5ad_file: Path,
    peaks: Path | list[str] | None = None,
    **kwargs,
) -> dict:
    """
    Generate 10x-style ATAC QC metrics from a raw BAM file.

    Use this recipe to convert a BAM file to fragments, import the fragments into
    an h5ad file, compute targeting metrics, call peaks when needed, and summarize
    library-level QC values in one dictionary. Keyword arguments are forwarded to
    the individual preprocessing and metric functions when their signatures accept
    those names.

    Anti-Patterns
    -------------
    - Do NOT use this recipe when you only need an existing fragment file
      imported; call `snap.pp.import_fragments` directly.
    - Do NOT omit required downstream inputs such as `chrom_sizes` and
      `gene_anno`; they are supplied through `**kwargs` and used by the called
      functions.

    Parameters
    ----------
    bam_file
        Path to the input BAM file.
    output_fragment_file
        Path where the generated fragment file is written.
    output_h5ad_file
        Path where the intermediate AnnData object is written.
    peaks
        Path to a BED-like peak file or a list of peak regions formatted as
        `"chrom:start-end"`. If `None`, MACS3 is run to call peaks.
    **kwargs
        Additional arguments accepted by functions used in the recipe, including
        `make_fragment_file`, `import_fragments`, `metrics.tsse`, and related
        calls.

    Returns
    -------
    dict
        Nested dictionary containing sequencing, cell, library-complexity,
        mapping, and targeting QC metrics.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> bam_file = snap.datasets.pbmc500(type='bam')
    >>> metrics = snap.pp.recipe_10x_metrics(
    ...     bam_file,
    ...     'fragments.tsv.gz',
    ...     'data.h5ad',
    ...     barcode_tag='CB',
    ...     source='10x',
    ...     chrom_sizes=snap.genome.hg38,
    ...     gene_anno=snap.genome.hg38,
    ... )
    >>> sorted(metrics)
    ['Cells', 'Library Complexity', 'Mapping', 'Sequencing', 'Targeting']
    """
    qc = {
        "Sequencing": {},
        "Cells": {},
        "Library Complexity": {},
        "Mapping": {},
        "Targeting": {},
    }

    bam_qc = snapatac2.pp.make_fragment_file(
        bam_file,
        output_fragment_file,
        **filter_kwargs(snapatac2.pp.make_fragment_file, kwargs)
    )
    qc["Sequencing"]["Sequenced_reads"] = bam_qc["sequenced_reads"]
    qc["Sequencing"]["Sequenced_read_pairs"] = bam_qc["sequenced_read_pairs"]
    qc["Sequencing"]["Fraction_valid_barcode"] = bam_qc["frac_valid_barcode"]
    qc["Sequencing"]["Fraction_Q30_bases_in_read_1"] = bam_qc["frac_q30_bases_read1"]
    qc["Sequencing"]["Fraction_Q30_bases_in_read_2"] = bam_qc["frac_q30_bases_read2"]
    qc["Mapping"]["Fraction_confidently_mapped"] = bam_qc["frac_confidently_mapped"]
    qc["Mapping"]["Fraction_unmapped"] = bam_qc["frac_unmapped"]
    qc["Mapping"]["Fraction_nonnuclear"] = bam_qc["frac_nonnuclear"]
    qc["Mapping"]["Fraction_fragment_in_nucleosome_free_region"] = bam_qc["frac_fragment_in_nucleosome_free_region"]
    qc["Mapping"]["Fraction_fragment_flanking_single_nucleosome"] = bam_qc["frac_fragment_flanking_single_nucleosome"]
    qc["Library Complexity"]["Fraction_duplicates"] = bam_qc["frac_duplicates"]

    adata = snapatac2.pp.import_fragments(
        output_fragment_file,
        min_num_fragments=0,
        file=output_h5ad_file,
        **filter_kwargs(snapatac2.pp.import_fragments, kwargs),
    )
    snapatac2.metrics.tsse(adata, **filter_kwargs(snapatac2.metrics.tsse, kwargs))
    qc["Targeting"]["TSS_enrichment_score"] = adata.uns['library_tsse']
    qc["Targeting"]["Fraction_of_high-quality_fragments_overlapping_TSS"] = adata.uns['frac_overlap_TSS']

    if peaks is None:
        snapatac2.tl.macs3(adata, qvalue=0.001)
        peaks = [f"{row[0]}:{row[1]}-{row[2]}" for row in adata.uns['macs3_pseudobulk'].iter_rows()]
    else:
        if not isinstance(peaks, list):
            p = []
            with open(peaks, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        items = line.strip().split()
                        p.append(f'{items[0]}:{items[1]}-{items[2]}')
            peaks = p
    qc["Targeting"]["Number_of_peaks"] = len(peaks)
    qc["Targeting"]["Fraction_of_genome_in_peaks"] = snapatac2._snapatac2.total_size_of_peaks(peaks) / adata.uns['reference_sequences']['reference_seq_length'].sum()

    snapatac2.metrics.frip(adata, {"n_frag_overlap_peak": peaks}, normalized=False)
    qc["Targeting"]["Fraction_of_high-quality_fragments_overlapping_peaks"] = adata.obs['n_frag_overlap_peak'].sum() / adata.obs['n_fragment'].sum()

    cell_idx = snapatac2.pp.call_cells(adata, use_rep="n_frag_overlap_peak", inplace=False)
    n_cells = len(cell_idx)
    n_fragment = adata.obs['n_fragment'].to_numpy()
    qc["Cells"]["Number_of_cells"] = n_cells
    is_paired = bam_qc["sequenced_read_pairs"] > 0
    if is_paired:
        qc["Cells"]["Mean_raw_read_pairs_per_cell"] = bam_qc["sequenced_read_pairs"] / n_cells
    else:
        qc["Cells"]["Mean_raw_read_pairs_per_cell"] = bam_qc["sequenced_reads"] / n_cells
    qc["Cells"]["Median_high-quality_fragments_per_cell"] = np.median(n_fragment[cell_idx])
    qc["Cells"]["Fraction of high-quality fragments in cells"] = n_fragment[cell_idx].sum() / n_fragment.sum()

    adata.subset(cell_idx)
    frip = snapatac2.metrics.frip(adata, {"overlap_peak": peaks}, normalized=False, count_as_insertion=True, inplace=False)
    n_fragment = adata.obs['n_fragment'].to_numpy()
    if is_paired:
        qc["Cells"]["Fraction_of_transposition_events_in_peaks_in_cells"] = np.sum(frip['overlap_peak']) / (n_fragment.sum() * 2)
    else:
        qc["Cells"]["Fraction_of_transposition_events_in_peaks_in_cells"] = np.sum(frip['overlap_peak']) / n_fragment.sum()

    adata.close()
    return qc
