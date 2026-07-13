from __future__ import annotations

from typing import Literal
from typeguard import typechecked
from pathlib import Path
import pooch

from snapatac2._snapatac2 import PyDNAMotif
from snapatac2._datasets_impl import load

@typechecked
def pbmc500(type: Literal['fastq', 'bam', 'fragment'] = 'fragment', downsample: bool = False) -> Path | list[Path]:
    """Fetch the 10x Genomics 500 PBMC scATAC-seq example dataset.

    Use this helper to download and cache the fragment, BAM, or FASTQ files for
    a small PBMC dataset suitable for tutorials and smoke tests. Set the
    `SNAP_DATA_DIR` environment variable before calling this function to control
    where downloaded files are cached.

    Anti-Patterns
    -------------
    - Do NOT use the default full fragment file for fast examples; pass
      `downsample=True` when a small fragment file is sufficient.
    - Do NOT set `downsample=True` with `type="bam"` or `type="fastq"`; the
      downsampled file is only available for `type="fragment"`.

    Parameters
    ----------
    type : {"fastq", "bam", "fragment"}, default: "fragment"
        File type to fetch. Use "fragment" for a fragments TSV.GZ file, "bam"
        for the position-sorted BAM file, or "fastq" for the extracted FASTQ
        files from the downloaded archive.
    downsample : bool, default: False
        If True and `type="fragment"`, fetch the smaller downsampled fragments
        file instead of the full fragments file.

    Returns
    -------
    pathlib.Path or list[pathlib.Path]
        Path to the requested fragment or BAM file. For `type="fastq"`, returns
        a list of paths to the extracted FASTQ files.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> fragment_file = snap.datasets.pbmc500(downsample=True)
    >>> fragment_file.name
    'atac_pbmc_500_downsample.tsv.gz'
    """
    if type == 'fragment':
        file = "atac_pbmc_500_downsample.tsv.gz" if downsample else "atac_pbmc_500.tsv.gz"
        return load('atac_pbmc_500', file=file)
    elif type == 'bam':
        return load('atac_pbmc_500', file="atac_pbmc_500.bam")
    elif type == 'fastq':
        return load('atac_pbmc_500', file="atac_pbmc_500_fastqs.tar", processor=pooch.Untar())

@typechecked
def pbmc5k(type: Literal['fragment', 'h5ad', 'annotated_h5ad'] = 'fragment') -> Path:
    """Fetch the 10x Genomics 5k PBMC scATAC-seq example dataset.

    Use this helper to download and cache a fragments file, a preprocessed h5ad
    file, or an annotated h5ad file for PBMC analysis examples. Set the
    `SNAP_DATA_DIR` environment variable before calling this function to control
    where downloaded files are cached.

    Anti-Patterns
    -------------
    - Do NOT pass the returned h5ad path to fragment-import functions; use
      `snap.read(...)` for `type="h5ad"` and `type="annotated_h5ad"`.

    Parameters
    ----------
    type : {"fragment", "h5ad", "annotated_h5ad"}, default: "fragment"
        Dataset representation to fetch. Use "fragment" for a fragments TSV.GZ
        file, "h5ad" for a preprocessed AnnData file, or "annotated_h5ad" for a
        preprocessed AnnData file with cell annotations.

    Returns
    -------
    pathlib.Path
        Path to the requested cached dataset file.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> h5ad_file = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> data = snap.read(h5ad_file, backed="r")
    >>> data.n_obs > 0
    True
    """
    if type == "fragment":
        return load('atac_pbmc_5k', file="atac_pbmc_5k.tsv.gz")
    elif type == "h5ad":
        return load('atac_pbmc_5k', file="atac_pbmc_5k.h5ad")
    elif type == "annotated_h5ad":
        return load('atac_pbmc_5k', file="atac_pbmc_5k_annotated.h5ad")

@typechecked
def pbmc10k_multiome(
    modality: Literal['ATAC', 'RNA'] = 'RNA',
    type: Literal['fragment', 'h5ad'] = 'h5ad',
) -> Path:
    """Fetch the 10x Genomics 10k PBMC multiome example dataset.

    Use this helper to download and cache the paired RNA and ATAC example data
    for multiome workflows. RNA is available as h5ad only; ATAC is available as
    either h5ad or fragments.

    Anti-Patterns
    -------------
    - Do NOT request `modality="RNA"` with `type="fragment"`; RNA returns the
      RNA h5ad file regardless of `type`.

    Parameters
    ----------
    modality : {"ATAC", "RNA"}, default: "RNA"
        Modality to fetch. Use "ATAC" for chromatin accessibility data or
        "RNA" for gene-expression data.
    type : {"fragment", "h5ad"}, default: "h5ad"
        ATAC representation to fetch. This parameter is ignored when
        `modality="RNA"` because only the RNA h5ad file is available.

    Returns
    -------
    pathlib.Path
        Path to the requested cached dataset file.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> atac_file = snap.datasets.pbmc10k_multiome(modality="ATAC", type="h5ad")
    >>> rna_file = snap.datasets.pbmc10k_multiome(modality="RNA")
    >>> atac_file.suffix == rna_file.suffix == ".h5ad"
    True
    """
    if modality == 'RNA':
        return load('pbmc_10k_multiome', file="10x-Multiome-Pbmc10k-RNA.h5ad")
    elif modality == 'ATAC':
        if type == 'fragment':
            return load('pbmc_10k_multiome', file="pbmc_10k_atac.tsv.gz")
        else:
            return load('pbmc_10k_multiome', file="10x-Multiome-Pbmc10k-ATAC.h5ad")

def colon() -> list[tuple[str, Path]]:
    """Fetch five transverse colon scATAC-seq fragment datasets.

    Use this helper to download and extract the colon transverse sample archive
    from [Zhang21]_. Each returned tuple provides a sample name and the cached
    fragment-file path for that sample.

    Returns
    -------
    list[tuple[str, Path]]
        Tuples containing `(sample_name, fragment_file)`, where `sample_name` is
        a string parsed from the archive filename and `fragment_file` is a
        pathlib.Path pointing to a fragments file.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> samples = snap.datasets.colon()
    >>> name, fragment_file = samples[0]
    >>> isinstance(name, str) and fragment_file.exists()
    True
    """
    files = load('colon_transverse', file="colon_transverse.tar", processor=pooch.Untar())
    return [(f.name.split("_rep1_fragments")[0], f) for f in files]

def cre_HEA() -> Path:
    """Fetch the curated human colon cis-regulatory element BED file.

    Use this helper to download and cache the HEA cCRE set from [Zhang21]_ when
    computing FRiP or overlap statistics against curated regulatory regions.

    Returns
    -------
    pathlib.Path
        Path to the gzipped BED file containing the cis-regulatory elements.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> cre_file = snap.datasets.cre_HEA()
    >>> cre_file.name
    'HEA_cCRE.bed.gz'
    """
    return load('HEA_cCRE', file="HEA_cCRE.bed.gz")

def cis_bp(unique: bool = True) -> list[PyDNAMotif]:
    """Fetch CIS-BP transcription factor motifs for motif analysis.

    Use these motifs from [Weirauch14]_ to scan genomic sequences or run motif
    enrichment. When `unique=True`, this function keeps only the highest
    information-content motif for each transcription factor name.

    Anti-Patterns
    -------------
    - Do NOT set `unique=False` when downstream code expects one motif per
      transcription factor; CIS-BP can contain multiple motifs per factor.

    Parameters
    ----------
    unique : bool, default: True
        If True, return one motif per transcription factor by selecting the motif
        with the highest information content. If False, return all CIS-BP motifs.

    Returns
    -------
    list[PyDNAMotif]
        Motif objects with `name` set to the transcription factor name parsed
        from the motif identifier.

    See Also
    --------
    :func:`~snapatac2.tl.motif_enrichment`: compute motif enrichment.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> motifs = snap.datasets.cis_bp(unique=True)
    >>> len(motifs) > 0
    True
    """
    motifs = load('cisBP_human')
    for motif in motifs:
        motif.name = motif.id.split('+')[0]
    if unique:
        unique_motifs = {}
        for motif in motifs:
            name = motif.name
            if (
                    name not in unique_motifs or 
                    unique_motifs[name].info_content() < motif.info_content()
               ):
               unique_motifs[name] = motif
        motifs = list(unique_motifs.values())
    return motifs

def Meuleman_2020() -> list[PyDNAMotif]:
    """Fetch grouped transcription factor motifs from Meuleman 2020.

    Use these curated motifs from [Meuleman20]_ to scan genomic sequences or run
    motif enrichment. Each returned motif has `name` set to the parsed motif name
    and `family` set to the motif-family label parsed from the motif identifier.

    Returns
    -------
    list[PyDNAMotif]
        Motif objects with populated `name` and `family` attributes.

    See Also
    --------
    :func:`~snapatac2.tl.motif_enrichment`: compute motif enrichment.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> motifs = snap.datasets.Meuleman_2020()
    >>> hasattr(motifs[0], "family")
    True
    """
    motifs = load('Meuleman_2020')
    for motif in motifs:
        motif.name = motif.id.split('_')[0]
        motif.family = motif.id.split('+')[-1]
    return motifs
