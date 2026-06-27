from __future__ import annotations

from typing import Literal
from typeguard import typechecked
from pathlib import Path
import pooch

from snapatac2._snapatac2 import read_motifs, PyDNAMotif

# This is a global variable used to store all datasets. It is initialized only once
# when the data is requested.
_datasets = None

def register_datasets():
    """Create or return the cached SnapATAC2 example-dataset registry.

    Use this helper indirectly through dataset accessors such as `pbmc500` or
    `cre_HEA`. Set the `SNAP_DATA_DIR` environment variable before calling a
    dataset accessor to override the local cache directory.
    """
    global _datasets
    if _datasets is None:
        _datasets = pooch.create(
            path=pooch.os_cache("snapatac2"),
            base_url="http://renlab.sdsc.edu/kai/public_datasets/",
            env="SNAP_DATA_DIR",  # The user can overwrite the storage path by setting this environment variable.
            # The registry specifies the files that can be fetched
            registry={
                "atac_pbmc_500_fastqs.tar": "sha256:5897a4790d2841eff69c85b4bef2825166dd0cc2587da91f42eeed09000c5f47",
                "atac_pbmc_500.bam": "sha256:2fac56ca45186943a1daf9da71aed42263ad43a9428f2388fa5f3bcf6d2754ff",
                "atac_pbmc_500.tsv.gz": "sha256:196c5d7ee0169957417e9f4d5502abf1667ef99453328f8d290d4a7f3b205c6c",
                "atac_pbmc_500_downsample.tsv.gz": "sha256:6053cf4578a140bfd8ce34964602769dc5f5ec6b25ba4f2db23cdbd4681b0e2f",

                "atac_pbmc_5k.tsv.gz": "sha256:5fe44c0f8f76ce1534c1ae418cf0707ca5ef712004eee77c3d98d2d4b35ceaec",
                "atac_pbmc_5k.h5ad": "sha256:92ae7f185cdec26517fd8d5acb60b2ce92c71e0ace824de35589c6d7942cab06",
                "atac_pbmc_5k_annotated.h5ad": "sha256:592f1551c27d0cfe4d81e7febad624d6b7d3ebf977b0c3ea64e06b3f3d76f078",

                "colon_transverse.tar": "sha256:18c56bf405ec0ef8e0e2ea31c63bf2299f21bcb82c67f46e8f70f8d71c65ae0e",
                "HEA_cCRE.bed.gz": "sha256:d69ae94649201cd46ffdc634852acfccc317196637c1786aba82068618001408",

                "10x-Multiome-Pbmc10k-ATAC.h5ad": "sha256:24d030fb7f90453a0303b71a1e3e4e7551857d1e70072752d7fff9c918f77217",
                "10x-Multiome-Pbmc10k-RNA.h5ad": "sha256:a25327acff48b20b295c12221a84fd00f8f3f486ff3e7bd090fdef241b996a22",
                "pbmc_10k_atac.tsv.gz": "md5:a959ef83dfb9cae6ff73ab0147d547d1",

                # TF motifs
                "cisBP_human.meme": "sha256:8bf995450258e61cb1c535d5bf9656d580eb68ba68893fa36b77d17ee0730579",
                "Meuleman_2020.meme": "sha256:400dd60ca61dc8388aa0942b42c95920aad7f6bedf5324005cee7e84bcf5b6d0",

                # Genome files
                "gencode_v41_GRCh37.gff3.gz": "sha256:df96d3f0845127127cc87c729747ae39bc1f4c98de6180b112e71dda13592673",
                "gencode_v41_GRCh37.fa.gz": "sha256:ac73947d38df63ccb00724520a5c31d880c1ca423702ca7ccb7e6c2182a362d9",
                #"gencode_v41_GRCh37.fa.gz": "sha256:94330d402e53cf39a1fef6c132e2500121909c2dfdce95cc31d541404c0ed39e",
                "gencode_v41_GRCh38.gff3.gz": "sha256:b82a655bdb736ca0e463a8f5d00242bedf10fa88ce9d651a017f135c7c4e9285",
                "gencode_v41_GRCh38.fa.gz": "sha256:4fac949d7021cbe11117ddab8ec1960004df423d672446cadfbc8cca8007e228",
                "gencode_vM25_GRCm38.gff3.gz": "sha256:e8ed48bef6a44fdf0db7c10a551d4398aa341318d00fbd9efd69530593106846",
                "gencode_vM25_GRCm38.fa.gz": "sha256:617b10dc7ef90354c3b6af986e45d6d9621242b64ed3a94c9abeac3e45f18c17",
                "gencode_vM30_GRCm39.gff3.gz": "sha256:6f433e2676e26569a678ce78b37e94a64ddd50a09479e433ad6f75e37dc82e48",
                "gencode_vM30_GRCm39.fa.gz": "sha256:3b923c06a0d291fe646af6bf7beaed7492bf0f6dd5309d4f5904623cab41b0aa",
            },
            urls={
                "atac_pbmc_500_fastqs.tar": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/atac_pbmc_500_nextgem_fastqs.tar",
                "atac_pbmc_500.tsv.gz": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/atac_pbmc_500_nextgem_fragments.tsv.gz",
                "atac_pbmc_500.bam": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_500_nextgem/atac_pbmc_500_nextgem_possorted_bam.bam",
                "atac_pbmc_500_downsample.tsv.gz": "https://osf.io/download/wjv4b",

                "atac_pbmc_5k.tsv.gz": "https://cf.10xgenomics.com/samples/cell-atac/2.0.0/atac_pbmc_5k_nextgem/atac_pbmc_5k_nextgem_fragments.tsv.gz", 
                "atac_pbmc_5k.h5ad": "https://osf.io/download/rj9nc/",
                "atac_pbmc_5k_annotated.h5ad": "https://osf.io/download/e9vc3/",

                "10x-Multiome-Pbmc10k-ATAC.h5ad": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/165dfb5c-c557-42a0-bd21-1276d4d7b23e?a=758c37e5-4832-4c91-af89-9a1a83a051b3",
                "10x-Multiome-Pbmc10k-RNA.h5ad": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/d079a087-2913-4e29-979e-638e5932bd8c?a=758c37e5-4832-4c91-af89-9a1a83a051b3", 
                "pbmc_10k_atac.tsv.gz": "https://cf.10xgenomics.com/samples/cell-arc/1.0.0/pbmc_granulocyte_sorted_10k/pbmc_granulocyte_sorted_10k_atac_fragments.tsv.gz",

                "colon_transverse.tar": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/eaa46151-a73f-4ef5-8b05-9648c8d1efda?a=758c37e5-4832-4c91-af89-9a1a83a051b3", 
                "HEA_cCRE.bed.gz": "https://data.mendeley.com/api/datasets/dr2z4jbcx3/draft/files/91f93222-1a24-49a5-92e3-d9105ec53f91?a=758c37e5-4832-4c91-af89-9a1a83a051b3",

                "cisBP_human.meme": "https://osf.io/download/uk6vn",
                "Meuleman_2020.meme": "https://osf.io/download/6uet5/",

                "gencode_v41_GRCh37.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/gencode.v41lift37.basic.annotation.gff3.gz",
                "gencode_v41_GRCh37.fa.gz": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
                #"gencode_v41_GRCh37.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh37_mapping/GRCh37.primary_assembly.genome.fa.gz",
                "gencode_v41_GRCh38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/gencode.v41.basic.annotation.gff3.gz",
                "gencode_v41_GRCh38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_41/GRCh38.primary_assembly.genome.fa.gz",
                "gencode_vM25_GRCm38.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/gencode.vM25.basic.annotation.gff3.gz",
                "gencode_vM25_GRCm38.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M25/GRCm38.primary_assembly.genome.fa.gz",
                "gencode_vM30_GRCm39.gff3.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/gencode.vM30.basic.annotation.gff3.gz",
                "gencode_vM30_GRCm39.fa.gz": "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_mouse/release_M30/GRCm39.primary_assembly.genome.fa.gz",
            },
        )
    return _datasets

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
    datasets = register_datasets()
    if type == 'fragment':
        if downsample:
            return Path(datasets.fetch("atac_pbmc_500_downsample.tsv.gz", progressbar=True))
        else:
            return Path(datasets.fetch("atac_pbmc_500.tsv.gz", progressbar=True))
    elif type == 'bam':
        return Path(datasets.fetch("atac_pbmc_500.bam", progressbar=True))
    elif type == 'fastq':
        return [Path(f) for f in datasets.fetch("atac_pbmc_500_fastqs.tar", processor=pooch.Untar(), progressbar=True)]

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
    datasets = register_datasets()
    if type == "fragment":
        return Path(datasets.fetch("atac_pbmc_5k.tsv.gz", progressbar=True))
    elif type == "h5ad":
        return Path(datasets.fetch("atac_pbmc_5k.h5ad", progressbar=True))
    elif type == "annotated_h5ad":
        return Path(datasets.fetch("atac_pbmc_5k_annotated.h5ad", progressbar=True))

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
    datasets = register_datasets()
    if modality == 'RNA':
        return Path(datasets.fetch("10x-Multiome-Pbmc10k-RNA.h5ad"))
    elif modality == 'ATAC':
        if type == 'fragment':
            return Path(datasets.fetch("pbmc_10k_atac.tsv.gz"))
        else:
            return Path(datasets.fetch("10x-Multiome-Pbmc10k-ATAC.h5ad"))

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
    files = register_datasets().fetch("colon_transverse.tar", progressbar=True, processor = pooch.Untar())
    return [(fl.split("/")[-1].split("_rep1_fragments")[0], Path(fl)) for fl in files]

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
    return Path(register_datasets().fetch("HEA_cCRE.bed.gz"))

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
    motifs = read_motifs(register_datasets().fetch("cisBP_human.meme"))
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
    motifs = read_motifs(register_datasets().fetch("Meuleman_2020.meme"))
    for motif in motifs:
        motif.name = motif.id.split('_')[0]
        motif.family = motif.id.split('+')[-1]
    return motifs
