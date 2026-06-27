from __future__ import annotations

from pathlib import Path
from snapatac2._snapatac2 import AnnData, AnnDataSet
import snapatac2._snapatac2 as _snapatac2
import logging
from snapatac2.genome import Genome


def macs3(
    adata: AnnData | AnnDataSet,
    *,
    groupby: str | list[str] | None = None,
    qvalue: float = 0.05,
    call_broad_peaks: bool = False,
    broad_cutoff: float = 0.1,
    replicate: str | list[str] | None = None,
    replicate_qvalue: float | None = None,
    max_frag_size: int | None = None,
    selections: set[str] | None = None,
    nolambda: bool = False,
    shift: int = -100,
    extsize: int = 200,
    min_len: int | None = None,
    blacklist: Path | None = None,
    key_added: str = "macs3",
    tempdir: Path | None = None,
    inplace: bool = True,
    n_jobs: int = 8,
) -> dict[str, "polars.DataFrame"] | None:
    """Call open chromatin peaks with MACS3.

    Use this function to call peaks for all cells, for each cell group, or for
    reproducible group-by-replicate pseudobulk profiles.

    Anti-Patterns
    -------------
    - Do NOT set `replicate` without `groupby`; reproducible peak calling is
      defined within groups.
    - Do NOT use `call_broad_peaks=True` only to relax peak stringency; instead,
      raise `qvalue` when broad/nested peak structure is not needed.
    - Do NOT pass a blacklist in a non-BED coordinate system; it must match the
      genome used to generate fragments.

    Parameters
    ----------
    adata : AnnData | AnnDataSet
        Annotated fragment/count object with reference sequence metadata in
        `adata.uns["reference_sequences"]`.
    groupby : str | list[str] | None
        Grouping key in `adata.obs`, one group label per cell, or None to call
        peaks on all cells together.
    qvalue : float
        MACS3 q-value cutoff for peak calling.
    call_broad_peaks : bool
        If True, call broad peaks. The broad peak calling process
        utilizes two distinct cutoffs to discern broader, weaker peaks (`broad_cutoff`)
        and narrower, stronger peaks (`qvalue`), which are subsequently nested to
        provide a nested peak landscape.
    broad_cutoff : float
        MACS3 q-value cutoff for broad peaks.
    replicate : str | list[str] | None
        Replicate key in `adata.obs`, one replicate label per cell, or None.
    replicate_qvalue : float | None
        MACS3 q-value cutoff for replicate-level calls. If None, reuse `qvalue`.
    max_frag_size : int | None
        Maximum fragment size retained for peak calling. If None, use all
        fragments.
    selections : set[str] | None
        Subset of group names to call. Ignored when `groupby` is None.
    nolambda : bool
        If True, disable MACS3 local lambda bias correction.
    shift : int
        MACS3 shift size.
    extsize : int
        MACS3 extension size.
    min_len : int | None
        Minimum peak length. If None, use `extsize`.
    blacklist : pathlib.Path | None
        BED file of regions to remove from called peaks.
    key_added : str
        Key prefix in `adata.uns` used to store peak tables.
    tempdir : pathlib.Path | None
        Directory in which to create temporary files. If None, use the system
        temporary directory.
    inplace : bool
        If True, store peak tables in `adata.uns`; if False, return them.
    n_jobs : int
        Number of worker processes for grouped peak calling.

    Returns
    -------
    dict[str, 'polars.DataFrame'] | None
        If `inplace=True`, stores peak tables in `adata.uns[key_added]` for
        grouped calls or `adata.uns[key_added + "_pseudobulk"]` for bulk calls,
        then returns None. If `inplace=False`, returns peak tables keyed by group
        name, or a single table for bulk mode.

    See Also
    --------
    merge_peaks

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> peaks = snap.tl.macs3(adata, groupby="cell_type", inplace=False, n_jobs=1)
    >>> isinstance(peaks, dict)
    True
    """
    from MACS3.Signal.PeakDetect import PeakDetect
    from math import log
    import tempfile

    if isinstance(groupby, str):
        groupby = list(adata.obs[groupby])
    if replicate is not None and isinstance(replicate, str):
        replicate = list(adata.obs[replicate])

    # MACS3 options
    options = type("MACS3_OPT", (), {})()
    options.info = lambda _: None
    options.debug = lambda _: None
    options.warn = logging.warn
    options.name = "MACS3"
    options.bdg_treat = "t"
    options.bdg_control = "c"
    options.cutoff_analysis = False
    options.cutoff_analysis_file = "a"
    options.store_bdg = False
    options.do_SPMR = False
    options.trackline = False
    options.log_pvalue = None
    options.log_qvalue = log(qvalue, 10) * -1
    options.PE_MODE = False

    options.gsize = adata.uns["reference_sequences"][
        "reference_seq_length"
    ].sum()  # Estimated genome size
    options.maxgap = (
        30  # The maximum allowed gap between two nearby regions to be merged
    )
    options.minlen = extsize if min_len is None else min_len
    options.shift = shift
    options.nolambda = nolambda
    options.smalllocal = 1000
    options.largelocal = 10000
    options.call_summits = False if call_broad_peaks else True
    options.broad = call_broad_peaks
    if options.broad:
        options.log_broadcutoff = log(broad_cutoff, 10) * -1

    options.fecutoff = 1.0
    options.d = extsize
    options.scanwindow = 2 * options.d

    if groupby is None:
        peaks = _snapatac2.call_peaks_bulk(adata, options, max_frag_size)
        if inplace:
            adata.uns[key_added + "_pseudobulk"] = (
                peaks.to_pandas() if not adata.isbacked else peaks
            )
            return
        else:
            return peaks

    with tempfile.TemporaryDirectory(dir=tempdir) as tmpdirname:
        logging.info("Exporting fragments...")
        group_names = list(set(groupby))
        group_idx = {g: str(i) for i, g in enumerate(group_names)}
        fragments = _snapatac2.export_tags(
            adata,
            tmpdirname,
            [group_idx[x] for x in groupby],
            replicate,
            max_frag_size,
            selections,
        )

        def _call_peaks(tags):
            import tempfile

            tempfile.tempdir = tmpdirname  # Overwrite the default tempdir in MACS3
            merged, reps = _snapatac2.create_fwtrack_obj(tags)
            options.log_qvalue = log(qvalue, 10) * -1
            logging.getLogger().setLevel(
                logging.CRITICAL + 1
            )  # temporarily disable logging
            peakdetect = PeakDetect(treat=merged, opt=options)
            peakdetect.call_peaks()
            peakdetect.peaks.filter_fc(fc_low=options.fecutoff)
            merged = peakdetect.peaks

            others = []
            if replicate_qvalue is not None:
                options.log_qvalue = log(replicate_qvalue, 10) * -1
            for x in reps:
                peakdetect = PeakDetect(treat=x, opt=options)
                peakdetect.call_peaks()
                peakdetect.peaks.filter_fc(fc_low=options.fecutoff)
                others.append(peakdetect.peaks)

            logging.getLogger().setLevel(logging.INFO)  # enable logging
            return _snapatac2.find_reproducible_peaks(merged, others, blacklist)

        logging.info("Calling peaks...")
        if n_jobs == 1:
            peaks = [_call_peaks(x) for x in fragments.values()]
        else:
            peaks = _par_map(_call_peaks, [(x,) for x in fragments.values()], n_jobs)
        peaks = {group_names[int(k)]: v for k, v in zip(fragments.keys(), peaks)}
        if inplace:
            if adata.isbacked:
                adata.uns[key_added] = peaks
            else:
                adata.uns[key_added] = {k: v.to_pandas() for k, v in peaks.items()}
        else:
            return peaks


def merge_peaks(
    peaks: dict[str, "polars.DataFrame"],
    chrom_sizes: dict[str, int] | Genome,
    half_width: int = 250,
) -> "polars.DataFrame":
    """Merge group-specific peak calls into a non-overlapping peak set.

    Use this function after :func:`macs3` to create a shared peak universe for
    downstream counting and analysis.

    This function initially expands the summits of identified peaks by `half_width`
    on both sides. Following this expansion, it addresses the issue of overlapping
    peaks through an iterative process. The procedure begins by prioritizing the
    most significant peak, determined by the smallest p-value. This peak is retained,
    and any peak that overlaps with it is excluded. Subsequently, the same method
    is applied to the next most significant peak. This iteration continues until
    all peaks have been evaluated, resulting in a final list of non-overlapping
    peaks, each with a fixed width determined by the initial extension.

    Anti-Patterns
    -------------
    - Do NOT pass chromosome sizes from a different genome build than the peak
      coordinates.
    - Do NOT pass arbitrary BED-like tables unless they contain the columns
      produced by :func:`macs3`.

    Parameters
    ----------
    peaks : dict[str, polars.DataFrame]
        Peak tables keyed by group name.
    chrom_sizes : dict[str, int] | Genome
        Chromosome sizes, or a Genome object from which chromosome sizes are
        read.
    half_width : int
        Number of bases added on each side of each summit before overlap
        resolution.

    Returns
    -------
    'polars.DataFrame'
        Merged, non-overlapping peak table.

    See Also
    --------
    macs3

    Examples
    --------
    >>> import snapatac2 as snap
    >>> adata = snap.datasets.pbmc5k(type="annotated_h5ad")
    >>> peaks = snap.tl.macs3(adata, groupby="cell_type", inplace=False, n_jobs=1)
    >>> merged = snap.tl.merge_peaks(peaks, snap.genome.hg38)
    >>> merged.height > 0
    True
    """
    import pandas as pd
    import polars as pl

    chrom_sizes = (
        chrom_sizes.chrom_sizes if isinstance(chrom_sizes, Genome) else chrom_sizes
    )
    peaks = {
        k: pl.from_pandas(v) if isinstance(v, pd.DataFrame) else v
        for k, v in peaks.items()
    }
    return _snapatac2.py_merge_peaks(peaks, chrom_sizes, half_width)


def _par_map(mapper, args, nprocs):
    import time
    from multiprocess import get_context
    from tqdm import tqdm

    with get_context("spawn").Pool(nprocs) as pool:
        procs = set(pool._pool)
        jobs = [(i, pool.apply_async(mapper, x)) for i, x in enumerate(args)]
        results = []
        with tqdm(total=len(jobs)) as pbar:
            while len(jobs) > 0:
                if any(map(lambda p: not p.is_alive(), procs)):
                    raise RuntimeError("Some worker process has died unexpectedly.")

                remaining = []
                for i, job in jobs:
                    if job.ready():
                        results.append((i, job.get()))
                        pbar.update(1)
                    else:
                        remaining.append((i, job))
                jobs = remaining
                time.sleep(0.5)
        return [x for _, x in sorted(results, key=lambda x: x[0])]
