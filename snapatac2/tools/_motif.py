from __future__ import annotations

from typing import Literal
import numpy as np
from pathlib import Path
import logging

from snapatac2._snapatac2 import PyDNAMotif
from snapatac2._utils import fetch_seq
from snapatac2.genome import Genome
from snapatac2.tools._diff import _p_adjust_bh

def motif_enrichment(
    motifs: list[PyDNAMotif],
    regions: dict[str, list[str]],
    genome_fasta: Path | Genome,
    background: list[str] | None = None,
    method: Literal['binomial', 'hypergeometric'] | None = None,
) -> dict[str, 'polars.DataFrame']:
    """
    Test transcription factor motifs for enrichment in region sets.

    Use this function to compare motif occurrence in foreground region groups
    against either an explicit background or the union of all foreground regions.

    Anti-Patterns
    -------------
    - Do NOT use `method="hypergeometric"` with foreground regions that are not
      contained in `background`.
    - Do NOT pass region strings from a genome build different from
      `genome_fasta`.

    Parameters
    ----------
    motifs : list[PyDNAMotif]
        Motifs to scan in foreground and background sequences.
    regions : dict[str, list[str]]
        Foreground genomic regions keyed by group name. Region strings must use
        `chrom:start-end` coordinates.
    genome_fasta : pathlib.Path | Genome
        Genome FASTA path, or a Genome object containing a FASTA path.
    background : list[str] | None
        Background regions. If None, use the union of all foreground regions.
    method : {"binomial", "hypergeometric"} | None
        Statistical test. If None, use `"hypergeometric"` when `background` is
        None and `"binomial"` otherwise.

    Returns
    -------
    dict[str, polars.DataFrame]
        Enrichment tables keyed by group name. Each table contains motif id,
        name, family, log2 fold change, p-value, and adjusted p-value.

    Examples
    --------
    >>> import snapatac2 as snap
    >>> motifs = snap.datasets.cis_bp(unique=True)
    >>> regions = {"set1": ["chr1:10000-10200", "chr1:20000-20200"]}
    >>> result = snap.tl.motif_enrichment(motifs[:2], regions, snap.genome.hg38)
    >>> list(result)
    ['set1']
    """
    from pyfaidx import Fasta
    from tqdm import tqdm
    from scipy.stats import binom, hypergeom
    from math import log2
    import polars as pl

    def count_occurrence(query, idx_map, bound):
        return sum(bound[idx_map[q]] for q in query)

    if method is None:
        method = "hypergeometric" if background is None else "binomial"

    all_regions = set(p for ps in regions.values() for p in ps)
    if background is not None:
        for p in background:
            all_regions.add(p)
    all_regions = list(all_regions)
    region_to_idx = dict(map(lambda x: (x[1], x[0]), enumerate(all_regions)))

    logging.info("Fetching {} sequences ...".format(len(all_regions)))
    genome = genome_fasta.fasta if isinstance(genome_fasta, Genome) else str(genome_fasta)
    genome = Fasta(genome, one_based_attributes=False)
    sequences = [fetch_seq(genome, region) for region in all_regions]

    motif_id = []
    motif_name = []
    motif_family = []
    group_name = []
    fold_change = []
    n_fg = []
    N_fg = []
    n_bg = []
    N_bg = []
    logging.info("Computing enrichment ...")
    for motif in tqdm(motifs):
        bound = motif.with_nucl_prob().exists(sequences)
        if background is None:
            total_bg = len(bound)
            bound_bg = sum(bound)
        else:
            total_bg = len(background)
            bound_bg = count_occurrence(background, region_to_idx, bound)
        
        for key, val in regions.items():
            total_fg = len(val)
            bound_fg = count_occurrence(val, region_to_idx, bound)

            if bound_fg == 0:
                log_fc = 0 if bound_bg == 0 else float('-inf')
            else:
                log_fc = log2((bound_fg / total_fg) / (bound_bg / total_bg)) if bound_bg > 0 else float('inf')

            motif_id.append(motif.id)
            motif_name.append(motif.name)
            motif_family.append(motif.family)
            group_name.append(key)
            fold_change.append(log_fc)
            n_fg.append(bound_fg)
            N_fg.append(total_fg)
            n_bg.append(bound_bg)
            N_bg.append(total_bg)

          
    fold_change = np.array(fold_change)
    pval = np.zeros(len(fold_change))
    n_fg = np.array(n_fg)
    N_fg = np.array(N_fg)
    n_bg = np.array(n_bg)
    N_bg = np.array(N_bg)
    up_idx = fold_change >= 0
    down_idx = fold_change < 0
    if method == "binomial":
        pval[up_idx] = binom.sf(n_fg[up_idx] - 1, N_fg[up_idx], n_bg[up_idx] / N_bg[up_idx])
        pval[down_idx] = binom.cdf(n_fg[down_idx], N_fg[down_idx], n_bg[down_idx] / N_bg[down_idx])
    elif method == "hypergeometric":
        pval[up_idx] = hypergeom.sf(n_fg[up_idx] - 1, N_bg[up_idx], n_bg[up_idx], N_fg[up_idx])
        pval[down_idx] = hypergeom.cdf(n_fg[down_idx], N_bg[down_idx], n_bg[down_idx], N_fg[down_idx])
    else:
        raise NameError("'method' needs to be 'binomial' or 'hypergeometric'")
    pval = np.clip(pval, 1e-300, 1)

    result = dict(
        (key, {'id': [], 'name': [], 'family': [], 'log2(fold change)': [], 'p-value': []}) for key in regions.keys()
    )
    for i, key in enumerate(group_name):
        result[key]['id'].append(motif_id[i])
        result[key]['name'].append(motif_name[i])
        result[key]['family'].append(motif_family[i])
        result[key]['log2(fold change)'].append(fold_change[i])
        result[key]['p-value'].append(float(pval[i]))

    for key in result.keys():
        result[key]['adjusted p-value'] = _p_adjust_bh(result[key]['p-value'])
        result[key] = pl.DataFrame(result[key])
    return result
