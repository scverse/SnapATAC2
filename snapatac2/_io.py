from __future__ import annotations

from pathlib import Path
from anndata import AnnData
import snapatac2._snapatac2 as internal
from scipy.sparse import csr_matrix

def read_10x_mtx(
    path: Path,
    file: Path | None = None,
    prefix: str | None = None,
) -> AnnData:
    """Read a 10x Genomics MTX directory into an AnnData object.

    Use this function to load a directory containing a matrix file, feature file,
    and barcode file in the 10x MTX layout. Pass `file` to create a backed
    AnnData object on disk instead of returning a fully in-memory object.

    Anti-Patterns
    -------------
    - Do NOT pass the path to `matrix.mtx` directly; pass the directory that
      contains the matrix, feature, and barcode files.
    - Do NOT include multiple matching matrix, feature, or barcode files with the
      same prefix in the directory; exactly one of each file type must match.

    Parameters
    ----------
    path : pathlib.Path
        Directory containing the 10x MTX files. The directory must contain one
        matching file from each group:

        1. Matrix: "matrix.mtx" or "matrix.mtx.gz".
        2. Features: "genes.tsv", "genes.tsv.gz", "features.tsv", or
           "features.tsv.gz".
        3. Barcodes: "barcodes.tsv" or "barcodes.tsv.gz".
    file : pathlib.Path or None, default: None
        Output h5ad filename for a backed AnnData object. If None, return an
        in-memory AnnData object.
    prefix : str or None, default: None
        Optional filename prefix before the matrix, feature, and barcode names.
        For files named `patientA_matrix.mtx`, `patientA_genes.tsv`, and
        `patientA_barcodes.tsv`, pass `prefix="patientA_"`.

    Returns
    -------
    AnnData
        AnnData object with observations from the barcode file, variables from
        the feature file, and `.X` containing the transposed sparse count matrix.

    Examples
    --------
    >>> from pathlib import Path
    >>> import snapatac2 as snap
    >>> adata = snap.read_10x_mtx(Path("filtered_feature_bc_matrix"))
    >>> adata.n_obs >= 0
    True
    """
    import pandas as pd

    def get_files(prefix, names):
        return list(filter(
            lambda x: x.is_file(),
            map(lambda x: Path(path + "/" + prefix + x), names)
        ))

    prefix = "" if prefix is None else prefix

    matrix_files = get_files(prefix, ["matrix.mtx", "matrix.mtx.gz"])
    n = len(matrix_files)
    if n == 1:
        mat = csr_matrix(internal.read_mtx(str(matrix_files[0])).X[:].T)
        adata = AnnData(X=mat) if file is None else internal.AnnData(X=mat, filename=file)
    else:
        raise ValueError("Expecting a single 'matrix.mtx' or 'matrix.mtx.gz' file, but found {}.".format(n))

    feature_files = get_files(
        prefix,
        ["genes.tsv", "genes.tsv.gz", "features.tsv", "features.tsv.gz"]
    )
    n = len(feature_files)
    if n == 1:
        df = pd.read_csv(str(feature_files[0]), sep='\t', header=None, index_col=0)
        df.index.name = "index"
        adata.var_names = df.index
        adata.var = df
    else:
        raise ValueError("Expecting a single feature file, but found {}.".format(n))

    barcode_files = get_files(prefix, ["barcodes.tsv", "barcodes.tsv.gz"])
    n = len(barcode_files)
    if n == 1:
        df = pd.read_csv(str(barcode_files[0]), sep='\t', header=None, index_col=0)
        df.index.name = "index"
        adata.obs_names = df.index
        adata.obs = df
    else:
        raise ValueError("Expecting a single barcode file, but found {}.".format(n))

    return adata
