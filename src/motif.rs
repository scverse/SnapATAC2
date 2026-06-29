use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{prelude::*, pybacked::PyBackedStr};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::distribution::{Binomial, DiscreteCDF};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use snapatac2_core::motif;

/** Represent a DNA motif as a position weight matrix.

    Use this class when a Python workflow needs to scan DNA sequences or test
    motif enrichment from an explicit PWM. The PWM must have one row per motif
    position and exactly four columns in A/C/G/T order. Rows are normalized to
    probabilities during construction.

    Anti-Patterns
    -------------
    - Do NOT pass columns in alphabetical ambiguity-code order or any order other
      than A/C/G/T.
    - Do NOT pass log-odds scores to `pwm`; pass nonnegative weights or
      probabilities that can be normalized row by row.
    - Do NOT use this object directly for scanning. Create a scanner with
      `with_nucl_prob` first.


    Parameters
    ----------
    id
        Unique identifier of the motif.
    pwm
        Position weight matrix as a 2D numpy array of shape `(length, 4)`.

    See Also
    --------
    read_motifs

    Examples
    --------
    >>> import numpy as np
    >>> import snapatac2 as snap
    >>> motif = snap.PyDNAMotif(
    ...     "example",
    ...     np.array([
    ...         [0.9, 0.05, 0.03, 0.02],
    ...         [0.1, 0.8, 0.05, 0.05],
    ...     ]),
    ... )
    >>> scanner = motif.with_nucl_prob()
    >>> scanner.exist("ACGTACGT", pvalue=1e-5)
 */
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDNAMotif(pub motif::DNAMotif);

#[pymethods]
impl PyDNAMotif {
    /// Create a DNA motif from an identifier and an A/C/G/T PWM.
    ///
    /// Use this constructor when the motif matrix is already available in
    /// Python. Each row is normalized internally, so rows may contain weights or
    /// probabilities, but every row must contain four numeric values in A/C/G/T
    /// order.
    ///
    /// Anti-Patterns
    /// -------------
    /// - Do NOT pass a matrix with fewer or more than four columns.
    /// - Do NOT pass rows whose sum is zero.
    ///
    /// Parameters
    /// ----------
    /// id : str
    ///     Motif identifier.
    /// pwm : numpy.ndarray
    ///     Position weight matrix with shape `(length, 4)` in A/C/G/T order.
    ///
    /// Returns
    /// -------
    /// PyDNAMotif
    ///     A motif object that can create scanners and report information
    ///     content.
    #[new]
    fn new<'py>(id: &str, pwm: &Bound<'py, PyArray2<f64>>) -> Self {
        assert!(pwm.shape()[1] == 4, "PWM must have 4 columns for A/C/G/T");

        let probability: Vec<_> = pwm
            .readonly()
            .as_array()
            .rows()
            .into_iter()
            .map(|row| {
                let sum = row.sum();
                [row[0] / sum, row[1] / sum, row[2] / sum, row[3] / sum]
            })
            .collect();
        let motif = motif::DNAMotif {
            id: id.to_string(),
            name: None,
            family: None,
            probability,
        };
        PyDNAMotif(motif)
    }

    /// Return the unique motif identifier.
    #[getter]
    fn id(&self) -> String {
        self.0.id.clone()
    }

    #[setter]
    fn set_id(&mut self, value: String) -> PyResult<()> {
        self.0.id = value;
        Ok(())
    }

    /// Return the optional display name of the motif.
    #[getter]
    fn name(&self) -> Option<String> {
        self.0.name.clone()
    }

    #[setter]
    fn set_name(&mut self, value: String) -> PyResult<()> {
        self.0.name = Some(value);
        Ok(())
    }

    /// Return the optional motif family label.
    #[getter]
    fn family(&self) -> Option<String> {
        self.0.family.clone()
    }

    #[setter]
    fn set_family(&mut self, value: String) -> PyResult<()> {
        self.0.family = Some(value);
        Ok(())
    }

    /// Return the information content of the motif.
    ///
    /// Use this value to compare how specific multiple motifs are. Higher
    /// values indicate a more informative PWM.
    fn info_content(&self) -> f64 {
        self.0.info_content()
    }

    /// Create a motif scanner with specified nucleotide background probabilities.
    ///
    /// Use this method before scanning sequences or testing motif enrichment.
    /// The background probabilities are used to convert the PWM into a scanner.
    ///
    /// Anti-Patterns
    /// -------------
    /// - Do NOT pass probabilities in an order other than A/C/G/T.
    /// - Do NOT pass negative probabilities.
    ///
    /// Parameters
    /// ----------
    /// a : float
    ///     Background probability of nucleotide A. Default is 0.25.
    /// c : float
    ///     Background probability of nucleotide C. Default is 0.25.
    /// g : float
    ///     Background probability of nucleotide G. Default is 0.25.
    /// t : float
    ///     Background probability of nucleotide T. Default is 0.25.
    ///
    /// Returns
    /// -------
    /// PyDNAMotifScanner
    ///     A DNA motif scanner object.
    ///
    /// Examples
    /// --------
    /// >>> scanner = motif.with_nucl_prob(a=0.25, c=0.25, g=0.25, t=0.25)
    #[pyo3(signature = (a=0.25, c=0.25, g=0.25, t=0.25))]
    fn with_nucl_prob(&self, a: f64, c: f64, g: f64, t: f64) -> PyDNAMotifScanner {
        PyDNAMotifScanner(
            self.0
                .clone()
                .to_scanner(motif::BackgroundProb([a, c, g, t])),
        )
    }
}

/** Scan DNA sequences with a motif-specific scanner.

    Use this object to find motif hits, test whether a motif exists in one or
    more sequences, or construct a motif enrichment test from background
    sequences. Create it from `PyDNAMotif.with_nucl_prob`.

    Anti-Patterns
    -------------
    - Do NOT instantiate this class directly; create it from a `PyDNAMotif`.
    - Do NOT assume `find` checks reverse complements. Use `exist(..., rc=True)`
      or `exists(..., rc=True)` when reverse-complement matching is required.

    See Also
    --------
    PyDNAMotif.with_nucl_prob
 */
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDNAMotifScanner(pub motif::DNAMotifScanner);

#[pymethods]
impl PyDNAMotifScanner {
    /// Return the unique identifier of the scanned motif.
    #[getter]
    fn id(&self) -> String {
        self.0.motif.id.clone()
    }

    /// Return the optional display name of the scanned motif.
    #[getter]
    fn name(&self) -> Option<String> {
        self.0.motif.name.clone()
    }

    /// Find motif occurrences in the given sequence above the specified p-value threshold.
    ///
    /// Use this method when the positions and scores of forward-strand motif
    /// hits are needed. Set `report_pvalue=True` to also report log10 p-values
    /// estimated from the scanner's score CDF. This method does not scan the
    /// reverse complement.
    ///
    /// Anti-Patterns
    /// -------------
    /// - Do NOT use `find` when reverse-complement hits should be considered;
    ///   use `exist` or scan the reverse complement explicitly.
    /// - Do NOT pass RNA sequences; use DNA letters A/C/G/T.
    ///
    /// Parameters
    /// ----------
    /// seq : str
    ///     DNA sequence to scan.
    /// pvalue : float
    ///     P-value threshold for reporting motif occurrences. Default is 1e-5.
    /// report_pvalue : bool
    ///     Whether to report per-hit log10 p-values. Default is False.
    ///
    /// Returns
    /// -------
    /// list[tuple[int, float, float | None]]
    ///     A list of tuples where each tuple contains the position of the motif
    ///     occurrence, the natural-log likelihood-ratio score, and optionally
    ///     the log10 p-value.
    ///
    /// Examples
    /// --------
    /// >>> hits = scanner.find("ACGTACGT", pvalue=1e-5, report_pvalue=True)
    #[pyo3(signature = (seq, pvalue=1e-5, report_pvalue=false))]
    fn find(
        &self,
        seq: &str,
        pvalue: f64,
        report_pvalue: bool,
    ) -> Vec<(usize, f64, Option<f64>)> {
        self.0
            .find(seq.as_bytes(), pvalue, report_pvalue)
            .map(|x| (x.position, x.score, x.log10_p_value))
            .collect()
    }

    /// Check if the motif exists in the given sequence above the specified p-value threshold.
    ///
    /// Use this method when only a boolean motif-presence result is needed.
    /// Set `rc=True` to test both the sequence and its reverse complement.
    ///
    /// Anti-Patterns
    /// -------------
    /// - Do NOT use this method when hit positions are required; use `find` for
    ///   forward-strand hit positions.
    ///
    /// Parameters
    /// ----------
    /// seq : str
    ///     DNA sequence to scan.
    /// pvalue : float
    ///     P-value threshold for reporting motif occurrences. Default is 1e-5.
    /// rc : bool
    ///     Whether to consider reverse complement matches. Default is True.
    ///
    /// Returns
    /// -------
    /// bool
    ///     True if the motif exists in the sequence, False otherwise.
    ///
    /// Examples
    /// --------
    /// >>> scanner.exist("ACGTACGT", pvalue=1e-5, rc=True)
    #[pyo3(signature = (seq, pvalue=1e-5, rc=true))]
    fn exist(&self, seq: &str, pvalue: f64, rc: bool) -> bool {
        self.0.find(seq.as_bytes(), pvalue, false).next().is_some()
            || (rc
                && self
                    .0
                    .find(rev_compl(seq).as_bytes(), pvalue, false)
                    .next()
                    .is_some())
    }

    /// Batch check if the motif exists in the given sequences above the specified p-value threshold.
    ///
    /// Use this method to scan many sequences with parallel computation. The
    /// returned list preserves the input sequence order.
    ///
    /// Anti-Patterns
    /// -------------
    /// - Do NOT use this method when per-hit positions are required; it only
    ///   returns booleans.
    ///
    /// Parameters
    /// ----------
    /// seqs : list[str]
    ///     List of DNA sequences to scan.
    /// pvalue : float
    ///     P-value threshold for reporting motif occurrences. Default is 1e-5.
    /// rc : bool
    ///     Whether to consider reverse complement matches. Default is True.
    ///
    /// Returns
    /// -------
    /// list[bool]
    ///     A list of booleans indicating whether the motif exists in each sequence.
    ///
    /// Examples
    /// --------
    /// >>> scanner.exists(["ACGTACGT", "TTTTAAAA"], pvalue=1e-5, rc=True)
    #[pyo3(signature = (seqs, pvalue=1e-5, rc=true))]
    fn exists(&self, seqs: Vec<PyBackedStr>, pvalue: f64, rc: bool) -> Vec<bool> {
        seqs.into_par_iter()
            .map(|x| self.exist(x.as_ref(), pvalue, rc))
            .collect()
    }

    /// Create a motif test object using background sequences.
    ///
    /// Use this method to define the background motif occurrence rate before
    /// testing enrichment in target sequences. The resulting `PyDNAMotifTest`
    /// stores the number of background sequences containing the motif.
    ///
    /// Anti-Patterns
    /// -------------
    /// - Do NOT build the background from the same sequences later used as the
    ///   target set.
    /// - Do NOT use a background set that is much smaller or compositionally
    ///   unrelated to the target set.
    ///
    /// Parameters
    /// ----------
    /// seqs : list[str]
    ///     List of background DNA sequences.
    /// pvalue : float
    ///     P-value threshold for reporting motif occurrences. Default is 1e-5.
    ///
    /// Returns
    /// -------
    /// PyDNAMotifTest
    ///     A DNA motif test object.
    ///
    /// Examples
    /// --------
    /// >>> background = ["ACGTACGT", "TTTTAAAA"]
    /// >>> motif_test = scanner.with_background(background, pvalue=1e-5)
    #[pyo3(signature = (seqs, pvalue=1e-5))]
    fn with_background(&self, seqs: Vec<PyBackedStr>, pvalue: f64) -> PyDNAMotifTest {
        let n = seqs.len();
        PyDNAMotifTest {
            scanner: self.clone(),
            pvalue,
            occurrence_background: seqs
                .into_par_iter()
                .filter(|x| self.exist(x, pvalue, true))
                .count(),
            total_background: n,
        }
    }
}

fn rev_compl(dna: &str) -> String {
    dna.chars()
        .rev()
        .map(|x| match x {
            'A' | 'a' => 'T',
            'C' | 'c' => 'G',
            'G' | 'g' => 'C',
            'T' | 't' => 'A',
            c => c,
        })
        .collect()
}

/* Test motif enrichment in target sequences against a background set.

   Use this object after creating a scanner and calling `with_background`. The
   test compares motif occurrence in target sequences against occurrence in the
   stored background sequences and returns `(log2_fold_change, p_value)`.

   Anti-Patterns
   -------------
   - Do NOT instantiate this class directly; create it with
     `PyDNAMotifScanner.with_background`.
   - Do NOT interpret the p-value without checking the direction and magnitude
     of `log2_fold_change`.

   Examples
   --------
   >>> import snapatac2 as snap
   >>> motifs = snap.read_motifs("motifs.meme")
   >>> background_seqs = ["AAACGTTCC", "TTGCCAATACC"]
   >>> target_seqs = ["ACGTAGCTAG", "CGTACGTAGC"]
   >>> motif_test = motifs[0].with_nucl_prob().with_background(background_seqs)
   >>> log_fc, pval = motif_test.test(target_seqs)
 */
#[pyclass]
pub struct PyDNAMotifTest {
    scanner: PyDNAMotifScanner,
    pvalue: f64,
    occurrence_background: usize,
    total_background: usize,
}

#[pymethods]
impl PyDNAMotifTest {
    /// Return the unique identifier of the motif being tested.
    #[getter]
    fn id(&self) -> String {
        self.scanner.id()
    }

    /// Return the optional display name of the motif being tested.
    #[getter]
    fn name(&self) -> Option<String> {
        self.scanner.name()
    }

    /// Test motif enrichment in the given sequences.
    ///
    /// Use this method to compare target sequence motif occurrence against the
    /// background occurrence rate stored in this test object.
    ///
    /// Anti-Patterns
    /// -------------
    /// - Do NOT pass an empty target list.
    /// - Do NOT reuse a background built with a different p-value threshold when
    ///   comparing results across tests.
    ///
    /// Parameters
    /// ----------
    /// seqs : list[str]
    ///     List of DNA sequences to test.
    ///
    /// Returns
    /// -------
    /// tuple[float, float]
    ///     A tuple containing the log2 fold change and p-value of motif enrichment.
    ///
    /// Examples
    /// --------
    /// >>> log_fc, pval = motif_test.test(["ACGTAGCTAG", "CGTACGTAGC"])
    #[pyo3(signature = (seqs))]
    fn test(&self, seqs: Vec<PyBackedStr>) -> (f64, f64) {
        let n = seqs.len().try_into().unwrap();
        let occurrence: u64 = seqs
            .into_par_iter()
            .filter(|x| self.scanner.exist(x, self.pvalue, true))
            .count()
            .try_into()
            .unwrap();
        let p = self.occurrence_background as f64 / self.total_background as f64;
        let log_fc = ((occurrence as f64 / n as f64) / p).log2();
        let bion = Binomial::new(p, n).unwrap();
        let pval = if log_fc >= 0.0 {
            bion.sf(occurrence)
        } else {
            bion.cdf(occurrence)
        };
        (log_fc, pval)
    }
}

/// Read DNA motifs from a MEME format file.
///
/// Use this function to load motif collections before motif scanning or motif
/// enrichment analysis. Each MEME motif is returned as a `PyDNAMotif`.
///
/// Anti-Patterns
/// -------------
/// - Do NOT pass non-MEME files; parsing expects MEME motif syntax.
/// - Do NOT assume motif names are unique unless the source collection enforces
///   uniqueness.
///
/// Parameters
/// ----------
/// filename : str | Path
///     Path to the MEME format file.
///
/// Returns
/// -------
/// list[PyDNAMotif]
///     List of `PyDNAMotif` objects.
///
/// Examples
/// --------
/// >>> import snapatac2 as snap
/// >>> motifs = snap.read_motifs("motifs.meme")
/// >>> scanner = motifs[0].with_nucl_prob()
/// >>> scanner.exist("ACGTACGT", pvalue=1e-5)
#[pyfunction]
pub(crate) fn read_motifs(filename: &str) -> Vec<PyDNAMotif> {
    let path = Path::new(filename);
    let mut file = match File::open(&path) {
        Err(why) => panic!("couldn't open file: {}", why),
        Ok(file) => file,
    };
    let mut s = String::new();
    file.read_to_string(&mut s).unwrap();
    motif::parse_meme(&s)
        .into_iter()
        .map(|x| PyDNAMotif(x))
        .collect()
}
