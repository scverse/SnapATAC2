use numpy::{PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::{prelude::*, pybacked::PyBackedStr};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use statrs::distribution::{Binomial, DiscreteCDF};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use snapatac2_core::motif;

/** Python object representing DNA position weight matrix.

    The matrix is expected to be a PWM shaped `(length, 4)` in A/C/G/T order.

    Parameters
    ----------
    id
        Unique identifier of the motif.
    pwm
        Position weight matrix as a 2D numpy array of shape `(length, 4)`.

    See Also
    --------
    read_motifs
*/
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDNAMotif(pub motif::DNAMotif);

#[pymethods]
impl PyDNAMotif {
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

    /// The unique identifier of the motif.
    #[getter]
    fn id(&self) -> String {
        self.0.id.clone()
    }

    #[setter]
    fn set_id(&mut self, value: String) -> PyResult<()> {
        self.0.id = value;
        Ok(())
    }

    /// The name of the motif.
    #[getter]
    fn name(&self) -> Option<String> {
        self.0.name.clone()
    }

    #[setter]
    fn set_name(&mut self, value: String) -> PyResult<()> {
        self.0.name = Some(value);
        Ok(())
    }

    /// The family of the motif.
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
    fn info_content(&self) -> f64 {
        self.0.info_content()
    }

    /// Create a motif scanner with specified nucleotide background probabilities.
    /// 
    /// Parameters
    /// ----------
    /// a: float
    ///     Background probability of nucleotide A. Default is 0.25.
    /// c: float
    ///     Background probability of nucleotide C. Default is 0.25.
    /// g: float
    ///     Background probability of nucleotide G. Default is 0.25.
    /// t: float
    ///     Background probability of nucleotide T. Default is 0.25.
    /// 
    /// Returns
    /// -------
    /// PyDNAMotifScanner
    ///     A DNA motif scanner object.
    #[pyo3(signature = (a=0.25, c=0.25, g=0.25, t=0.25))]
    fn with_nucl_prob(&self, a: f64, c: f64, g: f64, t: f64) -> PyDNAMotifScanner {
        PyDNAMotifScanner(
            self.0
                .clone()
                .to_scanner(motif::BackgroundProb([a, c, g, t])),
        )
    }
}

/**
    Python object for scanning DNA sequences with a motif.
 */
#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDNAMotifScanner(pub motif::DNAMotifScanner);

#[pymethods]
impl PyDNAMotifScanner {
    /// The unique identifier of the motif.
    #[getter]
    fn id(&self) -> String {
        self.0.motif.id.clone()
    }

    /// The name of the motif.
    #[getter]
    fn name(&self) -> Option<String> {
        self.0.motif.name.clone()
    }

    /// Find motif occurrences in the given sequence above the specified p-value threshold.
    /// 
    /// Note it does not consider reverse complement matches.
    /// 
    /// Parameters
    /// ----------
    /// seq: str
    ///     DNA sequence to scan.
    /// pvalue: float
    ///     P-value threshold for reporting motif occurrences. Default is 1e-5.
    /// 
    /// Returns
    /// -------
    /// list[tuple[int, float]]
    ///     A list of tuples where each tuple contains the position of the motif occurrence
    ///     and the corresponding p-value.
    #[pyo3(signature = (seq, pvalue=1e-5))]
    fn find(&self, seq: &str, pvalue: f64) -> Vec<(usize, f64)> {
        self.0.find(seq.as_bytes(), pvalue).collect()
    }

    /// Check if the motif exists in the given sequence above the specified p-value threshold.
    /// 
    /// Parameters
    /// ----------
    /// seq: str
    ///     DNA sequence to scan.
    /// pvalue: float
    ///     P-value threshold for reporting motif occurrences. Default is 1e-5.
    /// rc: bool
    ///     Whether to consider reverse complement matches. Default is True.
    /// 
    /// Returns
    /// -------
    /// bool
    ///     True if the motif exists in the sequence, False otherwise.
    #[pyo3(signature = (seq, pvalue=1e-5, rc=true))]
    fn exist(&self, seq: &str, pvalue: f64, rc: bool) -> bool {
        self.0.find(seq.as_bytes(), pvalue).next().is_some()
            || (rc
                && self
                    .0
                    .find(rev_compl(seq).as_bytes(), pvalue)
                    .next()
                    .is_some())
    }

    /// Batch check if the motif exists in the given sequences above the specified p-value threshold.
    /// 
    /// This performs parallel computation over the input sequences.
    /// 
    /// Parameters
    /// ----------
    /// seqs: list[str]
    ///     List of DNA sequences to scan.
    /// pvalue: float
    ///     P-value threshold for reporting motif occurrences. Default is 1e-5.
    /// rc: bool
    ///     Whether to consider reverse complement matches. Default is True.
    /// 
    /// Returns
    /// -------
    /// list[bool]
    ///     A list of booleans indicating whether the motif exists in each sequence.
    #[pyo3(signature = (seqs, pvalue=1e-5, rc=true))]
    fn exists(&self, seqs: Vec<PyBackedStr>, pvalue: f64, rc: bool) -> Vec<bool> {
        seqs.into_par_iter()
            .map(|x| self.exist(x.as_ref(), pvalue, rc))
            .collect()
    }

    /// Create a motif test object using background sequences.
    /// 
    /// This create a PyDNAMotifTest object which can be later used to test motif enrichment.
    /// 
    /// Parameters
    /// ----------
    /// seqs: list[str]
    ///     List of background DNA sequences.
    /// pvalue: float
    ///     P-value threshold for reporting motif occurrences. Default is 1e-5.
    /// 
    /// Returns
    /// -------
    /// PyDNAMotifTest
    ///     A DNA motif test object.
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

/* Python object for testing motif enrichment in sequences.

   Examples
   --------
   ```python
   import snapatac2 as snap
   motifs = snap.read_motifs("motifs.meme")
   background_seqs = ['AAACGTTCC', 'TTGCCAATACC']  # list of background sequences
   target_seqs = ['ACGTAGCTAG', 'CGTACGTAGC']      # list of target sequences
   motif_test = motif[0].with_nucl_prob().with_background(background_seqs)
   log_fc, pval = motif_test.test(target_seqs)
   ```
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
    /// The unique identifier of the motif.
    #[getter]
    fn id(&self) -> String {
        self.scanner.id()
    }

    /// The name of the motif.
    #[getter]
    fn name(&self) -> Option<String> {
        self.scanner.name()
    }

    /// Test motif enrichment in the given sequences.
    /// 
    /// Parameters
    /// ----------
    /// seqs: list[str]
    ///     List of DNA sequences to test.
    /// 
    /// Returns
    /// -------
    /// tuple[float, float]
    ///     A tuple containing the log2 fold change and p-value of motif enrichment.
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

/// Read motifs from a MEME format file.
///
/// Parameters
/// ----------
/// filename: str | Path
///     Path to the MEME format file.
///
/// Returns
/// -------
/// list[PyDNAMotif]
///     List of `PyDNAMotif` objects.
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
