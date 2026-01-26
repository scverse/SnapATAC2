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

    #[getter]
    fn id(&self) -> String {
        self.0.id.clone()
    }

    #[setter]
    fn set_id(&mut self, value: String) -> PyResult<()> {
        self.0.id = value;
        Ok(())
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.0.name.clone()
    }

    #[setter]
    fn set_name(&mut self, value: String) -> PyResult<()> {
        self.0.name = Some(value);
        Ok(())
    }

    #[getter]
    fn family(&self) -> Option<String> {
        self.0.family.clone()
    }

    #[setter]
    fn set_family(&mut self, value: String) -> PyResult<()> {
        self.0.family = Some(value);
        Ok(())
    }

    fn info_content(&self) -> f64 {
        self.0.info_content()
    }

    #[pyo3(signature = (a=0.25, c=0.25, g=0.25, t=0.25))]
    fn with_nucl_prob(&self, a: f64, c: f64, g: f64, t: f64) -> PyDNAMotifScanner {
        PyDNAMotifScanner(
            self.0
                .clone()
                .to_scanner(motif::BackgroundProb([a, c, g, t])),
        )
    }
}

#[pyclass]
#[repr(transparent)]
#[derive(Clone)]
pub struct PyDNAMotifScanner(pub motif::DNAMotifScanner);

#[pymethods]
impl PyDNAMotifScanner {
    #[getter]
    fn id(&self) -> String {
        self.0.motif.id.clone()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.0.motif.name.clone()
    }

    #[pyo3(signature = (seq, pvalue=1e-5))]
    fn find(&self, seq: &str, pvalue: f64) -> Vec<(usize, f64)> {
        self.0.find(seq.as_bytes(), pvalue).collect()
    }

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

    #[pyo3(signature = (seqs, pvalue=1e-5, rc=true))]
    fn exists(&self, seqs: Vec<PyBackedStr>, pvalue: f64, rc: bool) -> Vec<bool> {
        seqs.into_par_iter()
            .map(|x| self.exist(x.as_ref(), pvalue, rc))
            .collect()
    }

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

#[pyclass]
pub struct PyDNAMotifTest {
    scanner: PyDNAMotifScanner,
    pvalue: f64,
    occurrence_background: usize,
    total_background: usize,
}

#[pymethods]
impl PyDNAMotifTest {
    #[getter]
    fn id(&self) -> String {
        self.scanner.id()
    }

    #[getter]
    fn name(&self) -> Option<String> {
        self.scanner.name()
    }

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
