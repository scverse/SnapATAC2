use nalgebra_sparse::CsrMatrix;
use itertools::Itertools;
use rayon::iter::{ParallelBridge, ParallelIterator};

pub trait InverseDocumentFrequency {
    /// Compute inverse document frequency (IDF) for a given sparse matrix.
    /// The input matrix is expected to be in CSR format,
    /// where each column represents a document and each row represents a term.
    /// The IDF is computed as `log(N / df)`, where `N` is the total number of documents
    /// and `df` is the document frequency of the term.
    /// If a term appears in all documents, its IDF is set to log(N / (N - 1)).
    /// If a term does not appear in any document, its IDF is set to log(N / 1) to 
    /// avoid division by zero.
    fn idf(self) -> Vec<f64>;
}

/*
impl InverseDocumentFrequency for &CsrMatrix<f64> {
    fn idf(self) -> Vec<f64> {
        let mut idf = vec![0.0; self.ncols()];
        // Compute document frequency for each term
        self.col_indices().iter().for_each(|i| idf[*i] += 1.0);
        let n = self.nrows() as f64;
        if idf.iter().all_equal() {
            vec![1.0; idf.len()]
        } else {
            idf.iter_mut().for_each(|x| {
                if *x == 0.0 {
                    *x = 1.0;
                } else if *x == n {
                    *x = n - 1.0;
                }
                *x = (n / *x).ln()
            });
            idf
        }
    }
}
    */

impl<I: Iterator<Item = CsrMatrix<f64>>> InverseDocumentFrequency for I {
    fn idf(self) -> Vec<f64> {
        let mut iter = self.peekable();
        let mut idf = vec![0.0; iter.peek().unwrap().ncols()];
        let mut n = 0.0;
        iter.for_each(|mat| {
            mat.col_indices().iter().for_each(|i| idf[*i] += 1.0);
            n += mat.nrows() as f64;
        });
        if idf.iter().all_equal() {
            vec![1.0; idf.len()]
        } else {
            idf.iter_mut().for_each(|x| {
                if *x == 0.0 {
                    *x = 1.0;
                } else if *x == n {
                    *x = n - 1.0;
                }
                *x = (n / *x).ln()
            });
            idf
        }
    }
}


// idf_from_chunks that parallelizes the counting step
pub fn idf_from_chunks_parallel<I>(input: I) -> Vec<f64>
where
    I: IntoIterator<Item = CsrMatrix<f64>>,
{
    let mut idf: Option<Vec<f64>> = None;
    let mut n = 0.0;
    for mat in input {
        let ncols = mat.ncols();
        if idf.is_none() {
            idf = Some(vec![0.0; ncols]);
        }
        let local: Vec<f64> = mat
            .row_iter()
            .par_bridge()
            .map(|row| {
                let mut local = vec![0.0; ncols];
                for i in row.col_indices() {
                    local[*i] += 1.0;
                }
                local
            })
            .reduce(|| vec![0.0; ncols], |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    *x += y;
                }
                a
            });
        if let Some(ref mut idf_vec) = idf {
            for (x, y) in idf_vec.iter_mut().zip(local) {
                *x += y;
            }
        }
        n += mat.nrows() as f64;
    }
    let idf = idf.unwrap_or_default();
    if idf.iter().all_equal() {
        vec![1.0; idf.len()]
    } else {
        idf.into_iter()
            .map(|mut x| {
                if x == 0.0 {
                    x = 1.0;
                } else if x == n {
                    x = n - 1.0;
                }
                (n / x).ln()
            })
            .collect()
    }
}