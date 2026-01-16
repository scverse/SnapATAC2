use crate::utils::AnnDataLike;
use snapatac2_core::utils::PrefetchIterator;

use anndata::{
    data::{
        array::utils::to_csr_data, ArrayConvert, DynCsrMatrix, SelectInfoElem,
        SelectInfoElemBounds, Stackable,
    },
    AnnDataOp, ArrayData, ArrayElemOp, Backend, Selectable,
};
use anndata_hdf5::H5;
use anyhow::Result;
use indicatif::{ProgressIterator, ProgressStyle};
use itertools::Itertools;
use log::info;
use nalgebra::{DMatrix, DVector};
use nalgebra_sparse::CsrMatrix;
use ndarray::{Array1, Array2, Axis};
use numpy::{array::PyArrayMethods, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyanndata::data::PyArrayData;
use pyo3::{ffi::c_str, prelude::*};
use rand::SeedableRng;
use rayon::{
    iter::IntoParallelIterator,
    prelude::{ParallelBridge, ParallelIterator},
};
use std::{collections::HashSet, ops::Deref};

#[pyfunction]
#[pyo3(signature = (anndata, selected_features, n_components, random_state, feature_weights=None))]
pub(crate) fn spectral_embedding<'py>(
    py: Python<'py>,
    anndata: AnnDataLike,
    selected_features: &Bound<'_, PyAny>,
    n_components: usize,
    random_state: i64,
    feature_weights: Option<Vec<f64>>,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    macro_rules! run {
        ($data:expr) => {{
            let slice = pyanndata::data::to_select_elem(selected_features, $data.n_vars())?;
            let mut mat: CsrMatrix<f64> = $data
                .x()
                .slice_axis::<DynCsrMatrix, _>(1, slice)?
                .unwrap()
                .try_convert()?;
            if let Some(weights) = feature_weights {
                normalize(&mut mat, &weights);
            } else {
                let weights = idf(&mat);
                normalize(&mut mat, &weights);
            }

            let (v, u, _) = spectral_mf(mat, n_components, random_state)?;
            anyhow::Ok((v, u))
        }};
    }
    let (evals, evecs) = crate::with_anndata!(&anndata, run)?;

    Ok((
        PyArray1::from_owned_array(py, evals),
        PyArray2::from_owned_array(py, evecs),
    ))
}

/// Matrix-free spectral embedding.
/// The input is assumed to be a csr matrix with rows normalized to unit L2 norm.
fn spectral_mf(
    mut input: CsrMatrix<f64>,
    n_components: usize,
    random_state: i64,
) -> Result<(Array1<f64>, Array2<f64>, Array1<f64>)> {
    let mut col_sum = vec![0.0; input.ncols()];
    input
        .col_indices()
        .iter()
        .zip(input.values().iter())
        .for_each(|(i, x)| col_sum[*i] += x);
    let mut degree_inv: DVector<_> = &input * &DVector::from(col_sum);
    degree_inv.iter_mut().for_each(|x| *x = (*x - 1.0).recip());

    // row-wise normalization using degrees.
    input
        .row_iter_mut()
        .zip(degree_inv.iter())
        .for_each(|(mut row, d)| row.values_mut().iter_mut().for_each(|x| *x *= d.sqrt()));

    // Compute eigenvalues and eigenvectors
    let (v, u) = Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(
                "def eigen(X, D, k, seed):
                from scipy.sparse.linalg import LinearOperator, eigsh
                import numpy
                numpy.random.seed(seed)
                def f(v):
                    return X @ (v.T @ X).T - D * v

                n = X.shape[0]
                A = LinearOperator((n, n), matvec=f, dtype=numpy.float64)
                evals, evecs = eigsh(A, k=k, v0=numpy.random.rand(n))
                ix = evals.argsort()[::-1]
                evals = evals[ix]
                evecs = evecs[:, ix]
                return (evals, evecs)"
            ),
            c_str!(""),
            c_str!(""),
        )?
        .getattr("eigen")?
        .into();
        let args = (
            PyArrayData::from(ArrayData::from(input)),
            PyArray1::from_iter(py, degree_inv.into_iter().copied()),
            n_components,
            random_state,
        );
        let result = fun.call1(py, args)?;
        let (evals, evecs): (PyReadonlyArray1<'_, f64>, PyReadonlyArray2<'_, f64>) =
            result.extract(py)?;

        anyhow::Ok((evals.to_owned_array(), evecs.to_owned_array()))
    })?;
    degree_inv.iter_mut().for_each(|x| *x = x.recip());
    Ok((v, u, degree_inv.into_iter().copied().collect()))
}

#[pyfunction]
pub(crate) fn spectral_embedding_nystrom<'py>(
    py: Python<'py>,
    anndata: AnnDataLike,
    selected_features: &Bound<'_, PyAny>,
    n_components: usize,
    sample_size: usize,
    weighted_by_degree: bool,
    chunk_size: usize,
    feature_weights: Option<Vec<f64>>,
    num_threads: usize,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    macro_rules! run {
        ($data:expr) => {{
            // Get feature indices
            let selected_features =
                pyanndata::data::to_select_elem(selected_features, $data.n_vars())?;

            // Get sample indices
            let n_obs = $data.n_obs();
            let mut rng = rand::rngs::StdRng::seed_from_u64(2023);
            let idx = if weighted_by_degree {
                todo!()
                /*
                let weights = compute_probs(&compute_degrees($data, &selected_features, &weights));
                rand::seq::index::sample_weighted(&mut rng, n_obs, |i| weights[i], sample_size)?
                    .into_vec()
                */
            } else {
                rand::seq::index::sample(&mut rng, n_obs, sample_size).into_vec()
            };
            let selected_samples: HashSet<usize> = idx.into_iter().collect();

            let style = ProgressStyle::with_template(
                "[{elapsed}] {bar:40.cyan/blue} {pos:>7}/{len:7} (eta: {eta})",
            )
            .unwrap();
            info!("Compute IDF and extract submatrix...");
            // Extract submatrix and compute idf
            let feat = selected_features.clone();
            let (seed_mat, idf): (CsrMatrix<f64>, Vec<f64>) = get_submatrix_and_idf(
                PrefetchIterator::new(
                    $data
                        .x()
                        .iter::<DynCsrMatrix>(chunk_size * num_threads)
                        .map(move |(x, i, j)| {
                            let mat: CsrMatrix<f64> = x.try_convert().unwrap();
                            (mat.select_axis(1, &feat), i, j)
                        })
                        .progress_with_style(style.clone())
                        .with_finish(indicatif::ProgressFinish::Abandon),
                    1,
                ),
                selected_samples,
            )?;
            let weights = if let Some(weights) = feature_weights {
                weights
            } else {
                idf
            };

            let nystrom = Nystrom::new(seed_mat, n_components, weights)?;
            info!("Apply Nystrom to out-of-sample data...");
            let data_iter = PrefetchIterator::new(
                $data
                    .x()
                    .iter(chunk_size * num_threads)
                    .map(move |x: (DynCsrMatrix, _, _)| {
                        let mat: CsrMatrix<f64> = x.0.try_convert().unwrap();
                        mat.select_axis(1, &selected_features)
                    })
                    .progress_with_style(style)
                    .with_finish(indicatif::ProgressFinish::Abandon),
                1,
            );
            let results: Vec<f64> = data_iter
                .flat_map(|mat| {
                    nystrom
                        .par_transform(mat, n_obs, num_threads)
                        .into_iter()
                        .flat_map(|x| x.row_iter().flatten().copied().collect::<Vec<_>>())
                })
                .collect();

            anyhow::Ok((
                nystrom.evals,
                Array2::from_shape_vec((n_obs, n_components), results).unwrap(),
            ))
        }};
    }

    let (evals, evecs) = crate::with_anndata!(&anndata, run)?;
    Ok((
        PyArray1::from_owned_array(py, evals),
        PyArray2::from_owned_array(py, evecs),
    ))
}

struct Nystrom {
    evals: Array1<f64>,
    qmat: DMatrix<f64>,
    feature_weights: Vec<f64>,
}

impl Nystrom {
    fn new(
        mut seed_mat: CsrMatrix<f64>,
        n_components: usize,
        feature_weights: Vec<f64>,
    ) -> Result<Self> {
        // feature weighting and L2 norm normalization.
        normalize(&mut seed_mat, &feature_weights);

        info!("Compute embeddings for {} landmarks...", seed_mat.nrows());
        let (evals, mut evecs, degrees) = spectral_mf(seed_mat.clone(), n_components, 0)?;

        // normalize the eigenvectors by degrees.
        evecs
            .axis_iter_mut(Axis(0))
            .zip(degrees.iter())
            .for_each(|(mut row, d)| row *= d.sqrt().recip());

        // normalize the eigenvectors by eigenvalues.
        evecs
            .axis_iter_mut(Axis(1))
            .zip(evals.iter())
            .for_each(|(mut col, v)| col *= v.recip());

        let evecs_norm: DMatrix<_> =
            DMatrix::from_row_iterator(evecs.nrows(), evecs.ncols(), evecs);
        let qmat = &seed_mat.transpose() * evecs_norm;

        Ok(Self {
            evals,
            qmat,
            feature_weights,
        })
    }

    fn par_transform(&self, mut mat: CsrMatrix<f64>, scale_factor: usize, num_threads: usize) -> Vec<DMatrix<f64>> {
        let scale_factor = scale_factor as f64 / mat.nrows() as f64;
        // feature weighting and L2 norm normalization.
        normalize(&mut mat, &self.feature_weights);
        let nrows = mat.nrows();
        let chunk_size = (nrows + num_threads - 1) / num_threads;
        (0..num_threads)
            .into_par_iter()
            .map(|i| {
                let start = (i * chunk_size).min(nrows);
                let end = ((i + 1) * chunk_size).min(nrows);
                let mut qmat = spmm_dense(start, end, &mat, &self.qmat);
                let mut q_sum = qmat.row_sum_tr();
                q_sum.iter_mut().enumerate().for_each(|(i, x)| {
                    *x *= self.evals[i] * scale_factor;
                });
                let mut d = &qmat * q_sum;

                // make sure d > 0
                let mut d_min = f64::MAX;
                d.iter().for_each(|x| {
                    if *x > 0.0 && *x < d_min {
                        d_min = *x
                    }
                });
                d_min = d_min.sqrt();
                d.iter_mut().for_each(|x| {
                    if *x <= 0.0 {
                        *x = d_min
                    } else {
                        *x = x.sqrt();
                    }
                });

                qmat.row_iter_mut().enumerate().for_each(|(i, mut row)| {
                    row /= d[i];
                });
                qmat
            })
            .collect()
    }

    fn transform(&self, mut mat: CsrMatrix<f64>, scale_factor: usize) -> DMatrix<f64> {
        let scale_factor = scale_factor as f64 / mat.nrows() as f64;
        // feature weighting and L2 norm normalization.
        normalize(&mut mat, &self.feature_weights);
        let mut qmat = &mat * &self.qmat;
        let mut q_sum = qmat.row_sum_tr();
        q_sum.iter_mut().enumerate().for_each(|(i, x)| {
            *x *= self.evals[i] * scale_factor;
        });
        let mut d = &qmat * q_sum;

        // make sure d > 0
        let mut d_min = f64::MAX;
        d.iter().for_each(|x| {
            if *x > 0.0 && *x < d_min {
                d_min = *x
            }
        });
        d_min = d_min.sqrt();
        d.iter_mut().for_each(|x| {
            if *x <= 0.0 {
                *x = d_min
            } else {
                *x = x.sqrt();
            }
        });

        qmat.row_iter_mut().enumerate().for_each(|(i, mut row)| {
            row /= d[i];
        });
        qmat
    }
}

fn idf(input: &CsrMatrix<f64>) -> Vec<f64> {
    let mut idf = vec![0.0; input.ncols()];
    input.col_indices().iter().for_each(|i| idf[*i] += 1.0);
    let n = input.nrows() as f64;
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

/// Extract submatrix given row and column indices from an iterator of matrices.
/// Also compute idf from the matrices.
fn get_submatrix_and_idf<I>(
    mat_iter: I,
    row_indices: HashSet<usize>,
) -> Result<(CsrMatrix<f64>, Vec<f64>)>
where
    I: IntoIterator<Item = (CsrMatrix<f64>, usize, usize)>,
{
    let mut idf: Option<Vec<f64>> = None;
    let mut n = 0.0;
    let mut update_idf = |mat: &CsrMatrix<f64>| {
        let ncols = mat.ncols();
        if idf.is_none() {
            idf = Some(vec![0.0; ncols]);
        }

        n += mat.nrows() as f64;
        mat.col_indices()
            .iter()
            .for_each(|i| idf.as_mut().unwrap()[*i] += 1.0);
    };

    let mat = {
        let results = mat_iter.into_iter().map(|(m, i, j)| {
            update_idf(&m);
            let row_idx = (i..j)
                .filter(|x| row_indices.contains(x))
                .map(|x| x - i)
                .collect::<Vec<_>>();
            m.select_axis(0, SelectInfoElem::from(row_idx))
        });
        Stackable::vstack(results)?
    };

    let idf = idf.unwrap_or_default();
    let idf = if idf.iter().all_equal() {
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
    };
    Ok((mat, idf))
}

/// feature weighting and L2 norm normalization.
fn normalize(input: &mut CsrMatrix<f64>, feature_weights: &[f64]) {
    input.row_iter_mut().par_bridge().for_each(|mut row| {
        let (indices, data) = row.cols_and_values_mut();
        indices
            .iter()
            .zip(data.iter_mut())
            .for_each(|(i, x)| *x *= feature_weights[*i]);

        let norm = data.iter().map(|x| x * x).sum::<f64>().sqrt();
        data.iter_mut().for_each(|x| *x /= norm);
    });
}

fn spmm_dense(i: usize, j: usize, mat: &CsrMatrix<f64>, dense: &DMatrix<f64>) -> DMatrix<f64> {
    assert!(i <= j, "Invalid row range, {}-{}", i, j);
    let mut result = DMatrix::zeros(j - i, dense.ncols());
    for row_idx in i..j {
        let row = mat.row(row_idx);
        for (col_idx, val) in row.col_indices().iter().zip(row.values().iter()) {
            for k in 0..dense.ncols() {
                result[(row_idx - i, k)] += val * dense[(*col_idx, k)];
            }
        }
    }
    result
}

fn compute_degrees<A: AnnDataOp>(
    adata: &A,
    selected_features: &SelectInfoElem,
    feature_weights: &[f64],
) -> Vec<f64> {
    let n = SelectInfoElemBounds::new(selected_features, adata.n_vars()).len();
    let mut col_sum = vec![0.0; n];

    // First pass to compute the sum of each column.
    adata.x().iter(5000).for_each(|x: (CsrMatrix<f64>, _, _)| {
        let mut mat = x.0.select_axis(1, selected_features);
        normalize(&mut mat, feature_weights);
        mat.row_iter().for_each(|row| {
            row.col_indices()
                .iter()
                .zip(row.values().iter())
                .for_each(|(i, x)| col_sum[*i] += x);
        });
    });
    let col_sum = DVector::from(col_sum);

    // Second pass to compute the degree.
    adata
        .x()
        .iter(5000)
        .flat_map(|x: (CsrMatrix<f64>, _, _)| {
            let mut mat = x.0.select_axis(1, selected_features);
            normalize(&mut mat, feature_weights);
            let v = &mat * &col_sum;
            v.into_iter().map(|x| *x - 1.0).collect::<Vec<_>>()
        })
        .collect::<Vec<_>>()
}

fn compute_probs(degrees: &[f64]) -> Vec<f64> {
    let s: f64 = degrees.iter().map(|x| x.recip()).sum();
    degrees.iter().map(|x| x.recip() / s).collect()
}

fn hstack(m1: CsrMatrix<f64>, m2: CsrMatrix<f64>) -> CsrMatrix<f64> {
    let c1 = m1.ncols();
    let vec = m1
        .row_iter()
        .zip(m2.row_iter())
        .map(|(r1, r2)| {
            let mut indices = r1.col_indices().to_vec();
            let mut data = r1.values().to_vec();
            indices.extend(r2.col_indices().iter().map(|x| x + c1));
            data.extend(r2.values().iter().map(|x| *x));
            indices
                .into_iter()
                .zip(data.into_iter())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let (r, c, offset, ind, data) = to_csr_data(vec, c1 + m2.ncols());
    CsrMatrix::try_from_csr_data(r, c, offset, ind, data).unwrap()
}

/// Multi-view spectral embedding.
#[pyfunction]
pub(crate) fn multi_spectral_embedding<'py>(
    py: Python<'py>,
    anndata: Vec<AnnDataLike>,
    selected_features: Vec<Bound<'_, PyAny>>,
    weights: Vec<f64>,
    n_components: usize,
    random_state: i64,
) -> Result<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray2<f64>>)> {
    info!("Compute normalized views...");
    let mats = anndata
        .into_iter()
        .zip(selected_features.into_iter())
        .map(|(a, s)| {
            macro_rules! get_mat {
                ($data:expr) => {{
                    let slice = pyanndata::data::to_select_elem(&s, $data.n_vars())
                        .expect("Invalid feature selection");
                    let mut mat: CsrMatrix<f64> = $data
                        .x()
                        .slice_axis::<DynCsrMatrix, _>(1, slice)
                        .unwrap()
                        .expect("X is None")
                        .try_convert()
                        .unwrap();
                    let feature_weights = idf(&mat);

                    // feature weighting and L2 norm normalization.
                    normalize(&mut mat, &feature_weights);
                    let norm = if mat.nrows() <= 2000 {
                        frobenius_norm(&mat)
                    } else {
                        frobenius_norm(&sample_csr(&mat, 2000))
                    };
                    anyhow::Ok((norm, mat))
                }};
            }
            crate::with_anndata!(&a, get_mat).unwrap()
        })
        .collect::<Vec<_>>();
    let ws = mats
        .iter()
        .map(|x| x.0)
        .zip(weights.iter())
        .map(|(n, w)| w / n)
        .collect::<Vec<_>>();
    let w_sum = ws.iter().sum::<f64>();
    let mat = mats
        .into_iter()
        .zip(ws.into_iter())
        .map(|((_, mut mat), w)| {
            let w = (w / w_sum).sqrt();
            mat.values_mut().iter_mut().for_each(|x| *x *= w);
            mat
        })
        .reduce(|a, b| hstack(a, b))
        .unwrap();

    info!("Compute embedding...");
    let (evals, evecs, _) = spectral_mf(mat, n_components, random_state)?;
    Ok((
        PyArray1::from_owned_array(py, evals),
        PyArray2::from_owned_array(py, evecs),
    ))
}

fn frobenius_norm(x: &CsrMatrix<f64>) -> f64 {
    let sum: f64 = Python::with_gil(|py| {
        let fun: Py<PyAny> = PyModule::from_code(
            py,
            c_str!(
                "def f(X):
                import numpy as np
                return np.power(X @ X.T, 2).sum()"
            ),
            c_str!(""),
            c_str!(""),
        )?
        .getattr("f")?
        .into();
        let args = (PyArrayData::from(ArrayData::from(x)),);
        fun.call1(py, args)?.extract(py)
    })
    .unwrap();
    (sum - x.nrows() as f64).sqrt()
}

fn sample_csr(mat: &CsrMatrix<f64>, n: usize) -> CsrMatrix<f64> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(2023);
    let idx = rand::seq::index::sample(&mut rng, mat.nrows(), n).into_vec();
    mat.select_axis(0, SelectInfoElem::from(idx))
}
