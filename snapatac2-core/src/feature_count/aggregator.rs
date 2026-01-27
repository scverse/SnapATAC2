use anndata::{data::ArrayConvert, AnnDataOp, ArrayData, ArrayElemOp};
use anyhow::{ensure, Result};
use indexmap::IndexSet;
use nalgebra_sparse::CsrMatrix;
use ndarray::Array2;

pub fn aggregate_x<A: AnnDataOp>(
    adata: &A,
    groupby: Option<&[Option<String>]>,
) -> Result<(Option<Vec<String>>, Array2<f64>)> {
    let n_vars = adata.n_vars();
    if let Some(groupby) = groupby {
        ensure!(
            groupby.len() == adata.n_obs(),
            "Number of observations in groupby must match number of observations in adata"
        );
        let groups: IndexSet<String> = groupby.iter().flatten().map(|x| x.to_string()).collect();
        let mut result = Array2::<f64>::zeros((groups.len(), n_vars));
        adata
            .x()
            .iter::<ArrayData>(5000)
            .for_each(|(chunk, pos, _)| match chunk {
                ArrayData::Array(arr) => {
                    let arr: Array2<f64> = arr.try_convert().unwrap();
                    arr.axis_iter(ndarray::Axis(0))
                        .enumerate()
                        .for_each(|(i, row)| {
                            if let Some(g) = &groupby[pos+i] {
                                let i = groups.get_index_of(g).unwrap();
                                row.iter().enumerate().for_each(|(j, v)| {
                                    result[(i, j)] += *v;
                                });
                            }
                        });
                }
                ArrayData::CsrMatrix(csr) => {
                    let csr: CsrMatrix<f64> = csr.try_convert().unwrap();
                    for (i, row) in csr.row_iter().enumerate() {
                        if let Some(g) = &groupby[pos+i] {
                            let i = groups.get_index_of(g).unwrap();
                            row.col_indices()
                                .iter()
                                .zip(row.values().iter())
                                .for_each(|(j, v)| {
                                    result[(i, *j)] += *v;
                                });
                        }
                    }
                }
                _ => panic!("Unsupported array data type"),
            });
        Ok((Some(groups.into_iter().collect()), result))
    } else {
        let mut result = vec![0.0; n_vars];
        adata
            .x()
            .iter::<ArrayData>(5000)
            .for_each(|(chunk, _, _)| match chunk {
                ArrayData::Array(arr) => {
                    let arr: Array2<f64> = arr.try_convert().unwrap();
                    arr.axis_iter(ndarray::Axis(0)).for_each(|row| {
                        row.iter().enumerate().for_each(|(i, v)| {
                            result[i] += *v;
                        });
                    });
                }
                ArrayData::CsrMatrix(csr) => {
                    let csr: CsrMatrix<f64> = csr.try_convert().unwrap();
                    for row in csr.row_iter() {
                        row.col_indices()
                            .iter()
                            .zip(row.values().iter())
                            .for_each(|(i, v)| {
                                result[*i] += *v;
                            });
                    }
                }
                _ => panic!("Unsupported array data type"),
            });
        Ok((None, Array2::from_shape_vec((1, adata.n_vars()), result).unwrap()))
    }
}