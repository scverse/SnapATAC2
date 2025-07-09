use itertools::Itertools;
use nalgebra_sparse::CsrMatrix;
use rayon::prelude::{ParallelBridge, ParallelIterator};
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

/// Sequential IDF computation
fn idf_from_chunks<I>(input: I) -> Vec<f64>
where
    I: IntoIterator<Item = CsrMatrix<f64>>,
{
    let mut iter = input.into_iter().peekable();
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

/// Parallel IDF computation
fn idf_from_chunks_parallel<I>(input: I) -> Vec<f64>
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

/// Generate a random binary CSR matrix with given shape and density
fn random_csr_matrix(rows: usize, cols: usize, density: f64, rng: &mut StdRng) -> CsrMatrix<f64> {
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    
    indptr.push(0);
    for _ in 0..rows {
        let mut row_indices = Vec::new();
        for j in 0..cols {
            if rng.gen::<f64>() < density {
                row_indices.push(j);
            }
        }
        indices.extend(&row_indices);
        data.extend(std::iter::repeat(1.0).take(row_indices.len()));
        indptr.push(indices.len());
    }
    
    CsrMatrix::try_from_csr_data(rows, cols, indptr, indices, data).unwrap()
}

/// Create a matrix with all columns having the same count (edge case test)
fn uniform_csr_matrix(rows: usize, cols: usize) -> CsrMatrix<f64> {
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    
    indptr.push(0);
    for _ in 0..rows {
        // Each row has all columns filled
        indices.extend(0..cols);
        data.extend(std::iter::repeat(1.0).take(cols));
        indptr.push(indices.len());
    }
    
    CsrMatrix::try_from_csr_data(rows, cols, indptr, indices, data).unwrap()
}

/// Create a matrix with some columns having zero counts (edge case test)
fn sparse_csr_matrix(rows: usize, cols: usize) -> CsrMatrix<f64> {
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    
    indptr.push(0);
    for _ in 0..rows {
        // Only fill first half of columns
        indices.extend(0..(cols/2));
        data.extend(std::iter::repeat(1.0).take(cols/2));
        indptr.push(indices.len());
    }
    
    CsrMatrix::try_from_csr_data(rows, cols, indptr, indices, data).unwrap()
}

fn assert_vectors_equal(a: &[f64], b: &[f64], tolerance: f64, test_name: &str) {
    assert_eq!(a.len(), b.len(), "{}: Vector lengths differ", test_name);
    
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let diff = (x - y).abs();
        assert!(
            diff < tolerance,
            "{}: Mismatch at index {}: sequential={:.12}, parallel={:.12}, diff={:.2e}",
            test_name, i, x, y, diff
        );
    }
    println!("✓ {}: Vectors are mathematically identical (max diff < {:.2e})", test_name, tolerance);
}

fn main() {
    let tolerance = 1e-12;
    
    println!("Testing mathematical equivalence of idf_from_chunks vs idf_from_chunks_parallel");
    println!("Tolerance: {:.2e}\n", tolerance);
    
    // Test 1: Random matrices with different densities
    println!("Test 1: Random matrices with varying density");
    let mut rng = StdRng::seed_from_u64(42);
    let densities = [0.05, 0.1, 0.3, 0.7];
    
    for &density in &densities {
        let matrices: Vec<CsrMatrix<f64>> = (0..5)
            .map(|_| random_csr_matrix(100, 50, density, &mut rng))
            .collect();
        
        let idf_seq = idf_from_chunks(matrices.clone());
        let idf_par = idf_from_chunks_parallel(matrices);
        
        assert_vectors_equal(&idf_seq, &idf_par, tolerance, &format!("Random (density={})", density));
    }
    
    // Test 2: Edge case - uniform matrix (all columns have same count)
    println!("\nTest 2: Uniform matrix (all columns have same count)");
    let uniform_matrices = vec![uniform_csr_matrix(50, 30)];
    let idf_seq = idf_from_chunks(uniform_matrices.clone());
    let idf_par = idf_from_chunks_parallel(uniform_matrices);
    
    // Should return vector of 1.0s due to all_equal() check
    assert!(idf_seq.iter().all(|&x| (x - 1.0).abs() < tolerance), "Sequential: Expected all 1.0s");
    assert!(idf_par.iter().all(|&x| (x - 1.0).abs() < tolerance), "Parallel: Expected all 1.0s");
    assert_vectors_equal(&idf_seq, &idf_par, tolerance, "Uniform matrix");
    
    // Test 3: Edge case - sparse matrix with zero columns
    println!("\nTest 3: Sparse matrix (some columns have zero counts)");
    let sparse_matrices = vec![sparse_csr_matrix(40, 20)];
    let idf_seq = idf_from_chunks(sparse_matrices.clone());
    let idf_par = idf_from_chunks_parallel(sparse_matrices);
    
    assert_vectors_equal(&idf_seq, &idf_par, tolerance, "Sparse matrix");
    
    // Test 4: Multiple chunks of different sizes
    println!("\nTest 4: Multiple chunks with different sizes");
    let mut rng = StdRng::seed_from_u64(123);
    let chunk_sizes = [20, 50, 30, 80, 15];
    let matrices: Vec<CsrMatrix<f64>> = chunk_sizes.iter()
        .map(|&size| random_csr_matrix(size, 40, 0.15, &mut rng))
        .collect();
    
    let idf_seq = idf_from_chunks(matrices.clone());
    let idf_par = idf_from_chunks_parallel(matrices);
    
    assert_vectors_equal(&idf_seq, &idf_par, tolerance, "Multiple chunks");
    
    // Test 5: Single large matrix
    println!("\nTest 5: Single large matrix");
    let large_matrix = vec![random_csr_matrix(1000, 200, 0.05, &mut StdRng::seed_from_u64(999))];
    let idf_seq = idf_from_chunks(large_matrix.clone());
    let idf_par = idf_from_chunks_parallel(large_matrix);
    
    assert_vectors_equal(&idf_seq, &idf_par, tolerance, "Large matrix");
    
    println!("\nBoth idf_from_chunks and idf_from_chunks_parallel produce identical results.");
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_equivalence_comprehensive() {
        let mut rng = StdRng::seed_from_u64(42);
        let matrices: Vec<CsrMatrix<f64>> = (0..3)
            .map(|_| random_csr_matrix(50, 30, 0.1, &mut rng))
            .collect();
        
        let idf_seq = idf_from_chunks(matrices.clone());
        let idf_par = idf_from_chunks_parallel(matrices);
        
        for (i, (a, b)) in idf_seq.iter().zip(idf_par.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-12,
                "Mismatch at index {}: sequential={}, parallel={}",
                i, a, b
            );
        }
    }
}
