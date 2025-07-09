use criterion::{criterion_group, criterion_main, Criterion};
use nalgebra_sparse::CsrMatrix;
use rand::{rngs::StdRng, Rng, SeedableRng};
use snapatac2_core::embedding::{InverseDocumentFrequency, idf_from_chunks_parallel};

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

fn bench_idf(c: &mut Criterion) {
    let mut group = c.benchmark_group("IDF");
    group.sample_size(20);

    let rng = &mut StdRng::seed_from_u64(42);

    for n in [1000usize, 3000, 10000].into_iter() {
        let csr = random_csr_matrix(n, n, 0.5, rng);

        group.bench_with_input(format!("Noraml ({} x {})", n, n), &csr, |b, csr|
            b.iter(|| std::iter::once(csr.clone()).idf())
        );

        group.bench_with_input(format!("Parallel ({} x {})", n, n), &csr, |b, csr|
            b.iter(|| idf_from_chunks_parallel(std::iter::once(csr.clone())))
        );
    }
    group.finish();
}


criterion_group!(benches, bench_idf);
criterion_main!(benches);