# IDF Functions Mathematical Equivalence Test

This directory contains a comprehensive test to verify that `idf_from_chunks` and `idf_from_chunks_parallel` produce mathematically identical results.

## Running the Tests

### Method 1: Run as executable
```bash
cd idf_test
cargo run
```

### Method 2: Run as unit tests
```bash
cd idf_test
cargo test
```

## Test Coverage

The test suite covers:

1. **Random matrices** with varying densities (5%, 10%, 30%, 70%)
2. **Uniform matrices** where all columns have the same count (edge case)
3. **Sparse matrices** with some columns having zero counts (edge case) 
4. **Multiple chunks** of different sizes
5. **Large matrices** (1000x200)

## Expected Output

If the functions are mathematically equivalent, you should see:

```
Testing mathematical equivalence of idf_from_chunks vs idf_from_chunks_parallel
Tolerance: 1.00e-12

Test 1: Random matrices with varying density
✓ Random (density=0.05): Vectors are mathematically identical (max diff < 1.00e-12)
✓ Random (density=0.1): Vectors are mathematically identical (max diff < 1.00e-12)
✓ Random (density=0.3): Vectors are mathematically identical (max diff < 1.00e-12)
✓ Random (density=0.7): Vectors are mathematically identical (max diff < 1.00e-12)

Test 2: Uniform matrix (all columns have same count)
✓ Uniform matrix: Vectors are mathematically identical (max diff < 1.00e-12)

... (more tests)

Both idf_from_chunks and idf_from_chunks_parallel produce identical results.
```

## Summary

The cleaned up code removes all unused imports and focuses only on the core functionality needed for IDF calculation. The test comprehensively verifies that both the sequential and parallel implementations produce identical mathematical results across various edge cases and input sizes.
