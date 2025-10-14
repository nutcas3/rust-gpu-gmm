# Data Directory

This directory contains sample input/output matrices for testing and benchmarking the GEMM kernel.

## File Format

Binary matrix files are stored in row-major order as 32-bit floating-point values.

### File Naming Convention
- `input_A_<M>x<K>.bin` - Input matrix A (M rows, K columns)
- `input_B_<K>x<N>.bin` - Input matrix B (K rows, N columns)
- `output_C_<M>x<N>.bin` - Output matrix C (M rows, N columns)

## Generating Test Data

You can generate test matrices using the provided utility (to be implemented) or manually:

### Using Python
```python
import numpy as np

# Generate random matrices
M, N, K = 1024, 1024, 1024
A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(K, N).astype(np.float32)

# Save to binary files
A.tofile('input_A_1024x1024.bin')
B.tofile('input_B_1024x1024.bin')
```

### Using Rust
```rust
use std::fs::File;
use std::io::Write;

fn generate_matrix(rows: usize, cols: usize, filename: &str) {
    let data: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32).sin())
        .collect();
    
    let bytes: Vec<u8> = data.iter()
        .flat_map(|&f| f.to_le_bytes())
        .collect();
    
    let mut file = File::create(filename).unwrap();
    file.write_all(&bytes).unwrap();
}
```

## Standard Test Cases

### Small (for debugging)
- 16x16x16
- 32x32x32
- 64x64x64

### Medium (for development)
- 256x256x256
- 512x512x512
- 1024x1024x1024

### Large (for benchmarking)
- 2048x2048x2048
- 4096x4096x4096
- 8192x8192x8192

### Non-square (for edge cases)
- 1024x2048x512
- 512x4096x1024
- 2048x512x2048

## Memory Requirements

| Matrix Size | Memory per Matrix | Total (A+B+C) |
|-------------|-------------------|---------------|
| 1024x1024   | 4 MB             | 12 MB         |
| 2048x2048   | 16 MB            | 48 MB         |
| 4096x4096   | 64 MB            | 192 MB        |
| 8192x8192   | 256 MB           | 768 MB        |

## Notes

- All matrices use single-precision (FP32) floating-point format
- Row-major layout is assumed unless otherwise specified
- For very large matrices, consider using memory-mapped files
